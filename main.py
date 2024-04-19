import argparse
import logging
import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from utils.dataset import Crack
from utils.dataset import random_seed
from models.unet import UNet
from utils.losses import CrossEntropyLoss, DiceCELoss, DiceLoss, FocalLoss, SiLogLoss
from utils.metrics import pix_acc, iou

def strip_optimizers(f: str, s =''):  
    """Strip optimizer from 'f' to finalize training, optionally save as 's' """

    x = torch.load(f, map_location=torch.device('cpu')) 
    if x.get('ema'): 
        x['model'] = x['ema']  # replace model with ema 
    for k in 'optimizer', 'best_score':  # keys 
        x[k] = None 
    x['epoch'] = -1 
    x['model'].half()  # to FP16 
    for p in x['model'].parameters(): 
        p.requires_grad = False 
    torch.save(x, s or f) 
    mb = os.path.getsize(s or f) / 1E6  # get file size 
    logging.info(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB") 

def load_dataset(opt):
    # Dataset
    train_dataset = Crack(root=f"{opt.data}/train", image_size=opt.image_size, mask_suffix="")
    test_dataset = Crack(root=f"{opt.data}/test", image_size=opt.image_size,mask_suffix="")

    # Split
    n_val = int(len(train_dataset) * 0.1)
    n_train = len(train_dataset) - n_val
    train_data, val_data = random_split(train_dataset, [n_train, n_val])

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=8, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=8, drop_last=False, pin_memory=True)

    logging.info(f"TRAIN SIZE: , {len(train_loader.dataset)}")
    logging.info(f"VALADATION SIZE: , {len(val_loader.dataset)}")
    logging.info(f"TEST SIZE: , {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader    

def train(opt, model, device, criterion):
    """Train model with train data
    Args:
        opt: options to train
        model (torch.nn.Module): model to train        
        device (str): device to train model ('cpu' or 'cuda')
        criterion (torch.nn.Module): loss function       
    """
    best_score = 0.0
    best, last = f"{opt.save_dir}/best.pt", f"{opt.save_dir}/last.pt"

    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)   
    
    model.to(device)    
    
    train_loader, val_loader, _ = load_dataset(opt)
    
    for epoch in range(0, opt.epochs):
        model.train()
        epoch_loss = 0.0

        logging.info(("\n" + "%12s" * 3) % ("Epochs", "GPU Mem", "Loss"))
        progress_bar = tqdm(train_loader, total=len(train_loader))
        for image, target in progress_bar:
            image, target = image.to(device), target.to(device)

            with torch.cuda.amp.autocast(enabled=opt.amp):
                output = model(image)                
                loss = criterion(output, target)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            progress_bar.set_description(("%12s" * 2 + "%12.4g") % (f"{epoch + 1}/{opt.epochs}", mem, loss))
            
        dice_score, dice_loss, class_iou, batch_iou, pixel_acc = validation(opt, model, val_loader, device, criterion)
        logging.info(f"VALIDATION: Dice Score: {dice_score:.4f}, Dice Loss: {dice_loss:.4f}, Classes_IOU: {class_iou}, Batch_IOU: {batch_iou}, Pixel Accuracy: {pixel_acc}")

        scheduler.step(epoch)

        ckpt = {
            "epoch": epoch,
            "best_score": best_score,
            "model": deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(ckpt, last)
        if best_score < dice_score:
            best_score = max(best_score, dice_score)
            torch.save(ckpt, best)

    # Strip optimizers & save weights
    for f in best, last:
        strip_optimizers(f)

@torch.inference_mode()
def validation(opt, model, data_loader, device, criterion):
    """Evaluate model performance with validation data
    Args:
        opt: options to evaluate
        model (torch.nn.Module): model to evaluate        
        data_loader (object): iterator to load data
        device (str): device to evaluate model ('cpu' or 'cuda')
        criterion (torch.nn.Module): loss function       
    """
    model.eval()
    dice_score = 0
    n = len(data_loader.dataset)
    class_iou = [0.] * opt.num_classes
    pixel_acc = 0.0    
   
    for image, target in tqdm(data_loader, total=len(data_loader)):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)
            if model.out_channels == 1:
                output = F.sigmoid(output) > opt.conf_threshold
            dice_loss = criterion(output, target)
            dice_score += 1 - dice_loss

            pred = torch.argmax(output, dim=1)
            batch_size = image.shape[0]
            pred = pred.view(batch_size, -1)
            target = target.squeeze(1).long()  # remove channel dimension
            target = target.view(batch_size, -1)
            batch_iou = iou(pred, target, batch_size, opt.num_classes)
            class_iou += batch_iou * (batch_size / n)
            pixel_acc += pix_acc(pred, target, batch_size) * (batch_size / n)            
  
    return dice_score / len(data_loader), dice_loss, class_iou, batch_iou, pixel_acc

def parse_opt():
    parser = argparse.ArgumentParser(description="Crack Segmentation Training Arguments")
    parser.add_argument("--data", type=str, default="./data", help="Path to root folder of data")
    parser.add_argument("--image-size", type=int, default=448, help="Input image size, default: 448")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs, default: 10")
    parser.add_argument("--weights", type=str, default="./weights/best.pt", help="Initial weights path, default : ./weights/best.pt")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size, default: 4")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate, default: 1e-5")   
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold, default: 0.4")
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Specify whether to run in "train" or "test" mode (default: "train")')

    return parser.parse_args()

def main(opt):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    model = UNet(in_channels=3, out_channels=opt.num_classes).to(device)

    logging.info(
        f"Network:\n"
        f"\t{model.in_channels} input channels\n"
        f"\t{model.out_channels} output channels (number of classes)"
    )
    random_seed()

    # Create folder to save weights
    os.makedirs(opt.save_dir, exist_ok=True)

    criterion = DiceCELoss() # SiLogLoss() #CrossEntropyLoss() # DiceLoss() 
    ## Compared to cross-entropy loss, dice loss is very robust against imbalanced segmentation mask.##
    
    if opt.mode == "train":
        train(opt, model, device, criterion)

    elif opt.mode == "test":
        assert os.path.isfile(opt.weights), f"Inputthe path to the trained model, opt.weight: {opt.weights}"

        ckpt = torch.load(opt.weights, map_location=device)       
        _, _, test_loader = load_dataset(opt)
        model.load_state_dict(ckpt["model"].float().state_dict())
        dice_score, dice_loss, class_iou, batch_iou, pixel_acc = validation(opt, model, test_loader, device, criterion)
        logging.info(f"TEST: Dice Score: {dice_score:.4f}, Dice Loss: {dice_loss:.4f}, Classes_IOU: {class_iou}, Batch_IOU: {batch_iou}, Pixel Accuracy: {pixel_acc}")
    
    else:
        raise ValueError(f"Just select `test` or `train` mode, your input: {opt.mode}")

if __name__ == "__main__":
    params = parse_opt()
    main(params)

