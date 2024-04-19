import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from models import UNet

# Preprocess
def preprocess(image, is_mask):
    """Preprocess image and mask"""
    img_ndarray = np.asarray(image)
    if not is_mask:
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255.0

    return img_ndarray

# Inference
def inference(image):
    with torch.no_grad():
        output = model(image).cpu()
        
        if model.out_channels > 1:
            output = output.argmax(dim=1)
        else:
            output = torch.sigmoid(output) > 0.4 # out_threshold

    mask = output[0].long().squeeze().numpy()

    return mask

# Path to weigth and image file
model_weight = "./weights/best.pt"
input_path = "./data/test/images/CRACK500_20160222_081908_641_721.jpg"

# Load weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_weight, map_location=device)

# Initialize model and load checkpoint
model = UNet(in_channels=3, out_channels=2)
model.load_state_dict(checkpoint["model"].float().state_dict())

input_image = Image.open(input_path)
model.eval()
model.to(device)

# Preprocess 
image = torch.from_numpy(preprocess(input_image, is_mask=False))
image = image.unsqueeze(0)
image = image.to(device, dtype=torch.float32)

mask = inference(image)

# Postprocessing
result = Image.fromarray((mask * 255).astype(np.uint8))

# Show the prediction mask
plt.imshow(np.array(result))
plt.axis('off')
plt.show()

# Show the input image
plt.imshow(np.array(input_image))
plt.axis('off')
plt.show()