# UNet: Road Crack Segmentation implematation in PyTorch

|    Metrics    | DiceCE Loss | Dice Loss |  CE Loss | Focal Loss | SiLog Loss |
|:-------------:|:-----------:|:---------:|:--------:|:----------:|:----------:|
|   Dice score  |  0.9334     | 0.9824    |    0.9571      |   0.9950         |   0.8490 |
| Pixel Accuracy| 0.98221      |  0.9823  |      0.9821    |   0.9815    |   0.9821  |   
| Classes_IOU   |0.9816 0.4136 |0.9817 0.4259 | 0.9815 0.4195 | 0.9809 0.4643 | 0.9815 0.4227 | 
|  Batch_IOU    |0.9826 0.4659|0.9833 0.4353| 0.9737 0.4376  | 0.9818 0.5343 | 0.9825 0.4557 |            


Road crack segmentation is the task of identifying and segmenting road cracks in images or videos of roads. In this
project, [UNet](https://arxiv.org/abs/1505.04597v1) is applied to segment the cracks on road.

## Table of Contents

* [Project Description](#project-description)
* [Installation](#installation)
* [Usage](#usage)
* [Reference](#reference)
* [License](#license)

## Project Description

In our Road Crack Segmentation project, we successfully deployed the UNet model for accurately segmenting road cracks utilizing dataset 
[Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset). 
We assessed the model's performance by employing various loss functions and conducting comparative analysis. The implemented loss functions include:

- [**Cross Entropy Loss**](./utils/losses.py)
- [**Dice Loss**](./utils/losses.py)
- [**Scale-Invariant Logarithmic Loss**](./utils/losses.py)
- [**Dice Cross Entropy Loss**](./utils/losses.py)
- [**Focal Loss**](./utils/losses.py)

We trained the model using the aforementioned loss functions and assessed their performance through the **Dice coefficient** (`dice
score = 1 - dice loss`)

## Installation

Download the project:

```commandline
git clone https://github.com/jabborov/road-crack-pothole-segmentation.git
cd road-crack-pothole-segmentation
```

Install requirements:

```commandline
pip install -r requirements
```

## Usage

### Dataset

To train the model, download the dataset and organize it by placing the train and test folders within a parent directory named data, structured as follows:

```
data-|
     |-train-|
             |-images
             |-masks

                
     |-test -|
             |-images
             |-masks
```

### Train

```commandline
python -m main
```
Training arguments:
```
usage: main.py [-h] [--data DATA] [--image-size IMAGE_SIZE] [--save-dir SAVE_DIR] [--epochs EPOCHS] [--weights WEIGHTS] [--amp] [--num-classes NUM_CLASSES] [--batch-size BATCH_SIZE] [--lr LR]
               [--conf-threshold CONF_THRESHOLD] [--mode {train,test}]

Crack Segmentation Training Arguments

options:
  -h, --help            show this help message and exit
  --data DATA           Path to root folder of data
  --image-size IMAGE_SIZE
                        Input image size, default: 448
  --save-dir SAVE_DIR   Directory to save weights
  --epochs EPOCHS       Number of epochs, default: 10
  --weights WEIGHTS     Initial weights path, default : ./weights/best.pt
  --amp                 Use mixed precision
  --num-classes NUM_CLASSES
                        Number of classes
  --batch-size BATCH_SIZE
                        Batch size, default: 4
  --lr LR               Learning rate, default: 1e-5
  --conf-threshold CONF_THRESHOLD
                        Confidence threshold, default: 0.4
  --mode {train,test}   Specify whether to run in "train" or "test" mode (default: "train")
```

### Inference
```commandline
python -m demo.py
```

## Reference
[Crack Segmentation](https://github.com/yakhyo/crack-segmentation/tree/main)

## License
The project is licensed under the [MIT license](https://github.com/jabborov/road-crack-pothole-segmentation/blob/main/LICENSE).
