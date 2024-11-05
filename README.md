# VLP finetune code for AI for Vision-Language Models in Medical Imaging (IN2107)

**This repository contains code for the "AI for Vision-Language Models in Medical Imaging (IN2107)." For more information, visit the [VLP Seminar page](https://compai-lab.github.io/teaching/vlm_seminar/).**

The code is designed to fine-tune Vision-Language Pre-trained models for downstream tasks, including classification, segmentation, and detection. You can also develop additional downstream tasks based on this repository :)

This project is built upon the code from MGCA: [HKU-MedAI GitHub Repository](https://github.com/HKU-MedAI). A special thanks to them for their contributions~


# Structure of the Repository
Here are the base strucres of our repository.
```
.
├── annotations # Stores the outputs of the preprocessing and annotations for each dataset.
├── configs # Configuration files for each dataset (e.g., chexpert.yaml, rsna.yaml).
├── data # Outputs for the model (checkpoints, log outputs).
├── Finetune # Main code for fine-tuning the models.
├── preprocess_datasets # Code to preprocess the downstream datasets.
└── README.md
```


Here’s a polished version of your "Preprocess Datasets" section for the README, focusing on the RSNA dataset. I’ve improved the grammar, clarity, and formatting:


# Preprocess Datasets
Here give examples for two dataset: rsna and also chexkpert.

## RSNA Dataset
The RSNA dataset includes:
- **Annotations**: Image, bounding box, and label.
- **Use Cases**: Suitable for tasks such as classification, detection, and segmentation (using bounding boxes as masks).

### Download the RSNA Dataset
To download the dataset, follow the MGCA setup instructions from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data):

```bash
mkdir ~/datasets/rsna
cd ~/datasets/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d ./
```

### Preprocess the Dataset Format
Details can be found in `rsna.ipynb` under the `preprocess_datasets` folder. The outputs will be saved in `annotations/rsna/`:
- `train.csv`
- `val.csv`
- `test.csv`


## Chexpert Dataset

The Chexpert dataset includes:
- **Annotations**: Covers 14 diseases, with labels of 0, 1, and -1 (where 0 indicates absence, 1 indicates presence, and -1 indicates uncertainty).
- **Use Cases**: Primarily for classification tasks.

### Download the Chexpert Dataset
- You can download the dataset from Kaggle using the following link: [Chexpert Dataset](https://www.kaggle.com/datasets/ashery/chexpert). 
- Alternatively, you can download it directly via command line:

```bash
mkdir ~/datasets/chexpert
cd ~/datasets/chexpert
kaggle datasets download ashery/chexpert
unzip chexpert-v10-small.zip -d ./
```


# Finetune
Note: The training part of the code is based on PyTorch Lightning. If you are not familiar with PyTorch Lightning, you can look at the introduction here: [PyTorch Lightning Introduction](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) 📚.

With PyTorch Lightning, there is no need to write the trainer yourself; you can directly use the traineer in Lightning module ⚡.

## Code Structure for Finetune
The code structure for fine-tuning is divided into two main parts:
- **datasets**: Main part for loading different datasets.
- **methods**: Main part for different methods.
```
.Finetune
├── datasets # main part for load different datasets.
│   ├── cls_dataset.py
│   ├── data_module.py
│   ├── det_dataset.py
│   ├── __init__.py
│   ├── seg_dataset.py
│   ├── transforms.py
│   └── utils.py
|....
```
```
.Finetune
├── methods # main parts for different methods
│   ├── backbones # contains the backbones need for methods.
│   │   ├── cnn_backbones.py
│   │   ├── detector_backbone.py
│   │   ├── encoder.py
│   │   ├── __init__.py
│   │   ├── med.py
│   │   ├── transformer_seg.py
│   │   └── vits.py
│   ├── cls_model.py  # model for classfcation
│   ├── det_model.py # model for detection
│   ├── __init__.py
│   ├── seg_model.py # model for segmentation.
│   └── utils # some losses, and utils,
│       ├── box_ops.py
│       ├── detection_utils.py
│       ├── __init__.py
│       ├── segmentation_loss.py
│       └── yolo_loss.py
|....
```

## Example for Fine-tuning
Here we provide an example based on the RSNA dataset. You just need to modify the [rsna.yaml](configs/rsna.yaml) file for different tasks.

The `rsna.yaml` file contains four parts:
- **dataset**: Base information for the dataset.
- **cls**: Configuration for classification.
- **det**: Configuration for detection.
- **seg**: Configuration for segmentation.
```
dataset: # base information for setting the dataset
  img_size: 224 # the input size will be 244
  dataset_dir: /u/home/lj0/datasets/RSNA_Pneumoni # dataset base dir
  train_csv: /u/home/lj0/Code/VLP-Seminars/annotations/rsna/train.csv # annotations path for train, test, val
  valid_csv: /u/home/lj0/Code/VLP-Seminars/annotations/rsna/val.csv
  test_csv: /u/home/lj0/Code/VLP-Seminars/annotations/rsna/test.csv
  
cls:
  img_size: 224
  backbone: resnet_50 #resnet and vit are supported (backbone u want to test)
  multilabel: False # whether the classfication task is a multilabel task
  embed_dim: 128 
  in_features: 2048 
  num_classes: 2 # classfication class numbers
  pretrained: True # whether utlize pretrain model to initilize
  freeze: True # whether freeze the entire backbone. 
  checkpoint: /home/june/Code/MGCA-main/data/ckpts/resnet_50.ckpt # initilize checkpint path
  lr: 5.0e-4 # lr for classfication
  dropout: 0.0 # dropout
  weight_decay: 1.0e-6 # weight_decay

det:
    img_size: 224
    backbone: resnet_50 #only resnet_50 is supported
    lr: 5.0e-4
    weight_decay: 1.0e-6
    conf_thres: 0.5 # confidence thereshold 0.5
    iou_thres: [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75] # iou thres for yolo
    nms_thres: 0.5
    pretrained: True
    freeze: True
    max_objects: 10
    checkpoint: /home/june/Code/MGCA-main/data/ckpts/resnet_50.ckpt
  
seg:
    img_size: 224 #224 for vit
    backbone: vit_base #vit_base and resnet_50
    lr: 2e-4
    weight_decay: 0.05
    pretrained: True
    freeze: True
    embed_dim: 128  
    checkpoint: /home/june/Code/MGCA-main/data/ckpts/vit_base.ckpt

```

### 1) Classification (Finetune)
```bash
python train_cls.py --batch_size 46 --num_workers 16 --max_epochs 50 --config ../configs/chexpert.yaml --gpus 1 --dataset chexpert --data_pct 1 --ckpt_dir ../data/ckpts --log_dir ../data/log_output
```
Alternatively, you can directly run:
```bash
python train_cls.py
```

### 2) Detection (Finetune)
```bash
python train_det.py --batch_size 32 --num_workers 16 --max_epochs 50 --config ../configs/rsna.yaml --gpus 1 --dataset rsna --data_pct 1 --ckpt_dir ../data/ckpts --log_dir ../data/log_output
```
Alternatively, you can directly run:
```bash
python train_det.py
```

### 3) Segmentation (Finetune)
```bash
python train_seg.py --batch_size 48 --num_workers 4 --max_epochs 50 --config ../configs/rsna.yaml --gpus 1 --dataset rsna --data_pct 1 --ckpt_dir ../data/ckpts --log_dir ../data/log_output
```
Alternatively, you can directly run:
```bash
python train_seg.py
```

