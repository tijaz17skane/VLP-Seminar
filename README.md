# VLP-Semniar
This framework contains code for finetune the pretrain-model to downstream tasks: classfication, segmentation and detection.
The code is based on MGCA's code. Really thank you for them:)


# Structure of the Code
```
.
├── annotations (annotations store the ouputs of the preprocess, the annotations for each datasts.)
├── configs # config for each dataset (chexpert.yaml, rsna.yaml)
├── data # Ouput for the model (ckpts, log_outputs)
├── Finetune # 2. Main code for Finetune
├── preprocess_datasets # 1. preprocess the downstream dataset
└── README.md 
```


# Preprocess Datasets.
## rsna datast
Intro to rsna dataset
- annotaions: image, boungdingbox, label
- can use for tasks: classfication, detection and segmentation (use box as the mask)

Download the rsna datasets
- follow mgca setup, we download the dataset form https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- u can directly

```
mkdir ~/datasets/rsna
cd datasets/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge -d ./
```

Preprocess the dataset format
- details could be fine in rsna.ipynb under process_datasets folder
- ouputs will be in annotations/rsna/
    - train.csv, val.csv. test.csv

## chexkpert datset
Intro to chexkpert dataet
- annotations for 14 diesease, with lable 0, 1, -1 (0 absent, 1 exisit, -1 uncertain)
- use for classfication

Download the chexkpert datasets
- I use the link via kaggle: https://www.kaggle.com/datasets/ashery/chexpert
- u can directly via command:

```
mkdir ~/datasets/chexpert
cd datasets/chexpert
kaggle datasets download ashery/chexpert
unzip chexpert-v10-small.zip -d ./
```