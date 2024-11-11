# Cardiac Arrhythmia Detection Model
This repository contains pre-trained models for detecting and classifying Ventricular Arrhythmias (Ventricular Fibrillation/VF and Ventricular Tachycardia/VT) from normal sinus rhythms (SR) in ECG recordings.

## Contents
- Training records including loss curves, training logs, confusion matrices, feature importance plots, and reconstructed inputs for each encoder.
- Model Checkpoints of encoders in .H5 format. Classifiers are publicly-available in cloud storage, due to file sizes exceeding GitHub limits.

## Installation
Please install all necessary requirements:
```bash
pip install -r requirements.txt
```

## Examples
Run inference on a pre-trained model, optionally on your custom data:
```bash
python inference.py conv 32 lead_ii imbalanced path/to/your/data
```
Replicate training of a new model on the same provided data
```bash
python train.py transf 64 12-lead balanced #
```

When supplying your own data for evaluating pretrained models, ensure that the data is in `.MAT` format with the following structure:
```data[rhythm_label][0][recording_num][timestep][voltage_reading][channel]```
-With rhythm labels: `{'VT', 'VF, 'SR'}` and sampled at 100Hz. Please edit load_datasets.load_custom_dataset() to accommodate any deviations from this format.

## Arguments
Each model comprises an encoder and a random forest classifier. Use the following arguments to select one of 36 models which vary on:
* The encoder's model architecture: CNN, Transformer or GRU (referred to as 'conv', 'transf', 'gru')
* Size of the encoding vector (32, 64 or 128)
* Whether the encoder was trained on ECG data from only lead-II (commonly used in wearable devices) or all 12 leads; specified as 'lead_ii' or '12_lead'
* Whether the random forest classifier was trained on a balanced set with roughly equal representation of each class, or a regular training dataset with an overrepresented 'SR' class, specified as 'balanced' or 'imbalanced'
18 encoders were trained and then 2 classifiers for each encoder, resulting in 36 total.

## Computing and Memory requirements for training
Each encoder model has approximately 300k trainable parameters. Datasets are considerably large and training requires a download size of 5.5GB (for 12-lead data) or 45GB (for lead-II data).
GPU use is recommended, as training time for each model was between 30min and 3 hours on an NVIDIA L4 Tensor Core GPU with 22.5GB RAM

## Datasets used:
* Encoder training: [CODE-15% Dataset](https://zenodo.org/records/4916206) [[Ribeiro et al 2021]](https://www.nature.com/articles/s41467-020-15432-4) comprises 300k recordings (or "exams") from 200k patients at approx 10 seconds each. All recordings are 12-lead. Total size is 45GB.
* Classification training: [Alwan & Cvetkovic 2017](https://ieeexplore.ieee.org/abstract/document/8231165), consisting of 55k recordings at approximately 8 seconds each, composed of two channels where Lead-II was used for  classification training. Total size is 160MB.

## Motivation
It is difficult and costly to gather labelled, high-quality ECG data on ventricular arrhythmias, which hinders classifier training. Therefore this project uses a self-supervised approach.
![image](https://github.com/user-attachments/assets/8c561172-7da5-4c60-8020-d8fb4f9a6b0f)
Firstly a large, unlabeled ECG dataset is used to train an autoregressive encoder-decoder. The encoder alone can produce encodings which preserve meaningful information on the input ECG waveform.
The encoder is then used with a smaller, high-quality arrhythmia dataset to train a classifier. The encoder leverages the large-scale and diverse dataset for improved results on the downstream classification task. 
