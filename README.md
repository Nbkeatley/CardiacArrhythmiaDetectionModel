# Cardiac Arrhythmia Detection Model
This repository contains pre-trained models for detecting and classifying Ventricular Arrhythmias (Ventricular Fibrillation/VF and Ventricular Tachycardia/VT) from normal sinus rhythms (SR) in ECG recordings. The best-performing models achieve F1 scores in the low 90s on both types of arrhythmias.

## Contents
- Training records, including loss curves, training logs, confusion matrices, feature importances, and reconstructed inputs for each auto-encoder).
- Model Checkpoints of encoders in .H5 format. Classifiers are publicly-available in cloud storage due to file sizes exceeding GitHub limits.

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

## Installation
Please install all necessary requirements:
```bash
pip install -r requirements.txt
```

## Computing and Memory requirements for training
Each encoder model has approximately 300k trainable parameters. Datasets are considerably large and training requires a download size of 5.5GB (for 12-lead data) or 45GB (for lead-II data).
GPU use is recommended, as training time for each model was between 30min and 3 hours on an NVIDIA L4 Tensor Core GPU with 22.5GB RAM

## Arguments
Each model comprises an encoder and a random forest classifier. A total of 36 models were trained, varying on:
The encoder's model architecture: CNN, Transformer or GRU (referred to as 'conv', 'transf', 'gru')
* Size of the encoding vector (length 32, 64, 128)
* Whether the encoder was trained on ECG data from only lead-II (commonly used in wearable devices) or all 12 leads; specified as 'lead_ii' or '12_lead'
* Whether the random forest classifier was trained on a balanced set with roughly equal representation of each class, or a regular training dataset with an overrepresented 'SR' class, specified as 'balanced' or 'imbalanced'
18 encoders were trained and then 2 classifiers for each encoder, resulting in 36 total.

## Datasets used:
* Encoder training: CODE-15% Dataset [Ribeiro et al 2021] comprises 300k recordings (or "exams") from 200k patients at approx 10 seconds each. All recordings are 12-lead
* Classification training: Alwan & Cvetkovic 2017, consisting of 55k recordings at approximately 8 seconds each, composed of two channels where Lead-II was used for  classification training. Total size is 160MB.
