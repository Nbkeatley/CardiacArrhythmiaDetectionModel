"""
Ventricular Arrhythmia Detection and Classification

Detects two forms of Ventricular Arrhythmia (Ventricular Fibrillation/VF and Ventricular Tachycardia/VT) from normal sinus rhythms (SR)
"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from enum import Enum
import sys
import os

from load_models import load_encoder_from_weights, load_classifier
from load_datasets import load_custom_dataset, load_alwan_cvetkovic_dataset
from train import model_description, plot_confusion_matrix, save_feature_importances
from utils import parse_inputs


INPUT_LENGTH=300 #ECG window size for each ECG sample, and input/output size to models
STRIDE=100

class Model_Type(Enum):
  CONV = 'conv'
  TRANSF = 'transf'
  GRU = 'gru'



def main():
  args = sys.argv[1:] 
  model_type, encoding_size, is_lead_ii, is_balanced, path_ecg_data = parse_inputs(args)
  
  if path_ecg_data:
    samples, sample_labels = load_custom_dataset(path_ecg_data)
  else:
    print('Custom ECG data not supplied, downloading Alwan & Cvetkovic 2017 dataset (160MB)')
    samples, sample_labels = load_alwan_cvetkovic_dataset()
  
  encoder_model = load_encoder_from_weights(model_type, encoding_size, is_lead_ii)
  encoded_features = encoder_model.predict(samples)

  random_forest_classifier = load_classifier(model_type, encoding_size, is_lead_ii, is_balanced)
  predictions = random_forest_classifier.predict(encoded_features)
  #accuracy = accuracy_score(sample_labels, predictions)

  print(classification_report(sample_labels, predictions, labels=['SR', 'VT', 'VF']))

  inference_results_path = './CardiacArrhythmiaDetectionModel/inference_results/'
  if not os.path.exists(inference_results_path): 
    os.makedirs(inference_results_path)
  model_descrip = model_description(model_type, encoding_size, is_lead_ii)
  plot_confusion_matrix(sample_labels, predictions, model_descrip, inference_results_path)
  save_feature_importances(random_forest_classifier, model_descrip, inference_results_path)


if __name__ == "__main__":
  main()