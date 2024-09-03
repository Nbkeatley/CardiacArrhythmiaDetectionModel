"""
Ventricular Arrhythmia Detection and Classification

Detects two forms of Ventricular Arrhythmia (Ventricular Fibrillation/VF and Ventricular Tachycardia/VT) from normal sinus rhythms (SR)
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from enum import Enum
import sys

from load_models import load_encoder_from_weights, load_classifier
from load_datasets import load_custom_dataset, load_alwan_cvetkovic_dataset
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
  
  encoder_model = load_encoder_from_weights(model_type)
  encoded_features = encoder_model.predict(samples)

  random_forest_classifier = load_classifier(model_type, encoding_size, is_lead_ii, is_balanced)
  predictions = random_forest_classifier.predict(encoded_features)
  #accuracy = accuracy_score(sample_labels, predictions)

  print(classification_report(sample_labels, predictions, labels=['SR', 'VT', 'VF']))

  cm = confusion_matrix(sample_labels, predictions)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['SR', 'VT', 'VF'], yticklabels=['SR', 'VT', 'VF'])
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.show()

  feature_importances = random_forest_classifier.feature_importances_
  plt.figure(figsize=(10, 6))
  plt.bar(range(len(feature_importances)), feature_importances)
  plt.axhline(y=1/len(feature_importances), color='k', linestyle='--')
  plt.title('Feature Importances in Random Forest Classifier')
  plt.xlabel('Feature Index')
  plt.ylabel('Importance')
  plt.show()

if __name__ == "__main__":
  main()