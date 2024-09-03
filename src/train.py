"""
Trains a single auto-regressive encoder and random forest classifier

This uses the CODE-15 dataset which is considerably large (45GB)

Encoder training options include model type (CNN/Transformer/GRU), encoder size (32/64/128), and whether it is trained on only lead-II data or all 12 leads

The 12-lead option has many more available samples than just a single lead, so it is ensured that both are trained on the same number of samples.
Lead II training uses all N samples of that channel, while 12-Lead uses all channels of the first (N/12) samples.


The classifier training option is whether to use a balanced dataset (where the non-Ventricular Rhythm class "SR" is not over-represented)

Because the classes SR/VT/VF are roughly split 14:1:1, the Balanced option splits all SR samples into 14 folds (in the training & validation data)
Each fold is combined with the same VT/VF samples and randomly shuffled, then used to train a classifier (resulting in an approximately 1:1:1 distribution)
This is then evaluated against a held-out test set, which was separated previous to any balancing

Results showed that training without Balancing had significantly higher accuracy results.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import joblib
import sys
import os
import datetime

from load_models import build_model
from load_datasets import load_code_15_dataset, load_custom_dataset, load_alwan_cvetkovic_dataset
from utils import parse_inputs

BATCH_SIZE = 32
path_training = './CardiacArrhythmiaDetectionModel/training_logs_and_checkpoints/'
class_weights = {'VT': 10, 'VF': 10, 'SR': 1} #Bias misclassifications of VAs due to over-representation

def time_str():
  return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def model_description(model_type, encoding_size, is_lead_ii):
  return f'{model_type}_{encoding_size}_{int(is_lead_ii)}'

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
  monitor='val_loss',
  min_delta = 0.001,
  patience=7
)

def save_loss_curves(history, model_descrip):
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss') 
  plt.legend()
  plt.savefig(path_training + f'{model_descrip}_loss.png')   

def train_autoencoder(model, model_descrip, train_dataset, valid_dataset, test_dataset): #saves the loss curve
  
  #Callbacks
  loss_logging_callback = CSVLogger(filename=path_training+f'{model_descrip}_training_log.csv', append=True)
  checkpoint_callback = ModelCheckpoint(
    filepath=f'{path_training}autoencoder_checkpoint{model_descrip}.weights.h5',
    save_weights_only=True,
    save_best_only=True,
    save_freq='epoch',
    verbose=1
  ) 

  history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=200,
    batch_size=BATCH_SIZE,
    validation_steps=500,
    steps_per_epoch=100,
    callbacks=[checkpoint_callback, loss_logging_callback, early_stopping_callback]
  )
  
  save_loss_curves(history, model_descrip)

  test_loss = model.evaluate(test_dataset, steps=50)
  print(f'Test Loss: {test_loss}')
  
  with open(path_training + 'metrics.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([model_descrip.split('_'), 'Autoregression loss (train, validation, test):', min(history.history['loss']), min(history.history['val_loss']), test_loss])
  return model

def plot_confusion_matrix(y_test, test_predictions, model_descrip, save_path):
  cm = confusion_matrix(y_test, test_predictions)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['SR', 'VT', 'VF'], yticklabels=['SR', 'VT', 'VF'])
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.savefig(save_path +f'{model_descrip}_confMat.png')
  plt.show()

def save_feature_importances(best_model, model_descrip, save_path):
  feature_importances = best_model.feature_importances_
  plt.figure(figsize=(10, 6))
  plt.bar(range(len(feature_importances)), feature_importances)
  plt.axhline(y=1/len(feature_importances), color='k', linestyle='--')
  plt.title('Feature Importances in Random Forest Classifier')
  plt.xlabel('Feature Index')
  plt.ylabel('Importance')
  plt.savefig(save_path + f'{model_descrip}_featImp.png')
  plt.show()

def split_SR_samples_into_folds(X_train_valid, y_train_valid, num_folds):
  SR_indices = np.where(y_train_valid == 'SR')[0]
  non_SR_indices = np.where(y_train_valid != 'SR')[0]

  SR_samples_folds = np.array_split(X_train_valid[SR_indices], num_folds) 
  SR_labels_folds = np.array_split(y_train_valid[SR_indices], num_folds) 

  #VT/VF samples that will be the same throughout all k-folds
  non_SR_samples = X_train_valid[non_SR_indices]
  non_SR_sample_labels = y_train_valid[non_SR_indices]
  return SR_samples_folds, SR_labels_folds, non_SR_samples, non_SR_sample_labels



def train_random_forest(model, model_descrip, is_balanced, samples, sample_labels):
  results, best_model, best_accuracy = [], None, 0

  encoder_model = Model(
      inputs = model.input,
      outputs = model.get_layer('encoding').output)
  encoded_features = encoder_model.predict(samples)

  X_train_valid, X_test, y_train_valid, y_test = train_test_split(encoded_features, sample_labels, test_size=0.1, shuffle=True, stratify=sample_labels)
    # Stratify ensures the same proportion of each class in both the test split and train/validation split (as opposed to a random shuffle alone)

  if is_balanced:
    # 'SR' class is over-represented by 14x -> Split that class into k folds and train a classifier on each one
    num_folds = 14
    SR_samples_folds, SR_labels_folds, non_SR_samples, non_SR_sample_labels = split_SR_samples_into_folds(X_train_valid, y_train_valid, num_folds)

  #Balanced training: train a classifier on each fold of the SR samples
  #Unbalanced training: train a classifier 5x on the train-validation set with different random splits each time
  num_iterations = num_folds if is_balanced else 5
  for i in range(num_iterations):
    if is_balanced:
      # Combine the SR samples of this one fold (out of 14) with the same VT/VF samples, creating a dataset with roughly equall class sizes
      X_train_valid_this_fold = np.vstack((SR_samples_folds[i], non_SR_samples))
      y_train_valid_this_fold = np.concatenate((SR_labels_folds[i], non_SR_sample_labels))
      X_train, X_val, y_train, y_val = train_test_split(X_train_valid_this_fold, y_train_valid_this_fold, test_size=0.1, shuffle=True)
    else:
      #Unbalanced training: SR class continues to be over-represented without any balancing of the dataset
      X_train, X_val, y_train, y_val = train_test_split(X_train_valid, y_train_valid, test_size=0.1, shuffle=True, stratify=y_train_valid)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
    rf_classifier.fit(X_train, y_train)

    #Evaluate this iteration's classifier on validation set
    train_predictions = rf_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    val_predictions = rf_classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f'Train Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%')
    results.append(val_accuracy)
    if val_accuracy > best_accuracy:
      best_accuracy = val_accuracy
      best_model = rf_classifier

  #save best model
  joblib.dump(best_model, f'{path_training}{model_descrip}_{is_balanced}_best_rf_classifier.joblib')

  #Evaluate best classifier on held-out test set
  average_accuracy = np.mean(results)
  best_val_accuracy = max(results)
  test_predictions = best_model.predict(X_test)
  test_accuracy = accuracy_score(y_test, test_predictions)
  print(f'Average Accuracy over {num_folds} splits: {average_accuracy:.4f}, best val accuracy {best_val_accuracy} Test Accuracy: {test_accuracy * 100:.2f}%')

  with open(path_training + 'metrics.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([model_descrip.split('_'), is_balanced, 'classification accuracy (average training, best validation, test)', average_accuracy, best_val_accuracy, test_accuracy])

  plot_confusion_matrix(y_test, test_predictions, model_descrip)
  save_feature_importances(best_model, model_descrip)



def save_reconstruct_examples(model, model_descrip, samples, sample_labels):
  #assumes already downloaded Alwan&Cvet
  rhythms = ['SR', 'VT', 'VF']
  example_rhythms_idx = [np.where(sample_labels == rhythm)[0][0] for rhythm in rhythms]
  for rhythm_index, rhythm in zip(example_rhythms_idx, rhythms):
    plt.plot(samples[rhythm_index])
    plt.plot(model.predict(samples[rhythm_index:rhythm_index+1])[0])
    plt.title('Autoregressive Encoder: Predicted and Actual ECG Waveform')
    plt.xlabel('Timestep')
    plt.ylabel('Normalised Voltage')
    plt.savefig(path_training + f'{model_descrip}_reconstruct_example_{rhythm}.png')


def main():
  args = sys.argv[1:] 
  model_type, encoding_size, is_lead_ii, is_balanced, path_ecg_data = parse_inputs(args)
  
  model = build_model(model_type, encoding_size)
  model_descrip = model_description(model_type, encoding_size, is_lead_ii)

  file_size_gb = 45 if is_lead_ii else 5.5
  print(f'Training requires the Code-15 dataset (Total download size: {file_size_gb}GB) - do you wish to proceed?')
  response = input('y/n: ')
  if response.lower() == 'n':
    print('Exiting')
    sys.exit()
  
  train_dataset_gen, valid_dataset_gen, test_dataset_gen = load_code_15_dataset(is_lead_ii)

  if path_ecg_data:
    samples, sample_labels = load_custom_dataset(path_ecg_data)
  else:
    print('Custom ECG data not supplied, downloading Alwan & Cvetkovic 2017 dataset (160MB)')
    samples, sample_labels = load_alwan_cvetkovic_dataset()

  if not os.path.exists(path_training): 
    os.makedirs(path_training)

  model = train_autoencoder(model, model_descrip, train_dataset_gen, valid_dataset_gen, test_dataset_gen)

  save_reconstruct_examples(model, model_descrip, samples, sample_labels)

  train_random_forest(model, model_descrip, is_balanced, samples, sample_labels)


if __name__ == "__main__":
  main()