"""
Load datasets for training and evaluation

Alwan & Cvetkovic 2017 Dataset comprises 55,000 records at approximately 8 seconds each
Format: data[rhythm_label][0][recording][timestep][voltage_reading][channel]
100Hz sampling rate
Data is mostly two channels, where the first is Lead-II in almost all cases (which will be used here)
160MB size


CODE-15% Dataset [Ribeiro et al 2021] comprises 300,000 recordings (or "exams") from 200,000 patients at approx 10 seconds each
All recordings are 12-lead
400Hz sampling rate (must be downsampled for better model prediction on both this and the 100Hz dataset)
Each recording 10sec -> 3Mn seconds total on all channels (Or 36Mn seconds with all 12 leads)
43.9GB total size

Format: 18 HDF5 files, each containing two datasets:
  'exam_id' has shape `(N,)` containing the UID of each recording - with further information on each patient in the `exam.csv` file
  'tracings' has shape `(N, 4096, 12)`
  - N recordings [around 20,000 in each .hdf5 file, each is approx 10 seconds duration]
  - 4096 sampled timesteps
  - 12 different leads of the ECG exams in the following order: `{DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}`.


Dataset generators because too large
Num of leads same total number records -> only first 28,814 recordings (345,779 / 12)
Train/valid/test splits are handled by selecting indices of the data samples (e.g. 2,7,12,17,22,27...)

"""

import h5py
import glob
import random
import numpy as np
import tensorflow as tf
import scipy
import requests
import os
import gdown

from preprocessing import preprocess
from utils import download_file, unzip, create_directory

INPUT_LENGTH = 300
STRIDE = 100
BATCH_SIZE = 32


#Assumes format of data[rhythm_label][0][recording][timestep][voltage_reading][channel]
def load_custom_dataset(path):
  ecg_data = scipy.io.loadmat(path, mdict=None, appendmat=True)
  samples, sample_labels = [],[]
  for label in ['SR', 'VT', 'VF']:
    for recording in ecg_data[label][0]:
      recording_channel_1 = preprocess(recording[:][:,0], downsample=False) #only using channel 1
      num_samples = (len(recording_channel_1) - INPUT_LENGTH) // STRIDE + 1
      for i in range(num_samples):
        start = i*STRIDE
        end = start + INPUT_LENGTH
        samples.append(recording_channel_1[start:end])
        sample_labels.append(label)

  return np.array(samples), np.array(sample_labels)

def load_alwan_cvetkovic_dataset():
  url = 'https://drive.google.com/uc?export=download&id=1buqmp39c3ng6EUt4Baa7EOuW8LVi9rDv'
  destination_dir = './CardiacArrhythmiaDetectionModel/alwan_cvetkovic_2017_dataset/'
  create_directory(destination_dir)
  destination_path = destination_dir+'RHYTHMS.mat'
  gdown.download(url, destination_path, quiet=False)
  samples, sample_labels = load_custom_dataset(destination_path)
  return samples, sample_labels





# Train/validation/test splits of CODE-15 recordings (note: each recording will later be split into multiple samples)
def create_splits(hdf5_files):

  # Calculate number of recordings across all downloaded files
  total_num_recordings = 0
  for hdf5_file in hdf5_files:
    with h5py.File(hdf5_file, 'r') as f:
      total_num_recordings += f['tracings'].shape[0]
  indices = np.arange(total_num_recordings)
  
  # Randomly split in the ratio 80:10:10, using modulus
  val_index, test_index = random.sample(range(10), 2)
  train_indices = indices[(indices % 10 != val_index) & (indices % 10 != test_index)]
  valid_indices = indices[indices % 10 == val_index]
  test_indices = indices[indices % 10 == test_index]
  return train_indices, valid_indices, test_indices

#Transform each recording into multiple samples of a shorter duration (assuming already downsampled to 100Hz)
def create_windowed_samples(recording):
  windows = []
  for start in range(0, len(recording) - INPUT_LENGTH + 1, STRIDE):
    window = recording[start:start + INPUT_LENGTH]
    windows.append(window)
  return np.array(windows)

#Used to generate preprocessed samples  
#For any of the train/validation/test sets
#And whether using all 12 ecg leads or only Lead II
def data_generator(hdf5_files, indices, is_lead_ii):
  num_recordings_total = 0
  for hdf5_file in hdf5_files:
    with h5py.File(hdf5_file, 'r') as f:
      file_data = f['tracings']
      num_recordings_in_file = file_data.shape[0] 
      if is_lead_ii:
        ecg_leads = np.array([1]) # Lead II
      else:
        ecg_leads = np.array(range(file_data.shape[2])) # All 12 leads
      
      #Get indices which are only in this .hdf5 file
      file_indices = indices[(indices >= num_recordings_total) & (indices < num_recordings_total + num_recordings_in_file)] - num_recordings_total
      
      stopping_condition = lambda: num_recordings_total < 28814 if not is_lead_ii else True
      while True: 
        for batch_start in range(0, len(file_indices), BATCH_SIZE):
          batch_end = min(batch_start + BATCH_SIZE, len(file_indices))
          ecg_batch_indices = file_indices[batch_start:batch_end]
          ecg_batch = file_data[ecg_batch_indices, :, :]

          processed_batch = []
          for ecg_lead in ecg_leads:
            for recording in range(ecg_batch.shape[0]):

              processed_recording = preprocess(ecg_batch[recording, :, ecg_lead], downsample=True)
              if len(processed_recording) == 0:
                continue

              windows = create_windowed_samples(processed_recording)
              for window in windows: #In case no windows are created
                if window.shape[0]!=INPUT_LENGTH:
                  continue
                processed_batch.append(window)
              
          processed_batch = np.array(processed_batch)
          yield processed_batch, processed_batch
    num_recordings_total += num_recordings_in_file 

def build_dataset(hdf5_files, indices, is_lead_ii):
  return tf.data.Dataset.from_generator(
    lambda: data_generator(hdf5_files, indices, is_lead_ii),
    output_signature=(
      tf.TensorSpec(shape=(None, INPUT_LENGTH), dtype=tf.float32),
      tf.TensorSpec(shape=(None, INPUT_LENGTH), dtype=tf.float32)
    )
  )

def is_file_already_downloaded(url, destination_file_path):
  expected_size = int(requests.head(url, allow_redirects=True).headers.get('Content-Length', 0))
  if os.path.exists(destination_file_path) and expected_size==os.path.getsize(destination_file_path):
    print(f"The file '{destination_file_path}' has already been fully downloaded, proceeding..")
    return True
  return False

def load_code_15_dataset(is_lead_ii, dataset_dir=None):
  if not dataset_dir:
    dataset_dir = './CardiacArrhythmiaDetectionModel/code_15_dataset_files/'
  num_zipped_files_to_download = 18 if is_lead_ii else 2

  for i in range(num_zipped_files_to_download):
    print(f'Downloading file {i+1} of {num_zipped_files_to_download}..')
    url = f'https://zenodo.org/records/4916206/files/exams_part{i}.zip?download=1'
    zipped_filename = f'exams_part{i}.zip'
    if not is_file_already_downloaded(url, dataset_dir+zipped_filename):
      download_file(url, dataset_dir+zipped_filename)
    if not os.path.exists(dataset_dir + f'exams_part{i}.hdf5'):
      unzip(zipped_filename, dataset_dir)
  
  hdf5_files = sorted(glob.glob(dataset_dir + 'exams_part*.hdf5'))

  # Note: These are only indices of the recordings, each recording is split into multiple samples
  train_indices, valid_indices, test_indices = create_splits(hdf5_files)

  train_dataset_gen = build_dataset(hdf5_files, train_indices, is_lead_ii)
  valid_dataset_gen = build_dataset(hdf5_files, valid_indices, is_lead_ii)
  test_dataset_gen = build_dataset(hdf5_files, test_indices, is_lead_ii)
  return train_dataset_gen, valid_dataset_gen, test_dataset_gen

