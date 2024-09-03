"""
Pre-processing and filtering
"""

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, resample

INPUT_LENGTH=300


def remove_zero_padding(signal):
    non_zero_mask = signal != 0
    non_zero_indices = np.nonzero(non_zero_mask)[0] # Find the indices of the first and last non-zero elements
    if len(non_zero_indices) == 0 or len(non_zero_indices) < 48:
      return np.array([])  # Return an empty array if no valid signal is found
    return signal[non_zero_indices[0] : non_zero_indices[-1] + 1]

# CODE-15% is sampled at 400Hz
def downsample_signal(signal):
    if len(signal) == 0:
      return signal
    num_samples = int(len(signal) * 0.25)
    return resample(signal, num_samples)

WAVELET = 'db4'
MAX_LEVEL = pywt.dwt_max_level(INPUT_LENGTH, pywt.Wavelet(WAVELET))
def wavelet_transform(ecg):
  if len(ecg)==0:
    return ecg
  coeffs = pywt.wavedec(ecg, WAVELET, level=MAX_LEVEL)
  coeffs[0] = np.zeros_like(coeffs[0]) # Baseline wander is usually in the lowest approximation coefficient -> set to zero
  return pywt.waverec(coeffs, WAVELET)

#As described in Arafat et al. 2011, "A simple time domain algorithm for the detection of ventricular fibrillation in electrocardiogram"
#Sampling freq = 100Hz -> nyquist = 50
CUTOFF_HIGH = 1/50
CUTOFF_LOW = 30/50
B_HIGH, A_HIGH = butter(1, CUTOFF_HIGH, btype='high', analog=False)
B_LOW,  A_LOW  = butter(4, CUTOFF_LOW, btype='low', analog=False)
def filtering(ecg):
  if len(ecg)==0:
    return ecg
  mean_ecg = ecg - np.mean(ecg)
  m_a_ecg = np.convolve(mean_ecg, np.ones(5)/5, mode='same') #moving average order 5
  hp_ecg = filtfilt(B_HIGH, A_HIGH, m_a_ecg)
  lp_ecg = filtfilt(B_LOW, A_LOW, hp_ecg)
  return lp_ecg

def normalise(ecg):
  return ecg / np.std(ecg)

def preprocess(ecg, downsample=False):
  if downsample:
    return normalise(filtering(wavelet_transform(downsample_signal(remove_zero_padding(ecg)))))
  else:
    return normalise(filtering(wavelet_transform(remove_zero_padding(ecg))))