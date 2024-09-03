"""
Common functions used throughout
"""

import requests
import shutil
from zipfile import ZipFile 
import os

def download_file(url, destination_dir=None):
  if not destination_dir:
    destination_dir = os.getcwd()+'/'
  filename = url.split('/')[-1].split('?')[0]
  with requests.get(url, stream=True) as r:
    with open(destination_dir+filename, 'wb') as f:
      shutil.copyfileobj(r.raw, f)
  return filename

def unzip(zipped_file_path, destination_dir=None): 
  if not destination_dir:
    destination_dir = os.path.dirname(zipped_file_path)
  with ZipFile(zipped_file_path, 'r') as zipfile:
    zipfile.extractall(path=destination_dir)

def create_directory(path):
  if not os.path.exists(path): 
    os.makedirs(path)

#Sample command line input: ```inference.py conv 32 lead_ii balanced path/to/ecg/data```
def parse_inputs(args):
  if len(args) > 5:
    raise ValueError("Expected 5 arguments: inference.py <model_type> <encoding_size> <ecg_leads> <classifier_dataset_type> <path_to_ecg_data>")

  #default values
  model_type = 'conv'
  encoding_size = 32
  is_lead_ii = True
  is_balanced = False
  path_ecg_data = None

  for arg in args:
    if arg.lower() in ['conv', 'transf', 'gru']:
      model_type = arg
    elif arg.isdigit() and int(arg) in [32, 64, 128]:
      encoding_size = int(arg)
    elif arg.lower() in ['lead_ii', '12_lead']:
      is_lead_ii = (arg == 'lead_ii')
    elif arg.lower() in ['balanced', 'imbalanced']:
      is_balanced = (arg == 'balanced')
    elif '/' in arg or '\\' in arg:  # if path
      path_ecg_data = arg
    
  return  model_type, encoding_size, is_lead_ii, is_balanced, path_ecg_data