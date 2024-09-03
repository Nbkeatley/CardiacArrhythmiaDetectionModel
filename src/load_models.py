"""
Load models from checkpoints and build models for training
Models vary by model type (CNN/Transformer/GRU) and encoder size (32/64/128)
Decoders all use the same architecture to allow better comparison

Classifier file sizes exceed GitHub limits (100MB) and so these are accessed via Google Drive
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import joblib

from utils import download_file 


INPUT_LENGTH = 300 #ECG window size for each ECG sample, and input/output size to models
STRIDE = 100
BATCH_SIZE = 32
AUTO_ENCODER_OPTIMIZER = 'adam'
AUTO_ENCODER_LOSS = 'huber'


# DECODER
intermediate_size = INPUT_LENGTH // 4
def build_decoder(encoding, input_size=INPUT_LENGTH):
  x = layers.Dense(intermediate_size * 32, activation='relu')(encoding)
  x = layers.Reshape((intermediate_size, 32))(x)
  x = layers.UpSampling1D(size=2)(x)
  x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
  x = layers.UpSampling1D(size=2)(x)
  x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)

  # Ensures that output is cropped correctly
  target_size = input_size
  cropping_amount = x.shape[1] - target_size
  if cropping_amount > 0:
    x = layers.Cropping1D(cropping=(0, cropping_amount))(x)

  output_ecg = layers.Conv1D(1, kernel_size=3, activation='linear', padding='same')(x)
  return output_ecg


def build_conv_autoencoder(encoding_size):
  input_ecg = layers.Input(shape=(INPUT_LENGTH, 1))
  x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_ecg)
  x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
  x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
  x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(encoding_size * 2, activation='relu')(x)
  encoding = layers.Dense(encoding_size, activation='relu', name='encoding')(x)

  output_ecg = build_decoder(encoding, input_size=INPUT_LENGTH)

  autoencoder = Model(inputs=input_ecg, outputs=output_ecg)
  autoencoder.compile(optimizer=AUTO_ENCODER_OPTIMIZER, loss=AUTO_ENCODER_LOSS)
  return autoencoder


#TRANSFORMER MODEL

def build_transformer_encoder(encoding_size):
  
  # Helper Function to create a single transformer encoder layer
  def transformer_encoder_layer(inputs, embed_dim, num_heads, ff_dim, rate=0.1, training=True):
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = layers.Dropout(rate)(attn_output, training=training)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)  # residual no. 1

    # Feed-forward network
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output, training=training)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)  # residual no. 2
  
  input_shape=(300,1)

  embed_dim = 4 # Embedding size per token
  num_heads = 4  # Number of attention heads
  ff_dim = 32 # Hidden layer size in feed forward network inside transformer
  num_layers = 2  # Number of transformer layers
  rate = 0.1  # Dropout rate
  training = True

  inputs = layers.Input(shape=input_shape)
  x = layers.Reshape((input_shape[0], 1))(inputs)
  x = layers.Dense(embed_dim)(inputs) # Projecting the input to a higher dimension

  # Adding positional encoding
  positions = tf.range(start=0, limit=INPUT_LENGTH, delta=1)
  position_embedding = layers.Embedding(input_dim=INPUT_LENGTH, output_dim=embed_dim)(positions)
  position_embedding = tf.expand_dims(position_embedding, axis=0)
  x = x + position_embedding

  for _ in range(num_layers):
    x = transformer_encoder_layer(x, embed_dim, num_heads, ff_dim, rate, training)

  x = layers.Flatten()(x)
  encoded_output = layers.Dense(encoding_size, activation='relu', name='encoding')(x)

  output_ecg = build_decoder(encoded_output)
  tfmr_autoencoder = Model(inputs=inputs, outputs=output_ecg)
  tfmr_autoencoder.compile(
    optimizer=AUTO_ENCODER_OPTIMIZER,
    loss=AUTO_ENCODER_LOSS)
  return tfmr_autoencoder


# GRU AUTOENCODER

def build_gru_autoencoder(encoding_size):
  inputs = layers.Input(shape=(INPUT_LENGTH, 1))
  x = layers.BatchNormalization()(inputs)
  x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
  x = layers.Bidirectional(layers.GRU(units=4, return_sequences=True))(x) #100
  x = layers.Dropout(rate=0.2)(x)
  x = layers.Bidirectional(layers.GRU(units=4, return_sequences=True))(x) #50
  x = layers.Dropout(rate=0.2)(x)
  x = layers.Conv1D(filters=1, kernel_size=5, activation='relu', padding='same')(x)

  #Crop outputs by differing amount, depending on encoding size
  if encoding_size==128:
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x) #150xN
    x = layers.Reshape((150,1))(x)
    x = layers.Cropping1D(cropping=(11, 11))(x)  #150x1 -> 128x1, cropping 22 (or 11 from both sides)
  elif encoding_size==64:
    x = layers.MaxPooling1D(pool_size=4, padding='same')(x) #75xN
    x = layers.Reshape((75,1))(x)
    x = layers.Cropping1D(cropping=(6, 5))(x)  #75x1 -> 64x1, cropping 11
  else: #32
    x = layers.MaxPooling1D(pool_size=8, padding='same')(x) #38xN
    x = layers.Reshape((38,1))(x)
    x = layers.Cropping1D(cropping=(3, 3))(x)  #38x1 -> 32x1, cropping 6

  x = layers.Flatten(name='encoding')(x)
  output_ecg = build_decoder(x, input_size=INPUT_LENGTH)
  autoencoder = Model(inputs=inputs, outputs=output_ecg)
  autoencoder.compile(
    optimizer=AUTO_ENCODER_OPTIMIZER,
    loss=AUTO_ENCODER_LOSS)
  return autoencoder

#Load classifiers (fitted to the encoded features of each encoder)
#Note that file sizes are 40..110MB each. File sizes can be reduced with little loss in performance using pruning, this can be completed in future work
def load_classifier(model_type, encoding_size, is_lead_ii, is_balanced):
  classifier_url_dct = {
    ('conv', 32, 0, 0):     'https://drive.google.com/file/d/1N73hMMQYVYBAKjBloWyYpbhE7OUO15jQ/view?usp=drive_link',
    ('conv', 32, 1, 0):     'https://drive.google.com/file/d/1SthemX5t2siV-q_iEHwmHAeUQpmH75Xp/view?usp=drive_link',
    ('conv', 64, 0, 0):     'https://drive.google.com/file/d/1H-xL_3LDyV0eiBYMZLeo51B5RIXip-S8/view?usp=drive_link',
    ('conv', 64, 1, 0):     'https://drive.google.com/file/d/18nU5lazXyJtJYkdxR3vsjocjp2zBC5UJ/view?usp=drive_link',
    ('conv', 128, 0, 0):    'https://drive.google.com/file/d/1-NEgJdCTjTYU6zyX3RAozIhvGA3PCXpk/view?usp=drive_link',
    ('conv', 128, 1, 0):    'https://drive.google.com/file/d/1Gcl5gNQB8brMzEuKL4RbQRR4iZAQYUoB/view?usp=drive_link',
    ('gru', 32, 0, 0):      'https://drive.google.com/file/d/1WaiyWCxiLoDKLy8m9nXHAeZkRdkQrIG6/view?usp=drive_link',
    ('gru', 32, 1, 0):      'https://drive.google.com/file/d/1-yrM7M6Lw-iS4h1tQ4YSQzZEU10VFvm6/view?usp=drive_link',
    ('gru', 64, 0, 0):      'https://drive.google.com/file/d/1J8ejNASKupsbvnnNUGjfop8wQkqvRwqW/view?usp=drive_link',
    ('gru', 64, 1, 0):      'https://drive.google.com/file/d/150CL2S3djhwNO9tgG0c8oMxsezrJcoKP/view?usp=drive_link',
    ('gru', 128, 0, 0):     'https://drive.google.com/file/d/1ODE5xFyoFRmc0jwvqKrGKrwR0BGHZ1Ah/view?usp=drive_link',
    ('gru', 128, 1, 0):     'https://drive.google.com/file/d/1qitQJUFSwp0NQTyDtqaXz9bBVXlqOC59/view?usp=drive_link',
    ('transf', 32, 0, 0):   'https://drive.google.com/file/d/1V-KVzsUGtpPBTfjdJIuftuNiIatcxi87/view?usp=drive_link',
    ('transf', 32, 1, 0):   'https://drive.google.com/file/d/18ldnQ2C37KyZXh4qZlrZSnpzmUK12M3I/view?usp=drive_link',
    ('transf', 64, 0, 0):   'https://drive.google.com/file/d/1SAgdUviW2yJhCg4izWc_pn-fOtFW5E7f/view?usp=drive_link',
    ('transf', 64, 1, 0):   'https://drive.google.com/file/d/1wZI56q-dosON4kqfyE8iHxRq99UuCfNM/view?usp=drive_link',
    ('transf', 128, 0, 0):  'https://drive.google.com/file/d/1WTugeipPUhevXGi20YVAM98JVcnJ8ssI/view?usp=drive_link',
    ('transf', 128, 1, 0):  'https://drive.google.com/file/d/1Jfs-Dlk6j1YLsf0IwGy_9D2EN5zyN7Sh/view?usp=drive_link',
    ('conv', 32, 0, 1):     'https://drive.google.com/file/d/1zO9ypPPckVcANTPVX3u7vaUOA-0a135D/view?usp=drive_link',
    ('conv', 32, 1, 1):     'https://drive.google.com/file/d/1GXyA-BCr9VHK8U1g11K8mvjyYrXGQoDl/view?usp=drive_link',
    ('conv', 64, 0, 1):     'https://drive.google.com/file/d/1DjmYQmkrLLUZdWFjm0f_usYXzYS7TSjW/view?usp=drive_link',
    ('conv', 64, 1, 1):     'https://drive.google.com/file/d/1Dwz-zeHxd4wEFVIXcQm5T-xMDEYGCZZF/view?usp=drive_link',
    ('conv', 128, 0, 1):    'https://drive.google.com/file/d/1R3vUHt9FAfkFECQ6PUoCTbMfW3MzrobK/view?usp=drive_link',
    ('conv', 128, 1, 1):    'https://drive.google.com/file/d/1P6KtdeCSvbZlpeS2vsZX7vz7gGcyueMX/view?usp=drive_link',
    ('gru', 32, 0, 1):      'https://drive.google.com/file/d/1jy-DZYGUXjniNtn95TpgQYLZz2S-an21/view?usp=drive_link',
    ('gru', 32, 1, 1):      'https://drive.google.com/file/d/1Rz1vlMYQ6qTUhrvEAAlq_Gd5pNxbIM-y/view?usp=drive_link',
    ('gru', 64, 0, 1):      'https://drive.google.com/file/d/11cz54GIeWIvsjNWibK9Km9ivWJrAWsGQ/view?usp=drive_link',
    ('gru', 64, 1, 1):      'https://drive.google.com/file/d/1ewAe2ccN93WejmKxHweZnGxtdEqiydfW/view?usp=drive_link',
    ('gru', 128, 0, 1):     'https://drive.google.com/file/d/11UJ_WXu00CcvXdFBFJjuUp4DYZ9-6lUN/view?usp=drive_link',
    ('gru', 128, 1, 1):     'https://drive.google.com/file/d/1ZE6lKiri48Oo3njqveBdwN7gcBprayy_/view?usp=drive_link',
    ('transf', 32, 0, 1):   'https://drive.google.com/file/d/1jog0DRR7UCJunxdCPCRQBggCRG5CGdTo/view?usp=drive_link',
    ('transf', 32, 1, 1):   'https://drive.google.com/file/d/12K3ym9MceZj1k7iDPGtWpKfsCmidkGKG/view?usp=drive_link',
    ('transf', 64, 0, 1):   'https://drive.google.com/file/d/1Q3oSpUo9pblPUqya0Fis0KNLftutWAr1/view?usp=drive_link',
    ('transf', 64, 1, 1):   'https://drive.google.com/file/d/1pWf_WkJC6vXUsNyINjwUaGYn20tbbaC7/view?usp=drive_link',
    ('transf', 128, 0, 1):  'https://drive.google.com/file/d/16QZ3-bfxwU_s8OiAGo7kYtCKQ7b5udon/view?usp=drive_link',
    ('transf', 128, 1, 1):  'https://drive.google.com/file/d/1tGaN7YdcGpNOZv-310cWJp8ZsqAZulM5/view?usp=drive_link'
  }
  classifier_url = classifier_url_dct[(model_type, encoding_size, int(is_lead_ii), int(is_balanced))]
  classifier_filename = download_file(classifier_url)
  classifier = joblib.load(classifier_filename)
  return classifier

def build_model(model_type, encoding_size):
  if model_type == 'conv':
    model = build_conv_autoencoder(encoding_size)
  elif model_type == 'transf':
    model = build_transformer_encoder(encoding_size)
  elif model_type == 'gru':
    model = build_gru_autoencoder(encoding_size)
  else:
    print('Model name not recognised, options are: conv, transf or gru')
    return None
  return model

def load_encoder_decoder_from_weights(model_type, encoding_size, is_lead_ii):
  model = build_model(model_type, encoding_size)
  filename = f'autoencoder_checkpoint_{model_type}_{encoding_size}_{int(is_lead_ii)}.weights.h5'
  model.load_weights('./models/'+filename)
  model.summary()
  return model

def load_encoder_from_weights(model_type, encoding_size, is_lead_ii):
  encoder_decoder = load_encoder_decoder_from_weights(model_type, encoding_size, is_lead_ii)
  encoder_model = Model(
    inputs = encoder_decoder.input,
    outputs = encoder_decoder.get_layer('encoding').output)
  return encoder_model
