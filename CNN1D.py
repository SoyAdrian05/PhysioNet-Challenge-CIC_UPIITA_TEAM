# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 22:42:57 2025

@author: Adrian
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tensorflow import keras
print("Version de tensorflow: {}".format(tf.__version__))
print("GPU: {}".format(tf.test.gpu_device_name()))

def input_layers_cnn(input_length):
    input_entry = keras.Input(shape=(input_length, 1))  # Cada rama procesa (4096, 1)
    
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_entry)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    return input_entry, x

def create_cnn_model(input_data):
    # input_data: (N, 4096, 12)
    num_channels = input_data.shape[2]  # 12
    signal_length = input_data.shape[1] # 4096

    input_layers = []
    branch_outputs = []

    for i in range(num_channels):
        input_layer, output = input_layers_cnn(signal_length)
        input_layers.append(input_layer)
        branch_outputs.append(output)

    merged = layers.Concatenate()(branch_outputs)
    x = layers.Flatten()(merged)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=input_layers, outputs=outputs)
    return model

