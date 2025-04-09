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

def input_layers_cnn(input_shape):

    input_entry = keras.Input(shape=(input_shape, 1))

    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_entry)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    return input_entry, x

def create_cnn_model(input_data):
    red_paralela = input_data.shape[1]
    data = input_data.shape[2]  
    
    input_layers = [] 
    branch_outputs = []  
    
    for i in range(red_paralela):
        input_layer, output = input_layers_cnn(data)
        input_layers.append(input_layer)
        branch_outputs.append(output)
    
    merged = layers.Concatenate()(branch_outputs)
    
    flattened = layers.Flatten()(merged)
    
    x = layers.Dense(64, activation='relu')(flattened)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layers, outputs=outputs)
    return model
    

