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


def input_layers_cnn(input_length):
    input_entry = keras.Input(shape=(input_length, 1))  # Cada rama procesa (4096, 1)
    
    x = layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(input_entry)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(x)
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
    x = layers.GlobalAveragePooling1D()(merged)
    # x = layers.Flatten()(merged)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=input_layers, outputs=outputs)
    return model

def cnn_model2(tensor):
    input_lenght = tensor.shape
    input1 = layers.Input(shape=(input_lenght[1], 2050), name="input_1")
    x1 = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(input1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.Flatten()(x1)

    # Entrada 2: (12, 1027)
    input2 = layers.Input(shape=(input_lenght[1], 1027), name="input_2")
    x2 = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(input2)
    x2 = layers.MaxPooling1D(pool_size=2)(x2)
    x2 = layers.Flatten()(x2)

    # Entrada 3: (12, 256)
    input3 = layers.Input(shape=(input_lenght[1], 516), name="input_3")
    x3 = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(input3)
    x3 = layers.MaxPooling1D(pool_size=2)(x3)
    x3 = layers.Flatten()(x3)

    # Concatenar las tres ramas
    concat = layers.Concatenate()([x1, x2, x3])

    # Capas densas finales
    dense = layers.Dense(128, activation="relu")(concat)
    output = layers.Dense(1, activation="sigmoid")(dense)  # cambiar a softmax si es multiclase

    # Definir el modelo
    model = keras.Model(inputs=[input1, input2, input3], outputs=output)


    return model

