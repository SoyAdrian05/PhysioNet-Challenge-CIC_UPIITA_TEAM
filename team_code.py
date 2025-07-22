#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from CNN1D import *
from helper_code import *
import tensorflow as tf

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
# Create the tensor
def create_signal_tensor(signals, signal_info, verbose):
    if not signals:
        return np.array([]), 0
    
    # Obtener estadísticas de las longitudes
    lengths = [signal.shape[0] for signal in signals]
    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = np.mean(lengths)
    
    
    # Decidir la longitud objetivo (puedes ajustar esta estrategia)
    # Opción 1: Usar la longitud máxima (más información, pero más memoria)
    target_length = max_length
    
    # Opción 2: Usar un percentil alto (balancear información vs memoria)
    # target_length = int(np.percentile(lengths, 95))
    
    # Opción 3: Usar longitud fija estándar para ECG (ej: 5000 muestras para 10 segundos a 500Hz)
    # target_length = 5000
    
    n_samples = len(signals)
    n_channels = 12  # Siempre 12 canales ECG
    
    # Crear tensor vacío
    tensor = np.zeros((n_samples, target_length, n_channels), dtype=np.float32)
    
    # Llenar el tensor
    for i, signal in enumerate(signals):
        original_length = signal.shape[0]
        
        if original_length >= target_length:
            # Truncar si es más larga
            tensor[i, :, :] = signal[:target_length, :]
        else:
            # Padding si es más corta
            tensor[i, :original_length, :] = signal
    
    # Reemplazar NaN e infinitos con ceros
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    return tensor, target_length
# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the signals and labels from the data.
    if verbose:
        print('Extracting signals and labels from the data...')

    # Iterate over the records to extract the signals and labels.
    all_signals = list()
    labels = list()
    signal_lengths = list()  # Para trackear las longitudes originales
    signal_info = list()  # Para guardar información adicional de cada señal
    
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        
        # Cargar las señales directamente
        signal, fields = load_signals(record)
        header = load_header(record)
        source = get_source(header)
        label = load_label(record)
        
        # Reordenar los canales para consistencia
        channels = fields['sig_name']
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        signal = reorder_signal(signal, channels, reference_channels)
        labels.append(label)
        all_signals.append(signal)
    # Convertir señales a tensor con padding/truncating
    labels = np.asarray(labels, dtype=bool)
    
    # Crear tensor de señales con dimensiones uniformes
    signal_tensor, padded_length = create_signal_tensor(all_signals, signal_info, verbose)

    # Train the models on the signals.
    if verbose:
        print('Training the model on the signal data...')
        print(f'Number of signals: {len(all_signals)}')
        print(f'Signal tensor shape: {signal_tensor.shape}')
        print(f'Padded length: {padded_length}')
    
    model = create_cnn_model(x_train)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=2)

    epochs = 100

    cnn_model_history = model.fit(signal_tensor, labels, epochs = epochs, batch_size=32)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder):
    filename = os.path.join(model_folder, 'model.keras')
    model = tf.keras.models.load_model(filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    if verbose:
        print('Finding the Challenge data...')
    records = find_records(data_folder)
    num_records = len(records)
    
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # Extract the signals and labels from the data.
    if verbose:
        print('Extracting signals and labels from the data...')
    
    # Iterate over the records to extract the signals and labels.
    all_signals = list()
     
        
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
    
        record = os.path.join(data_folder, records[i])
        
        # Cargar las señales directamente
        signal, fields = load_signals(record)
    
        
        # Reordenar los canales para consistencia
        channels = fields['sig_name']
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        signal = reorder_signal(signal, channels, reference_channels)
        all_signals.append(signal)
    
    
    # Crear tensor de señales con dimensiones uniformes
    signal_tensor, padded_length = create_signal_tensor(all_signals, signal_info, verbose)
     
        # Train the models on the signals.
    if verbose:
        print('Training the model on the signal data...')
        print(f'Number of signals: {len(all_signals)}')
        print(f'Signal tensor shape: {signal_tensor.shape}')
        print(f'Padded length: {padded_length}')
    probabilities = model.predict(signal_tensor, verbose=1)  # Salida (10000, 1)
    
    binary_outputs = (probabilities >= 0.5).astype(int)

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Save your trained model.
def save_model(model_folder, model):
    os.makedirs(model_folder, exist_ok=True)
    filename = os.path.join(model_folder, 'model.keras')
    model.save(filename)   
