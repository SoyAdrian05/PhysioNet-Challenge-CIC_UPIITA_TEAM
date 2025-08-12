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
def create_signal_tensor(signals, signal_info, verbose, target_length):
    if not signals:
        return np.array([]), 0
    
    lengths = [signal.shape[0] for signal in signals]
    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = np.mean(lengths)

    target_length = max_length
    
    n_samples = len(signals)
    n_channels = 12 

    tensor = np.zeros((n_samples, target_length, n_channels), dtype=np.float32)

    for i, signal in enumerate(signals):
        original_length = signal.shape[0]
        
        if original_length >= target_length:
            tensor[i, :, :] = signal[:target_length, :]
        else:

            tensor[i, :original_length, :] = signal
    

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
    signal_lengths = list()  
    signal_info = list()  
    
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        

        signal, fields = load_signals(record)
        header = load_header(record)
        source = get_source(header)
        label = load_label(record)
        

        channels = fields['sig_name']
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        signal = reorder_signal(signal, channels, reference_channels)
        labels.append(label)
        all_signals.append(signal)

    labels = np.asarray(labels, dtype=bool)
    
  
    signal_tensor, padded_length = create_signal_tensor(all_signals, signal_info, verbose, target_length = False)


    # Train the models on the signals.
    if verbose:
        print('Training the model on the signal data...')
        print(f'Number of signals: {len(all_signals)}')
        print(f'Signal tensor shape: {signal_tensor.shape}')
        print(f'Padded length: {padded_length}')
        
    signal_list = [signal_tensor[:, :, i:i+1] for i in range(signal_tensor.shape[2])]
    
    model = create_cnn_model(signal_tensor)
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    y_train = labels
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y = np.argmax(y_train_onehot, axis=1) 
    y = y.reshape(-1)
    
    print("Y_train: ", y_train.shape)
    print("y_train_onehot: ", y_train_onehot.shape)
    print("y: ", y.shape)


    epochs = 100
    print("Version de tensorflow: {}".format(tf.__version__))
    print("GPU: {}".format(tf.test.gpu_device_name()))

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    cnn_model_history = model.fit(signal_list, y, epochs = epochs, batch_size=10)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.keras')
    model = tf.keras.models.load_model(filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(data_folder, model, verbose):
    print("Data_folder: ", data_folder)
    if verbose:
        print('Finding the Challenge data...')
    
    # Iterate over the records to extract the signals and labels.
    signal_info = list()  
    
    signal, fields = load_signals(data_folder)
    
    signal_tensor, padded_length = create_signal_tensor([signal], signal_info, verbose, target_length = 4096)
    
    
    if len(signal_tensor.shape) == 2:  # Si tiene forma [4096, 12]
        signal_tensor = signal_tensor[np.newaxis, :, :]  # Agregar dimensión batch -> [1, 4096, 12]
        if verbose:
            print(f'Tensor shape corrected from 2D to 3D: {signal_tensor.shape}')
    elif len(signal_tensor.shape) == 3 and signal_tensor.shape[0] != 1:
        
        signal_tensor = signal_tensor[:1, :, :]
        if verbose:
            print(f'Tensor batch size adjusted to 1: {signal_tensor.shape}')
    
    
    if signal_tensor.shape[1] != 4096:
        if verbose:
            print(f'Warning: Signal length is {signal_tensor.shape[1]}, expected 4096')
        
        if signal_tensor.shape[1] < 4096:
            
            padding_needed = 4096 - signal_tensor.shape[1]
            padding = np.zeros((signal_tensor.shape[0], padding_needed, signal_tensor.shape[2]))
            signal_tensor = np.concatenate([signal_tensor, padding], axis=1)
            if verbose:
                print(f'Padded signal to shape: {signal_tensor.shape}')
        else:
           
            signal_tensor = signal_tensor[:, :4096, :]
            if verbose:
                print(f'Truncated signal to shape: {signal_tensor.shape}')
    
    if verbose:
        print('Training the model on the signal data...')
        print(f'Signal tensor shape: {signal_tensor.shape}')
        print(f'Padded length: {padded_length}')
        
    
    signal_list = []
    for i in range(signal_tensor.shape[2]):  
        
        channel_data = signal_tensor[:, :, i:i+1] 
        signal_list.append(channel_data)
    
    if verbose:
        print(f'Signal list length: {len(signal_list)}')
        print(f'Each signal shape: {signal_list[0].shape}')
    
    probability_output = model.predict(signal_list, verbose=1)  
    
    binary_outputs = (probability_output >= 0.5).astype(int)
    return binary_outputs, probability_output

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
