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
from scipy.signal import butter, filtfilt
import pywt
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from CNN1D import *
from helper_code import *
import tensorflow as tf
import psutil
print("Version de tensorflow: {}".format(tf.__version__))
print("GPU: {}".format(tf.test.gpu_device_name()))

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("NO GPU")
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
# Create the tensor

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs  
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def filtro_señal_3d(tensor_3d):

    n_samples, signal_length, n_channels = tensor_3d.shape
    tensor_filtrado = np.zeros_like(tensor_3d)
    
    for sample in range(n_samples):
        for channel in range(n_channels):
            # Extraer la señal de un canal específico de una muestra específica
            signal = tensor_3d[sample, :, channel]
            # Aplicar filtro
            tensor_filtrado[sample, :, channel] = butter_bandpass_filter(
                signal, 0.5, 150, 400, order=4
            )
    
    return tensor_filtrado

def wavelet_signals(tensor):
   n,m,chanels = tensor.shape
   matrix_detail1 = []
   matrix_detail2 = []
   matrix_detail3 = []
   
   for i in range(n):
       signal0 = apply_wavelet_transform(tensor[i,:,0])
       signal1 = apply_wavelet_transform(tensor[i,:,1])
       signal2 = apply_wavelet_transform(tensor[i,:,2])
       signal3 = apply_wavelet_transform(tensor[i,:,3])
       signal4 = apply_wavelet_transform(tensor[i,:,4])
       signal5 = apply_wavelet_transform(tensor[i,:,5])
       signal6 = apply_wavelet_transform(tensor[i,:,6])
       signal7 = apply_wavelet_transform(tensor[i,:,7])
       signal8 = apply_wavelet_transform(tensor[i,:,8])
       signal9 = apply_wavelet_transform(tensor[i,:,9])
       signal10 = apply_wavelet_transform(tensor[i,:,10])
       signal11 = apply_wavelet_transform(tensor[i,:,11])
       
       matrix_detail1.append([signal0[3],signal1[3],signal2[3],
                              signal3[3],signal4[3],signal5[3],
                              signal6[3],signal7[3],signal8[3],
                              signal9[3],signal10[3],signal11[3]]
                              )
       matrix_detail2.append([signal0[2],signal1[2],signal2[2],
                              signal3[2],signal4[2],signal5[2],
                              signal6[2],signal7[2],signal8[2],
                              signal9[2],signal10[2],signal11[2]]
                              )
       matrix_detail3.append([signal0[1],signal1[1],signal2[1],
                              signal3[1],signal4[1],signal5[1],
                              signal6[1],signal7[1],signal8[1],
                              signal9[1],signal10[1],signal11[1]]
                              )
   return np.array(matrix_detail1),np.array(matrix_detail2), np.array(matrix_detail3)


# Wavelets discretas
def apply_wavelet_transform(signal, wavelet='db3', levels=3):
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    return coeffs

def pad_signal_tensor(tensor, target_length=4096):
    n, m, channels = tensor.shape
    
    if m >= target_length:
        if m > target_length:
            print(f"")
        return tensor
    
    padding_needed = target_length - m
    
    if isinstance(tensor, np.ndarray):
        padding_zeros = np.zeros((n, padding_needed, channels), dtype=tensor.dtype)
        padded_tensor = np.concatenate([tensor, padding_zeros], axis=1)
    else:
        raise TypeError("El tensor debe ser un torch.Tensor o numpy.ndarray")
    
    print(f"Padding aplicado: {m} -> {target_length} (añadidos {padding_needed} ceros)")
    
    return padded_tensor




def log_memory(step=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
    print(f"[MEMORY][{step}] {mem:.2f} MB")
    
def create_signal_tensor(signals, signal_info, verbose, target_length):
    if not signals:
        return np.array([]), 0
    
    lengths = [signal.shape[0] for signal in signals]

    target_length = int(np.percentile(lengths, 95))
    
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

##Crea un dataset para que consuma menos memoria
 

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
    print(signal_tensor.shape)
    signal_tensor = pad_signal_tensor(signal_tensor)
    tensor_butter = filtro_señal_3d(signal_tensor)
    detail1,detail2,detail3 = wavelet_signals(tensor_butter)
    input_train = [detail1,detail2,detail3]

    selected_channels = ['I', 'II', 'III', 'V1', 'V2', 'V3']
    # Train the models on the signals.
    if verbose:
        print('Training the model on the signal data...')
        print(f'Number of signals: {len(all_signals)}')
        print(f'Signal tensor shape: {signal_tensor.shape}')
        print(f'Padded length: {padded_length}')
        
    all_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    selected_indices = [all_channels.index(ch) for ch in selected_channels]
    
    # Crear lista de señales solo con los canales seleccionados
    signal_list = [signal_tensor[:, :, i:i+1] for i in selected_indices]
    
    # dataset = build_dataset(signal_tensor, labels, batch_size=8)
    model = cnn_model2()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # y_train = labels
    # y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
    # y = np.argmax(y_train_onehot, axis=1) 
    # y = y.reshape(-1)
    
    # print("Y_train: ", y_train.shape)
    # print("y_train_onehot: ", y_train_onehot.shape)
    # print("y: ", y.shape)


    epochs = 40

    log_memory("Before training")
    cnn_model_history = model.fit([detail1,detail2,detail3],labels, epochs = epochs)

    log_memory("After Training")
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
    
    signal_tensor = pad_signal_tensor(signal_tensor)
    signal_tensor_raw = signal_tensor
    tensor_butter = filtro_señal_3d(signal_tensor_raw)
    detail1,detail2,detail3 = wavelet_signals(tensor_butter)
    input_train = [detail1,detail2,detail3]
    # signal_list = [signal_tensor[:, :, i:i+1] for i in range(signal_tensor.shape[2])]
    
    if verbose:
        print(f'Signal list length: {len(signal_list)}')
        print(f'Each signal shape: {signal_list[0].shape}')
    
    probability_output = model.predict(input_train, verbose=1)      
    
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
