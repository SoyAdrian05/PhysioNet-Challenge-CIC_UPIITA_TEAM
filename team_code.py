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
import matplotlib.pyplot as plt
import os
from sklearn.utils import class_weight
from tensorflow.keras.utils import plot_model


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
            print(f"Señal recortada: {m} -> {target_length}")
            tensor = tensor[:, :target_length, :]
        return tensor

    padded_tensor = np.zeros((n, target_length, channels), dtype=tensor.dtype)

    original_idx = np.arange(m)
    new_idx = np.linspace(0, m-1, target_length)

    for i in range(n):
        for c in range(channels):
            padded_tensor[i, :, c] = np.interp(new_idx, original_idx, tensor[i, :, c])

    print(f"Padding por interpolación aplicado: {m} -> {target_length}")
    return padded_tensor

def select_chanels(tensor,canales):
    num_canales = tensor.shape[1]
    for idx in canales:
        if idx < 0 or idx >= num_canales:
            raise ValueError(f"Índice {idx} fuera del rango válido [0, {num_canales-1}]")
    
    tensor_final = tensor[:, canales, :]
    
    return tensor_final


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

# Train your model.
def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')
    
    records = find_records(data_folder)
    num_records = len(records)
    
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    # Crear carpetas temporales si no existen
    os.makedirs('temp_signals', exist_ok=True)
    os.makedirs('temp_labels', exist_ok=True)
    
    if verbose:
        print('Extracting signals and labels from the data...')
    
    # Iterar sobre los registros
    for i, rec_name in enumerate(records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {rec_name}...')
    
        record_path = os.path.join(data_folder, rec_name)
    
        # Cargar señal y metadata
        signal, fields = load_signals(record_path)
        header = load_header(record_path)
        label = load_label(record_path)
    
        channels = fields['sig_name']
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        signal = reorder_signal(signal, channels, reference_channels)
    
        # Reducir tamaño de datos y convertir a float32 para ahorrar RAM
        signal = signal.astype(np.float32)
    
        # Guardar temporalmente
        np.save(f'temp_signals/{rec_name}.npy', signal)
        np.save(f'temp_labels/{rec_name}.npy', label)
    
    if verbose:
        print(f'{num_records} signals processed and saved temporarily.')
    
    signal_folder = 'temp_signals'
    label_folder = 'temp_labels'
    batch_size = 16
    canales = [0, 4, 6, 8]
    canales = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    signal_files = os.listdir(signal_folder)
    num_files = len(signal_files)
    
    # Listas para ir almacenando los batches procesados
    all_detail1 = []
    all_detail2 = []
    all_detail3 = []
    all_labels = []
    
    for i in range(0, num_files, batch_size):
        batch_files = signal_files[i:i+batch_size]
    
        all_signals = []
        labels_batch = []
    
        for f in batch_files:
            signal = np.load(os.path.join(signal_folder, f))
            label = np.load(os.path.join(label_folder, f))
            all_signals.append(signal)
            labels_batch.append(label)
    
        labels_batch = np.asarray(labels_batch, dtype=bool)
    
        # Crear tensor de señal
        signal_tensor, padded_length = create_signal_tensor(all_signals, signal_info=None, verbose=False, target_length=False)
    
        # Padding y filtrado
        signal_tensor = pad_signal_tensor(signal_tensor)
        tensor_butter = filtro_señal_3d(signal_tensor)
        detail1_batch, detail2_batch, detail3_batch = wavelet_signals(tensor_butter)
    
        # Selección de canales
        detail1_batch = select_chanels(detail1_batch, canales)
        detail2_batch = select_chanels(detail2_batch, canales)
        detail3_batch = select_chanels(detail3_batch, canales)
    
        # Guardar batch en listas
        all_detail1.append(detail1_batch)
        all_detail2.append(detail2_batch)
        all_detail3.append(detail3_batch)
        all_labels.append(labels_batch)
    
    # Concatenar todos los batches
    detail1 = np.concatenate(all_detail1, axis=0)
    detail2 = np.concatenate(all_detail2, axis=0)
    detail3 = np.concatenate(all_detail3, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(detail1.shape, detail2.shape, detail3.shape)
    print(labels.shape)
    
        
        # labels = tu vector booleano
    labels_int = labels.astype(int)  # convertir a 0/1
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_int),
        y=labels_int
    )
    class_weights = dict(enumerate(class_weights))
    print(class_weights)
    
    
    model = cnn_model2(detail1)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    epochs = 100
    
    idx = np.random.randint(0, detail1.shape[0])
    


    clases, conteos = np.unique(labels, return_counts=True)

    for c, n in zip(clases, conteos):
        print(f"Clase {c}: {n} muestras")
   
    cnn_model_history = model.fit(
    [detail1, detail2, detail3],
    labels.astype(np.float32),
    epochs=epochs,
    class_weight=class_weights
    )
    
    log_memory("Before training")
    
    plt.bar(clases.astype(str), conteos)
    plt.xlabel("Clases")
    plt.ylabel("Número de muestras")
    plt.title("Distribución de clases (True/False)")
    plt.show()
    


    history_dict = cnn_model_history.history
    print(history_dict.keys())
    
    
    plt.plot(cnn_model_history.history['accuracy'], label='Entrenamiento')
    # plt.plot(cnn_model_history.history['val_acc'], label='Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("metrics.png")
    plt.grid(True)
    plt.show()

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
    canales = [0,4,6,8]
    signal_tensor = pad_signal_tensor(signal_tensor)
    signal_tensor_raw = signal_tensor
    tensor_butter = filtro_señal_3d(signal_tensor_raw)
    detail1,detail2,detail3 = wavelet_signals(tensor_butter)
    detail1 = select_chanels(detail1,canales)
    detail2 = select_chanels(detail2,canales)
    detail3 = select_chanels(detail3,canales)
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
