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

def centered_window(signal, window_size=500):
    n_samples, n_channels = signal.shape

    if n_samples >= window_size:
        # Índice central
        center = n_samples // 2
        start = max(center - window_size // 2, 0)
        end = start + window_size
        # Ajuste por si la señal es ligeramente más corta que la ventana al final
        if end > n_samples:
            start = n_samples - window_size
            end = n_samples
        window = signal[start:end, :]
    else:
        # Si la señal es más corta que la ventana, rellenar con promedio
        avg = signal.mean(axis=0)
        window = np.zeros((window_size, n_channels), dtype=signal.dtype)
        window[:n_samples, :] = signal
        window[n_samples:, :] = avg

    return window.astype(np.float32)

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


def downsample_to_100(signal, fs_original, num_samples=None):
    """
    Downsamplea la señal a 100 Hz haciendo promedio por bloques.
    signal: np.array, forma (num_muestras, num_canales)
    fs_original: frecuencia original (400, 500, etc.)
    num_samples: si se proporciona, se recortan las primeras num_samples muestras
    """
    fs_original = int(fs_original)
    if fs_original < 100:
        raise ValueError(f"fs_original ({fs_original}) debe ser >= 100")

    # Recortar la señal si num_samples está dado
    if num_samples is not None:
        signal = signal[:num_samples, :]

    n_samples, n_channels = signal.shape
    factor = n_samples // (n_samples * 100 // fs_original)  # número de muestras por bloque
    if factor <= 0:
        factor = 1

    # Ajustar longitud para que sea divisible por factor
    usable_length = (n_samples // factor) * factor
    signal = signal[:usable_length, :]

    # Reshape y promedio por bloques
    signal = signal.reshape(-1, factor, n_channels)
    downsampled = signal.mean(axis=1)

    return downsampled.astype(np.float32)
            

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
    os.makedirs('temp_fs', exist_ok=True)
    os.makedirs('temp_ns', exist_ok=True)
    
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

        lines = header.splitlines()
        
        # Primera línea: número de muestras y frecuencia
        num_samples = int(lines[0].split()[3])
        fs = int(lines[0].split()[2])
        
        print(f"Número de muestras: {num_samples}, Frecuencia: {fs} Hz")
                
        
        channels = fields['sig_name']
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        signal = reorder_signal(signal, channels, reference_channels)
    
        # Reducir tamaño de datos y convertir a float32 para ahorrar RAM
        signal = signal.astype(np.float32)
    
        # Guardar temporalmente
        base_name = os.path.basename(rec_name)
        
        np.save(f'temp_signals/{base_name}.npy', signal)
        np.save(f'temp_labels/{base_name}.npy', label)
        np.save(f'temp_fs/{base_name}.npy', fs)
        np.save(f'temp_ns/{base_name}.npy', num_samples)

    
    if verbose:
        print(f'{num_records} signals processed and saved temporarily.')
    
    signal_folder = 'temp_signals'
    label_folder = 'temp_labels'
    fs_folder = 'temp_fs'
    ns_folder = 'temp_ns'

    # canales = [0, 4, 6, 8]
    canales = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    print('num_canales: ', canales)
    signal_files = os.listdir(signal_folder)
    num_files = len(signal_files)
    print(num_files)
    
    # Listas para ir almacenando los batches procesados
    all_detail1 = []
    all_detail2 = []
    all_detail3 = []
    all_labels = []
    
    all_signals = []
    labels_batch = []
    
        
    for f in signal_files:
        signal = np.load(os.path.join(signal_folder, f))
        fs = np.load(os.path.join(fs_folder, f))
        num_samples = np.load(os.path.join(ns_folder, f))
    
        # Downsample a 100 Hz
        signal = downsample_to_100(signal, fs, num_samples)
        label = np.load(os.path.join(label_folder, f))
        windowed_signal = centered_window(signal, window_size=500)
        all_signals.append(windowed_signal)
        all_labels.append(label)

            
    labels_batch = np.asarray(labels_batch, dtype=bool)

    
    all_signals = np.array(all_signals, dtype=np.float32)  # (num_signals, 500, 12)
    labels = np.array(all_labels, dtype=bool)          # (num_signals, ...)
    
    # Filtrado y wavelet
    tensor_butter = filtro_señal_3d(all_signals)
    detail1, detail2, detail3 = wavelet_signals(tensor_butter)
    
    # Selección de canales

    detail1 = select_chanels(detail1, canales)
    detail2 = select_chanels(detail2, canales)
    detail3 = select_chanels(detail3, canales)
    
    print(detail1.shape, detail2.shape, detail3.shape)
    print(labels.shape)
        
    
    model = cnn_model2(detail1)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    epochs = 50
    

    clases, conteos = np.unique(labels, return_counts=True)

    for c, n in zip(clases, conteos):
        print(f"Clase {c}: {n} muestras")
   
    cnn_model_history = model.fit([detail1, detail2, detail3], labels, epochs=epochs, validation_split=0.1)
    
    log_memory("Before training")
    
    plt.bar(clases.astype(str), conteos)
    plt.xlabel("Clases")
    plt.ylabel("Número de muestras")
    plt.title("Distribución de clases (True/False)")
    plt.show()
    


    history_dict = cnn_model_history.history
    print(history_dict.keys())
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(cnn_model_history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(cnn_model_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
    
    plt.title("Model Accuracy over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("metrics_accuracy.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # === Plot 2: Loss (Training vs Validation) ===
    plt.figure(figsize=(8, 6))
    plt.plot(cnn_model_history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(cnn_model_history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--', color='orange')
    
    plt.title("Model Loss over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("metrics_loss.png", dpi=300, bbox_inches='tight')
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


def run_model(record, model, verbose):
    # Load the model.
    print(record)
    signal, fields = load_signals(record)
    
    header = load_header(record)
    label = load_label(record)

    lines = header.splitlines()
            
    #         # Primera línea: número de muestras y frecuencia
    num_samples = int(lines[0].split()[3])
    fs = int(lines[0].split()[2])
            
    print(f"Número de muestras: {num_samples}, Frecuencia: {fs} Hz")
                    
            
    channels = fields['sig_name']
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    signal = reorder_signal(signal, channels, reference_channels)
        
    #         # Reducir tamaño de datos y convertir a float32 para ahorrar RAM
    signal = signal.astype(np.float32)
    
    print('Signal before pad -> ', signal.shape)
    signal = downsample_to_100(signal, fs, num_samples)
    
    windowed_signal = centered_window(signal, window_size=500)
    print('Signal after pad -> ', windowed_signal.shape)
    signal = np.expand_dims(windowed_signal, axis=0)

    tensor_butter = filtro_señal_3d(signal)
    detail1, detail2, detail3 = wavelet_signals(tensor_butter)
        
    #     # Selección de canales
    canales = [0,1,2,3,4,5,6,7,8,9,10,11]
    detail1 = select_chanels(detail1, canales)
    detail2 = select_chanels(detail2, canales)
    detail3 = select_chanels(detail3, canales)
        
        
        
    input_train = [detail1,detail2,detail3]
    # Extract the features.

    # Get the model outputs.
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
    model.save('model.h5')
