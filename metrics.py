#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 17:18:18 2025

@author: cic
"""

import joblib
import numpy as np
from scipy.signal import butter, filtfilt
import pywt
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from CNN1D import *
from helper_code import *
from team_code import *
import tensorflow as tf
import psutil
import matplotlib.pyplot as plt
import os
from sklearn.utils import class_weight
from tensorflow.keras.utils import plot_model

verbose = True

data_folder = '../debug_data'
# filename = '../model/model.keras'
# model = load_model(model_folder, verbose)
model = tf.keras.models.load_model('model.keras')
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

#canales = [0, 4, 6, 8]
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


#%%
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    balanced_accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import numpy as np
import os

# === Inputs ===
input_train = [detail1, detail2, detail3]
labels = np.array(all_labels, dtype=bool)

# === Create folder for evaluation results ===
os.makedirs("model_evaluation", exist_ok=True)

# === Model Predictions ===
probability_output = model.predict(input_train, verbose=1)
binary_outputs = (probability_output >= 0.5).astype(int).reshape(-1)
y_true = labels.reshape(-1)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, binary_outputs)
print("\n📊 Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', colorbar=False)
plt.title("Confusion Matrix - Model Evaluation")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("model_evaluation/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# === Classification Report ===
print("\n Classification Report:")
print(classification_report(
    y_true,
    binary_outputs,
    digits=4,
    zero_division=0
))

# === Balanced Accuracy ===
bal_acc = balanced_accuracy_score(y_true, binary_outputs)
print(f" Balanced Accuracy: {bal_acc:.4f}")

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_true, probability_output)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("model_evaluation/roc_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# === Precision-Recall Curve ===
precision, recall, _ = precision_recall_curve(y_true, probability_output)
avg_precision = average_precision_score(y_true, probability_output)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f"AP = {avg_precision:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.savefig("model_evaluation/precision_recall_curve.png", dpi=300, bbox_inches='tight')
plt.show()
