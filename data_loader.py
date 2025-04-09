# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 21:46:35 2025

@author: arian
"""

import pywt
import numpy as np
import h5py
from scipy.signal import butter,filtfilt
import pandas as pd
import os



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

def filtro_señal(signal):
    matriz_signal = []
    fil,col = signal.shape
    for i in range(col):
        filtrada = butter_bandpass_filter(signal[:,i],0.5, 150, 400, order=4)
        signal_sub = filtrada[0::4]
        matriz_signal.append(apply_wavelet_transform(signal_sub))
    return np.array(matriz_signal)

def apply_wavelet_transform(signal, wavelet='db3', levels=4):
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    value = np.concatenate((coeffs[1],coeffs[0]))
    return value

def signals_final(archivos,condicion):
    x = []
    y = []
    for i in range(len(archivos)):
        x.append(filtro_señal(archivos[i]))
        y.append(1) if condicion[i] else y.append(0)
    
    return np.array(x), np.array(y)

def load_data():
    if os.path.isdir("Samitrop"):
        df_samitrop = pd.read_csv('Samitrop/exams.csv')
        samitrop = h5py.File('Samitrop/exams.hdf5','r')
        archivos = samitrop['tracings']
        condicion = df_samitrop['normal_ecg']
        x,y = signals_final(archivos,condicion)
    else: 
        x = []
        y = []
    return x,y


def data_process():
    x1,y1 = load_data()
    indices = np.random.permutation(len(x1))
    
    # Calcular el punto de corte
    split_index = int(0.7 * len(x1))  # 70%
    
    # Usar los índices barajados para separar
    indices_70 = indices[:split_index]
    indices_30 = indices[split_index:]
    
    # Separar los datos
    x_train = x1[indices_70]
    y_train = y1[indices_70]
    x_val = x1[indices_30]
    y_val = y1[indices_30]
    
    return x_train,y_train, x_val, y_val

x_train,y_train, x_val, y_val = data_process()
    