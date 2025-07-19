
from data_loader import *
from CNN1D import * 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import numpy as np
import os
import tensorflow as tf
import numpy as np
from helper_code import load_record 


model = tf.keras.models.load_model('mi_modelo.h5')

input_folder = 'data'
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)

# Generar predicciones para cada registro
records = [f[:-4] for f in os.listdir(input_folder) if f.endswith('.hea')]
for record in records:
    data = load_record(os.path.join(input_folder, record))
    prediction = model.predict(data)

    # Obtener etiqueta binaria (por ejemplo umbral 0.5)
    binary_label = int(prediction >= 0.5)

    # Guardar resultados en formato requerido
    with open(os.path.join(output_folder, record + '.txt'), 'w') as f:
        f.write(f"{binary_label},{prediction[0]:.6f}")
