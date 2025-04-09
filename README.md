# PhysioNet-Challenge-CIC_UPIITA_TEAM

This project implements a classifier for the annual George B. Moody PhysioNet Challenges, which aims to develop an automatic process for the identification of Chagas cases using electrocardiogram (ECG) signal data.

In this first installment, a proposal is presented that includes the preprocessing of ECG signals. Filters are applied to correct the out-of-phase signal and highlight important characteristics of the signals. Subsequently, a wavelet transform is used to obtain the approximation and details of the signals. These components are concatenated and used as input for a one-dimensional convolutional neural network (1D-CNN), designed to distinguish between people with and without Chagas.

## Project structure: 
- CNN1D.py: Contiene la implementación de la red neuronal convolucional de una dimensión.
- data_loader.py: Carga y procesa las bases de datos utilizando técnicas de procesamiento de señales biológicas.
- run_code.py: Compila todo el código y calcula las métricas de rendimiento del modelo.
