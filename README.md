# PhysioNet-Challenge-CIC_UPIITA_TEAM

This project implements a classifier for the annual George B. Moody PhysioNet Challenges, which aims to develop an automatic process for the identification of Chagas cases using electrocardiogram (ECG) signal data.

In this first installment, a proposal is presented that includes the preprocessing of ECG signals. Filters are applied to correct the out-of-phase signal and highlight important characteristics of the signals. Subsequently, a wavelet transform is used to obtain the approximation and details of the signals. These components are concatenated and used as input for a one-dimensional convolutional neural network (1D-CNN), designed to distinguish between people with and without Chagas.

## Project structure: 
- CNN1D.py: Contains the implementation of the one-dimensional convolutional neural network.
- data_loader.py: Loads and processes databases using biological signal processing techniques.
- run_code.py: Compile all code and calculate model performance metrics.
- requirements.txt: Contains all the necessary dependencies to run the project.

## How do I run these scripts?

First, you can install the dependencies for these scripts by creating a virtual environment and running the following command: 
Create the virtual environmen. 
```
conda create --name <my-env> 
```

```
pip install -r requirements.txt
```

