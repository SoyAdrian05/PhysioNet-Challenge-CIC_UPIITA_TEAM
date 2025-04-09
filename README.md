# PhysioNet-Challenge-CIC_UPIITA_TEAM

This project implements a classifier for the annual George B. Moody PhysioNet Challenges, which aims to develop an automatic process for the identification of Chagas cases using electrocardiogram (ECG) signal data.

In this first installment, a proposal is presented that includes the preprocessing of ECG signals. Filters are applied to correct the out-of-phase signal and highlight important characteristics of the signals. Subsequently, a wavelet transform is used to obtain the approximation and details of the signals. These components are concatenated and used as input for a one-dimensional convolutional neural network (1D-CNN), designed to distinguish between people with and without Chagas.

## Project structure: 
- CNN1D.py: Contains the implementation of the one-dimensional convolutional neural network.
- data_loader.py: Loads and processes databases using biological signal processing techniques.
- run_code.py: Compile all code and calculate model performance metrics.
- requirements.txt: Contains all the necessary dependencies to run the project.

## How do I run these scripts?

First, you can install the dependencies for these scripts by creating a virtual environment and running the following command
```
conda create --name <my-env> 
```

```
pip install -r requirements.txt
```

Then, you can run the file `run_code.py` with the following command to train the model and evaluate the efficiency with the metrics: 
```
python run_code.py
```

## Load Data 
This project exclusively uses the SAMI-TROP database. In order to run the program correctly, this database needs to be loaded. Here's an example of how data is loaded into your code:
```
if os.path.isdir("Samitrop"):
    df_samitrop = pd.read_csv('Samitrop/exams.csv')
    samitrop = h5py.File('Samitrop/exams.hdf5','r')
```
Make sure you have the Samitrop folder in the working directory, which contains the exams.csv and exams.hdf5 files. The exams.csv file should include the tags needed for model training and evaluation, while exams.hdf5 should contain the electrocardiogram signals.

## Note
This is a first approximation in which we focus on the processing of the signals before applying the classification algorithm. The main objective of this proposal is to highlight the distinctive characteristics of each signal and provide the algorithm with a clean and optimized signal, thus facilitating its classification.

## Authors
1. Team Member 1:
   - Name: José-Adrián Chávez-Olvera
   - Email: jchavezo1603@alumno.ipn.mx
2. Team Member 2:
   - Name: Arián Villalba-Tapia
   - Email: avillalbat1600@alumno.ipn.mx
3. Team Member 3:
   - Name: Blanca Tovar-Corona
   - Email: bltovar@ipn.mx
4. Team Member 4:
   - Name: René Luna-García
   - Email: rlunag@ipn.mx
