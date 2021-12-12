import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow

from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Embedding, TimeDistributed, LeakyReLU
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot
from pickle import load

######################################################################################################################
# Constant Settings
DataFilesDir="./DataFiles/"
ProcessedFilesDir="./ProcessedFiles/"
ModelFileDir="./ModelSaves/"

#TrainCaseName = 'SolarAllDataM3FFT'
#TrainCaseName = 'SolarAllDataM3FFT_DWT'
#TrainCaseName = 'WindAllDataM3FFT'
#TrainCaseName = 'WindAllDataM3FFT_DWT'
#TrainCaseName = 'BeligumAllDataM3FFT'
TrainCaseName = 'BeligumAllDataM3FFT_DWT'

if TrainCaseName is not None:
    #Train Data 8 objects
    X_train     = np.load(ProcessedFilesDir + TrainCaseName + "_" + "X_train.npy", allow_pickle=True)
    y_train     = np.load(ProcessedFilesDir + TrainCaseName + "_" + "y_train.npy", allow_pickle=True)
    X_test      = np.load(ProcessedFilesDir + TrainCaseName + "_" + "X_test.npy", allow_pickle=True)
    y_test      = np.load(ProcessedFilesDir + TrainCaseName + "_" + "y_test.npy", allow_pickle=True)
    yc_train    = np.load(ProcessedFilesDir + TrainCaseName + "_" + "yc_train.npy", allow_pickle=True)
    yc_test     = np.load(ProcessedFilesDir + TrainCaseName + "_" + "yc_test.npy", allow_pickle=True)
    yScaler     = ProcessedFilesDir + TrainCaseName + "_" + "TargetValueScaler.pkl"
    xScaler     = ProcessedFilesDir + TrainCaseName + "_" + "BaseValueScaler.pkl"
else:
    X_train = np.load("X_train.npy", allow_pickle=True)
    y_train = np.load("y_train.npy", allow_pickle=True)
    X_test = np.load("X_test.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    yc_train = np.load("yc_train.npy", allow_pickle=True)
    yc_test = np.load("yc_test.npy", allow_pickle=True)
    yScaler = 'y_scaler.pkl'
    xScaler = 'X_scaler.pkl'


#Parameters
LR = 0.001
BATCH_SIZE = 64
N_EPOCH = 50

input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
output_dim = y_train.shape[1]

####
def basic_lstm(input_dim, feature_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(units= 128), input_shape=(input_dim, feature_size)))
    model.add(Dense(64))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(lr = LR), loss='mse')
    history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                        verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

    return model
####


print('Modeling LSTM ..............>' + TrainCaseName)
model = basic_lstm(input_dim, feature_size)
model.save(ModelFileDir + TrainCaseName + '_' + 'LSTM_3to1.h5')

print(model.summary())








