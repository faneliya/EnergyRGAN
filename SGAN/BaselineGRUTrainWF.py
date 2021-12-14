import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot
from pickle import load

# Constant Settings
DataFilesDir="./DataFiles/"
ProcessedFilesDir="./ProcessedFiles/"
ModelFileDir="./ModelSaves/"

######################################################################################################################
#TrainCaseName = 'SolarAllDataM3FFT'
#TrainCaseName = 'SolarAllDataM3FFT_DWT'
#TrainCaseName = 'WindAllDataM3FFT_DWT'
TrainCaseName = 'WindAllDataM3FFT'
#TrainCaseName = 'BelgiumAllDataM3FFT'
#TrainCaseName = 'BelgiumAllDataM3FFT_DWT'

# Parameters
LR = 0.0001
BATCH_SIZE = 128
N_EPOCH = 50
X_train = None
y_train = None
X_test = None
y_test = None
yc_train = None
yc_test = None
yScaler = None
xScaler = None
input_dim = None
feature_size = None
output_dim = None
# Load data

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

input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
output_dim = y_train.shape[1]


#######################################################################################################################
def basic_GRU(input_dim, output_dim, feature_size) -> tf.keras.models.Model:
    model = Sequential()
    model.add(GRU(units=256, return_sequences = True, input_shape=(input_dim, feature_size)))  # 256, return_sequences = True
    # model.add(GRU(units=256, recurrent_dropout = 0.2)) #, return_sequences = True
    model.add(GRU(units=128, input_shape=(input_dim, feature_size)))
    #model.add(Dense(128))
    model.add(Dense(64))
    # model.add(Dense(32))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(lr=LR), loss='mse')
    history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                        verbose=2, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.title(TrainCaseName + " GRU : loss/val_loss", fontsize=16)
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig('./PICS/'+TrainCaseName + '_GRU_lossResult.png')
    pyplot.show()


    return model
########################################################################################################################


model = basic_GRU(input_dim, output_dim, feature_size)
print(model.summary())
model.save(ModelFileDir + TrainCaseName + '_' + 'GRU_Model.h5')
