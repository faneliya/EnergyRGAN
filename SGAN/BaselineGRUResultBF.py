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
import tensorflow

# Constant Settings
DataFilesDir="./DataFiles/"
ProcessedFilesDir="./ProcessedFiles/"
ModelFileDir="./ModelSaves/"

######################################################################################################################
#TrainCaseName = 'SolarAllDataM3FFT'
#TrainCaseName = 'SolarAllDataM3FFT_DWT'

#TrainCaseName = 'WindAllDataM3FFT'
#TrainCaseName = 'WindAllDataM3FFT_DWT'

TrainCaseName = 'BelgiumAllDataM3FFT'
#TrainCaseName = 'BelgiumAllDataM3FFT_DWT'

# Parameters
LR = 0.0001
BATCH_SIZE = 128
N_EPOCH = 50

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


model = tensorflow.keras.models.load_model(ModelFileDir + TrainCaseName + '_' + 'GRU_30to1.h5')
print("PREDICT PROCESSING......" + TrainCaseName )

# %% --------------------------------------- Plot the result  -----------------------------------------------------------------
## TRAIN DATA
def plot_traindataset_result(X_train, y_train):

    print(TrainCaseName + "Predicting Data...START")
    train_yhat = model.predict(X_train, verbose=0)
    print(TrainCaseName + "Predicting Data...END")

    y_scaler = load(open(yScaler, 'rb'))
    train_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                  +"train_predict_index.npy", allow_pickle=True)

    rescaled_real_y = y_scaler.inverse_transform(y_train)
    rescaled_predicted_y = y_scaler.inverse_transform(train_yhat)

    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["PREDICT_VALUE"],
                                 index=train_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_value = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["REAL_VALUE"],
                               index=train_predict_index[i:i + output_dim])
        real_value = pd.concat([real_value, y_train], axis=1, sort=False)

    predict_result["PREDICTED_MEAN"] = predict_result.mean(axis=1)
    real_value["REAL_MEAN"] = real_value.mean(axis=1)

    predicted = predict_result["PREDICTED_MEAN"]
    real = real_value["REAL_MEAN"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_value["REAL_MEAN"])
    plt.plot(predict_result["PREDICTED_MEAN"], color='r')
    plt.xlabel("DATE")
    plt.ylabel("Real Value")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title(TrainCaseName + " GRU : result of Training, RMSE=" + str(RMSE), fontsize=16)
    plt.tight_layout()
    plt.savefig('./PICS/'+TrainCaseName + '_GRU_traindataset.png')
    plt.show()

    # Calculate RMSE
    print('-- Train RMSE -- ', RMSE)

    return RMSE


# %% --------------------------------------- Plot the result  ----
def plot_testdataset_result(X_test, y_test):

    print(TrainCaseName + "Predicting Data...START")
    test_yhat = model.predict(X_test, verbose=0)
    print(TrainCaseName + "Predicting Data...END")

    y_scaler = load(open(yScaler, 'rb'))
    test_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                 +"test_predict_index.npy", allow_pickle=True)

    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(test_yhat)

    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["PREDICT_VALUE"],
                                 index=test_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_value = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["REAL_VALUE"],
                               index=test_predict_index[i:i + output_dim])
        real_value = pd.concat([real_value, y_train], axis=1, sort=False)

    predict_result["PREDICTED_MEAN"] = predict_result.mean(axis=1)
    real_value["REAL_MEAN"] = real_value.mean(axis=1)

    # Calculate RMSE
    predicted = predict_result["PREDICTED_MEAN"]
    real = real_value["REAL_MEAN"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_value["REAL_MEAN"])
    plt.plot(predict_result["PREDICTED_MEAN"], color='r')
    plt.xlabel("DATE")
    plt.ylabel("Real Value")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title(TrainCaseName + " GRU : result of Training, RMSE=" + str(RMSE), fontsize=16)
    plt.tight_layout()
    plt.savefig('./PICS/'+TrainCaseName + '_GRU_testdataset.png')
    plt.show()


    print('-- Test RMSE -- ', RMSE)
    return RMSE

########################################################################################################################

train_RMSE = plot_traindataset_result(X_train, y_train)
print("----- Train_RMSE_LSTM -----", train_RMSE)

test_RMSE = plot_testdataset_result(X_test, y_test)
print("----- Test_RMSE_LSTM -----", test_RMSE)


