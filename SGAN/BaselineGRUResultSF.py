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
TrainCaseName = 'SolarAllDataM3FFT'
#TrainCaseName = 'SolarAllDataM3FFT_DWT'

#TrainCaseName = 'WindAllDataM3FFT'
#TrainCaseName = 'WindAllDataM3FFT_DWT'

#TrainCaseName = 'BeligumAllDataM3FFT'
#TrainCaseName = 'BeligumAllDataM3FFT_DWT'

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
    train_yhat = model.predict(X_train, verbose=0)

    y_scaler = load(open(yScaler, 'rb'))
    train_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                  +"train_predict_index.npy", allow_pickle=True)

    rescaled_real_y = y_scaler.inverse_transform(y_train)
    rescaled_predicted_y = y_scaler.inverse_transform(train_yhat)

    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["predict_PW"],
                                 index=train_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_PW = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["real_PW"],
                               index=train_predict_index[i:i + output_dim])
        real_PW = pd.concat([real_PW, y_train], axis=1, sort=False)

    predict_result['predict_mean'] = predict_result.mean(axis=1)
    real_PW['real_mean'] = real_PW.mean(axis=1)

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_PW["Real_mean"])
    plt.plot(predict_result["predict_mean"], color='r')
    plt.xlabel("Date")
    plt.ylabel("value trend")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title("The result of Train", fontsize=20)
    plt.show()
    plt.savefig('./PICS/'+TrainCaseName +  '_traindataset.png')

    # Calculate RMSE
    predicted = predict_result["predict_mean"]
    real = real_PW["Real_mean"]
    For_MSE = pd.concat([predicted, real], axis=1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Train RMSE -- ', RMSE)

    return RMSE


# %% --------------------------------------- Plot the result  ----
def plot_testdataset_result(X_test, y_test):
    test_yhat = model.predict(X_test, verbose=0)
    y_scaler = load(open(yScaler, 'rb'))
    test_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                 +"test_predict_index.npy", allow_pickle=True)

    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(test_yhat)

    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["predicted_PW"],
                                 index=test_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_PW = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["real_PW"],
                               index=test_predict_index[i:i + output_dim])
        real_PW = pd.concat([real_PW, y_train], axis=1, sort=False)

    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_PW['real_mean'] = real_PW.mean(axis=1)

    Input_Before = '2020-01-01'
    predict_result = predict_result.loc[predict_result.index < Input_Before]
    real_PW = real_PW.loc[real_PW.index < Input_Before]

    print(predict_result.tail(10))

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_PW["real_mean"])
    plt.plot(predict_result["predicted_mean"], color='r')
    plt.xlabel("Date")
    plt.ylabel("value trend")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title("The result of Testing", fontsize=20)
    plt.show()
    plt.savefig('./PICS/' + TrainCaseName + '_testdataset.png')

    # Calculate RMSE
    predicted = predict_result["predicted_mean"]
    real = real_PW["real_mean"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Test RMSE -- ', RMSE)
    return RMSE

########################################################################################################################

train_RMSE = plot_traindataset_result(X_train, y_train)
print("----- Train_RMSE_LSTM -----", train_RMSE)

test_RMSE = plot_testdataset_result(X_test, y_test)
print("----- Test_RMSE_LSTM -----", test_RMSE)


