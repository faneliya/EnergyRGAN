import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import GRU, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, ELU, ReLU
from tensorflow.keras import Sequential, regularizers
from tensorflow.python.client import device_lib

######################################################################################################################
# Constant Settings
DataFilesDir = "./DataFiles/"
ProcessedFilesDir = "./ProcessedFiles/"
ModelFileDir = "./ModelSaves/"

#TrainCaseName = 'SolarAllDataM3FFT'
# TrainCaseName = 'SolarAllDataM3FFT_DWT'
# TrainCaseName = 'WindAllDataM3FFT'
TrainCaseName = 'WindAllDataM3FFT_DWT'
# TrainCaseName = 'BelgiumAllDataM3FFT'
# TrainCaseName = 'BelgiumAllDataM3FFT_DWT'

DataVersion = 'simple_'

if TrainCaseName is not None:
    # Train Data 8 objects
    X_train = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "X_train.npy", allow_pickle=True)
    y_train = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "y_train.npy", allow_pickle=True)
    X_test = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "X_test.npy", allow_pickle=True)
    y_test = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "y_test.npy", allow_pickle=True)
    yc_train = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "yc_train.npy", allow_pickle=True)
    yc_test = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "yc_test.npy", allow_pickle=True)
    yScaler = ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "TargetValueScaler.pkl"
    xScaler = ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "BaseValueScaler.pkl"
else:
    X_train = np.load("X_train.npy", allow_pickle=True)
    y_train = np.load("y_train.npy", allow_pickle=True)
    X_test = np.load("X_test.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    yc_train = np.load("yc_train.npy", allow_pickle=True)
    yc_test = np.load("yc_test.npy", allow_pickle=True)
    yScaler = 'y_scaler.pkl'
    xScaler = 'X_scaler.pkl'


## TRAIN DATA
def plot_traindataset_result(RealValue, PredictedValue):
    X_scaler = load(open(xScaler, 'rb'))
    y_scaler = load(open(yScaler, 'rb'))
    train_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                  + DataVersion + "index_train.npy", allow_pickle=True)
    test_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                 + DataVersion + "index_test.npy", allow_pickle=True)

    print("----- predicted price -----", PredictedValue)

    rescaled_RealValue = y_scaler.inverse_transform(RealValue)
    rescaled_PredictedValue = y_scaler.inverse_transform(PredictedValue)

    print("----- rescaled predicted price -----", rescaled_PredictedValue)
    print("----- SHAPE rescaled predicted price -----", rescaled_PredictedValue.shape)

    predict_result = pd.DataFrame()
    for i in range(rescaled_PredictedValue.shape[0]):
        y_predict = pd.DataFrame(rescaled_PredictedValue[i], columns=["predicted_value"],
                                 index=train_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_value = pd.DataFrame()
    for i in range(rescaled_RealValue.shape[0]):
        y_train = pd.DataFrame(rescaled_RealValue[i], columns=["real_value"],
                               index=train_predict_index[i:i + output_dim])
        real_value = pd.concat([real_value, y_train], axis=1, sort=False)

    predict_result['PREDICTED_MEAN'] = predict_result.mean(axis=1)
    real_value['REAM_MEAN'] = real_value.mean(axis=1)

    # Calculate RMSE
    predicted = predict_result["PREDICTED_MEAN"]
    real = real_value["REAM_MEAN"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- RMSE -- ', RMSE)

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_value["REAM_MEAN"])
    plt.plot(predict_result["PREDICTED_MEAN"], color='r')
    plt.xlabel("DATE")
    plt.ylabel("Real Value")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title(TrainCaseName + " : GAN result of Training, RMSE=" + str(RMSE), fontsize=16)
    plt.tight_layout()
    plt.savefig('./PICS/' + TrainCaseName + '_wGAN_traindataset.png')
    plt.show()


def plot_testdataset_result(X_test, y_test, model):
    print(TrainCaseName + "Predicting Data...START")
    test_yhat = model.predict(X_test, verbose=0)
    print(TrainCaseName + "Predicting Data...END")

    y_scaler = load(open(yScaler, 'rb'))
    test_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion +
                                 "test_predict_index.npy", allow_pickle=True)

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
    plt.title(TrainCaseName + " wGAN : result of Training, RMSE=" + str(RMSE), fontsize=16)
    plt.tight_layout()
    plt.savefig('./PICS/' + TrainCaseName + '_wGAN_testdataset.png')
    plt.show()
    print('-- Test RMSE -- ', RMSE)
    return RMSE


if __name__ == '__main__':
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]
    epoch = 100

    ###############################################################################
    print('#Loading GAN Model-----------------')
    GanModel = tf.keras.models.load_model(ModelFileDir + TrainCaseName + '_' + 'WGanGeneratorModel.h5')
    # print(GanModel.summary())

    #################### plot train Model #########################################
    print("PREDICT DATASET PROCESSING......" + TrainCaseName)
    PredictedValue = GanModel.predict(X_train, verbose=0)
    plot_traindataset_result(y_train, PredictedValue)

    #################### plot test Model #########################################
    print("PREDICT TESTSET PROCESSING......" + TrainCaseName)
    plot_testdataset_result(X_test, y_test, GanModel)


