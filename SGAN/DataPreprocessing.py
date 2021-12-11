import os
import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

DataFilesDir="./DataFiles/"
ProcessedFilesDir="./ProcessedFiles/"
STD_BASE_TIME_TIC = 240
STD_TARGET_TIME_TIC = 8


def dataProcessByFile(fileName):
    if fileName is None:
        return

    # %% - Load Data  -----------------------------------------------------------------
    dataset = pd.read_csv(DataFilesDir + fileName, parse_dates=['DATE'])
    #dataset.replace(0, np.nan, inplace=True)
    dataset.isnull().sum()
    print(dataset.columns)
    # Set the date to datetime data
    datetime_series = pd.to_datetime(dataset['DATE'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    dataset = dataset.set_index(datetime_index)
    dataset = dataset.sort_values(by='DATE')
    dataset = dataset.drop(columns='DATE')

    # 아래 항목은 데이터 처리중 불필요및 오류발생
    # dataset = dataset.drop(columns='IDX')
    dataset = dataset.drop(columns='logmomentum')

    print(dataset.columns)
    # Check NA and fill them
    dataset.iloc[:, 0:] = pd.concat([dataset.iloc[:, 0:].ffill(), dataset.iloc[:, 0:].bfill()]).groupby(level=0).mean()
    # Get features and target
    BaseValue = pd.DataFrame(dataset.iloc[:, :]) # 모든 열
    TargetValue = pd.DataFrame(dataset.iloc[:, 0]) # 목포값 첫번째 열

    print(BaseValue)
    print(TargetValue)
    # Autocorrelation Check
    sm.graphics.tsa.plot_acf(TargetValue.squeeze(), lags=100)
    plt.show()

    # Normalized the data
    BaseValueScaler = MinMaxScaler(feature_range=(-1, 1))
    TargetValueScaler = MinMaxScaler(feature_range=(-1, 1))
    BaseValueScaler.fit(BaseValue)
    TargetValueScaler.fit(TargetValue)

    BaseScaleDataset = BaseValueScaler.fit_transform(BaseValue)
    TargetScaleDataset = TargetValueScaler.fit_transform(TargetValue)

    dump(BaseValueScaler, open(ProcessedFilesDir+ fileName[:-4]+'_BaseValueScaler.pkl', 'wb'))
    dump(TargetValueScaler, open(ProcessedFilesDir+ fileName[:-4]+'_TargetValueScaler.pkl', 'wb'))

    # Reshape the data
    '''Set the data input steps and output steps, 
        we use 30 days data to predict 1 day price here, 
        reshape it to (None, input_step, number of features) used for LSTM input'''
    # 4 x 24 x 30 = 2880
    # 4 x 24 = 96

    n_steps_in = STD_BASE_TIME_TIC
    # n_features = BaseValue.shape[1]
    n_steps_out = STD_TARGET_TIME_TIC

    # Get data and check shape ##################################################
    X, y, yc = getDataSlide(BaseScaleDataset, TargetScaleDataset, n_steps_in, n_steps_out)
    # ###########################################################################
    X_train, X_test, = split_train_test(X,X)
    y_train, y_test, = split_train_test(y,X)
    yc_train, yc_test, = split_train_test(yc,X)
    index_train, index_test, = predict_index(dataset, X_train, n_steps_in, n_steps_out)

    # %% - Save dataset -----------------------------------------------------------------
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('y_c_train shape: ', yc_train.shape)
    print('X_test shape: ', X_test.shape)
    print('y_test shape: ', y_test.shape)
    print('y_c_test shape: ', yc_test.shape)
    print('index_train shape:', index_train.shape)
    print('index_test shape:', index_test.shape)

    fileNamePreFix = fileName[:-4]
    np.save(ProcessedFilesDir+ fileNamePreFix+"_X_train.npy", X_train)
    np.save(ProcessedFilesDir+ fileNamePreFix+"_y_train.npy", y_train)
    np.save(ProcessedFilesDir+ fileNamePreFix+"_X_test.npy", X_test)
    np.save(ProcessedFilesDir+ fileNamePreFix+"_y_test.npy", y_test)
    np.save(ProcessedFilesDir+ fileNamePreFix+"_yc_train.npy", yc_train)
    np.save(ProcessedFilesDir+ fileNamePreFix+"_yc_test.npy", yc_test)
    np.save(ProcessedFilesDir+ fileNamePreFix+'_index_train.npy', index_train)
    np.save(ProcessedFilesDir+ fileNamePreFix+'_index_test.npy', index_test)
    np.save(ProcessedFilesDir+ fileNamePreFix+'_train_predict_index.npy', index_train)
    np.save(ProcessedFilesDir+ fileNamePreFix+'_test_predict_index.npy', index_test)


def dataProcess(filePreFixName):
    if filePreFixName is None:
        return
    elif filePreFixName == "Wind":
        preFix = "Wind"
    elif filePreFixName == "Solar":
        preFix = "Solar"
    else:
        return

    # %% - Load Data  -----------------------------------------------------------------
    dataset = pd.read_csv(DataFilesDir + preFix + "DataFFT.csv", parse_dates=['DATE'])
    #dataset.replace(0, np.nan, inplace=True)
    dataset.isnull().sum()
    print(dataset.columns)
    # Set the date to datetime data
    datetime_series = pd.to_datetime(dataset['DATE'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    dataset = dataset.set_index(datetime_index)
    dataset = dataset.sort_values(by='DATE')
    dataset = dataset.drop(columns='DATE')

    # 아래 항목은 데이터 처리중 불필요및 오류발생
    # dataset = dataset.drop(columns='IDX')
    dataset = dataset.drop(columns='logmomentum')

    print(dataset.columns)
    # Check NA and fill them
    dataset.iloc[:, 0:] = pd.concat([dataset.iloc[:, 0:].ffill(), dataset.iloc[:, 0:].bfill()]).groupby(level=0).mean()
    # Get features and target
    BaseValue = pd.DataFrame(dataset.iloc[:, :]) # 모든 열
    TargetValue = pd.DataFrame(dataset.iloc[:, 0]) # 목포값 첫번째 열

    print(BaseValue)
    print(TargetValue)
    # Autocorrelation Check
    sm.graphics.tsa.plot_acf(TargetValue.squeeze(), lags=100)
    plt.show()

    # Normalized the data
    BaseValueScaler = MinMaxScaler(feature_range=(-10, 10))
    TargetValueScaler = MinMaxScaler(feature_range=(-10, 10))
    BaseValueScaler.fit(BaseValue)
    TargetValueScaler.fit(TargetValue)

    BaseScaleDataset = BaseValueScaler.fit_transform(BaseValue)
    TargetScaleDataset = TargetValueScaler.fit_transform(TargetValue)

    dump(BaseValueScaler, open(ProcessedFilesDir+ preFix+'BaseValueScaler.pkl', 'wb'))
    dump(TargetValueScaler, open(ProcessedFilesDir+ preFix+'TargetValueScaler.pkl', 'wb'))

    # Reshape the data
    '''Set the data input steps and output steps, 
        we use 30 days data to predict 1 day price here, 
        reshape it to (None, input_step, number of features) used for LSTM input'''
    # 4 x 24 x 30 = 2880
    # 4 x 24 = 96

    n_steps_in = STD_BASE_TIME_TIC
    # n_features = BaseValue.shape[1]
    n_steps_out = STD_TARGET_TIME_TIC

    # Get data and check shape ##################################################
    X, y, yc = getDataSlide(BaseScaleDataset, TargetScaleDataset, n_steps_in, n_steps_out)
    # ###########################################################################
    X_train, X_test, = split_train_test(X,X)
    y_train, y_test, = split_train_test(y,X)
    yc_train, yc_test, = split_train_test(yc,X)
    index_train, index_test, = predict_index(dataset, X_train, n_steps_in, n_steps_out)

    # %% - Save dataset -----------------------------------------------------------------
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('y_c_train shape: ', yc_train.shape)
    print('X_test shape: ', X_test.shape)
    print('y_test shape: ', y_test.shape)
    print('y_c_test shape: ', yc_test.shape)
    print('index_train shape:', index_train.shape)
    print('index_test shape:', index_test.shape)

    np.save(ProcessedFilesDir+ preFix+"_X_train.npy", X_train)
    np.save(ProcessedFilesDir+ preFix+"_y_train.npy", y_train)
    np.save(ProcessedFilesDir+ preFix+"_X_test.npy", X_test)
    np.save(ProcessedFilesDir+ preFix+"_y_test.npy", y_test)
    np.save(ProcessedFilesDir+ preFix+"_yc_train.npy", yc_train)
    np.save(ProcessedFilesDir+ preFix+"_yc_test.npy", yc_test)
    np.save(ProcessedFilesDir+ preFix+'_index_train.npy', index_train)
    np.save(ProcessedFilesDir+ preFix+'_index_test.npy', index_test)
    np.save(ProcessedFilesDir+ preFix+'_train_predict_index.npy', index_train)
    np.save(ProcessedFilesDir+ preFix+'_test_predict_index.npy', index_test)


# Get X/y dataset
def getDataSlide(X_data, y_data, n_steps_in, n_steps_out):
    X = list()
    y = list()
    yc = list()
    length = len(X_data)
    print("> Total Data Size(Row)  =" + str(length))
    for i in range(0, length, 1):
        # print(i)
        BaseValue = X_data[i: i + n_steps_in][:, :]
        TargetValue = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        TargetAllValue = y_data[i: i + n_steps_in][:, :]
        #print(BaseValue)
        #print(TargetValue)
        #print(len(BaseValue))
        #print(len(TargetValue))
        if len(BaseValue) == STD_BASE_TIME_TIC and len(TargetValue) == STD_TARGET_TIME_TIC:
            X.append(BaseValue)
            y.append(TargetValue)
            yc.append(TargetAllValue)

    return np.array(X), np.array(y), np.array(yc)


# get the train test predict index
def predict_index(dataset, X_train, n_steps_in, n_steps_out):

    # get the predict data (remove the in_steps days)
    train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index

    return train_predict_index, test_predict_index


# Split train/test dataset
def split_train_test(data, X):
    train_size = round(len(X) * 0.7)
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test


# 프로그램 시작처리
if __name__ == '__main__':
    print('################ Data PreProcessing #########################')
    #dataProcess("Solar")
    #dataProcess("Wind")
    #dataProcessByFile("BelgiumAllDataFFT.csv")
    #dataProcessByFile("SolarAllDataM3FFT.csv")
    #dataProcessByFile("SolarAllDataM3FFT_DWT.csv")
    #dataProcessByFile("WindAllDataM3FFT.csv")
    #dataProcessByFile("WindAllDataM3FFT_DWT.csv")

    dataProcessByFile("BelgiumAllDataM3FFT.csv")
    dataProcessByFile("BelgiumAllDataM3FFT_DWT.csv")
'''
    dataProcessByFile("SolarAllDataM6FFT.csv")
    dataProcessByFile("SolarAllDataM6FFT_DWT.csv")
    dataProcessByFile("WindAllDataM6FFT.csv")
    dataProcessByFile("WindAllDataM6FFT_DWT.csv")

    dataProcessByFile("SolarAllDataY1FFT.csv")
    dataProcessByFile("SolarAllDataY1FFT_DWT.csv")
    dataProcessByFile("WindAllDataY1FFT.csv")
    dataProcessByFile("WindAllDataY1FFT_DWT.csv")

    dataProcessByFile("SolarAllDataY2FFT.csv")
    dataProcessByFile("SolarAllDataY2FFT_DWT.csv")
    dataProcessByFile("WindAllDataY2FFT.csv")
    dataProcessByFile("WindAllDataY2FFT_DWT.csv")

    dataProcessByFile("SolarAllDataFFT.csv")
    dataProcessByFile("SolarAllDataFFT_DWT.csv")
    dataProcessByFile("WindAllDataFFT.csv")
    dataProcessByFile("WindAllDataFFT_DWT.csv")

    dataProcessByFile("BelgiumAllDataFFT.csv")
    dataProcessByFile("BelgiumAllDataFFT_DWT.csv")
'''