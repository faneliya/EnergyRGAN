import warnings

import data.excelLoader as el
import data.dataLoader as dl

import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
base_dir = 'C:/DEVEL/DEVEL_DATA/DataTest/belium_weather_max_temp_blend'

#el.read_weather_data_from_excel(base_dir, None)

sqlText = "SELECT	CONCAT(WD.REG_YMD, WD.REG_HH24, WD.REG_MM) AS DATE," \
          "	PW_P AS PW," \
          "	CAPACITY_MW AS CAPA," \
          "	TD.TEMP_MAX " \
          "FROM	AST0102 WD,	AST0301 TD " \
          "WHERE	1 = 1	" \
          "AND WD.REG_YMD = TD.REC_YMD	" \
          "AND WD.REG_YMD BETWEEN '20191101' AND '20210531' "

# dl.dataLoadSQL(sqlText, "windData.csv")


## import data
df = pd.read_csv('windData.csv')
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)

fig, ax = plt.subplots(figsize=(10,3))
ax.plot(df[ 'IDX'], df['PW'], label='WIND_POWER')
ax.set(xlabel="Date",
       ylabel="MW",
       title="WIND_POWER BELGIUM")
#date_form = DateFormatter("%Y%m%d%H%M")
#ax.xaxis.set_major_formatter(date_form)
plt.show()


def get_technical_indicators(data):
    # Create 7 and 21 days Moving Average
    data['MA7'] = data.iloc[:,2].rolling(window=7).mean()
    data['MA21'] = data.iloc[:,2].rolling(window=21).mean()

    # Create MACD
    data['MACD'] = data.iloc[:,2].ewm(span=26).mean() - data.iloc[:,1].ewm(span=12,adjust=False).mean()

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 2].rolling(20).std()
    data['upper_band'] = data['MA21'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA21'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:,2].ewm(com=0.5).mean()

    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:,2] - 1)

    return data


warnings.filterwarnings(action='ignore')
T_df = get_technical_indicators(df)
warnings.filterwarnings(action='default')

dataset = T_df.iloc[20:,:].reset_index(drop=True)


def get_fourier_transfer(dataset):
    # Get the columns for doing fourier
    data_FT = dataset[['IDX', 'PW']]

    pw_fft = np.fft.fft(np.asarray(data_FT['PW'].tolist()))
    fft_df = pd.DataFrame({'fft': pw_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    fft_com_df = pd.DataFrame()

    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0

        fft_ = np.fft.ifft(fft_list_m10)
        fft_com = pd.DataFrame({'fft': fft_})
        fft_com['absolute of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.abs(x))
        fft_com['angle of ' + str(num_) + ' comp'] = fft_com['fft'].apply(lambda x: np.angle(x))
        fft_com = fft_com.drop(columns='fft')
        fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)

    return fft_com_df


dataset_F = get_fourier_transfer(dataset)
Final_data = pd.concat([dataset, dataset_F], axis=1)
print(Final_data.head())
Final_data.to_csv("WindData_with_Fourier.csv", index=False)


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    dataset = dataset.iloc[-last_days:, :]
    x_ = list(dataset.index)

    # Plot first subplot
    plt.plot(dataset['MA7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['PW'], label='Wind Power', color='b')
    plt.plot(dataset['MA21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('WindPower Band'.format(last_days))
    plt.ylabel('Power MW')
    plt.legend()
    plt.legend()
    plt.show()


plot_technical_indicators(T_df, 5000)


def plot_Fourier(dataset):
    data_FT = dataset[['IDX', 'PW']]

    #print(data_FT.info())
    PW_fft = np.fft.fft(np.asarray(data_FT['PW'].tolist()))
    fft_df = pd.DataFrame({'fft': PW_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    # plt.plot(data_FT['PW'], label='Real')
    plt.xlabel('TIC')
    plt.ylabel('MW')
    plt.title('WIND POWER Fourier transforms')
    plt.legend()
    plt.show()


warnings.filterwarnings(action='ignore')
plot_Fourier(dataset)
warnings.filterwarnings(action='default')


