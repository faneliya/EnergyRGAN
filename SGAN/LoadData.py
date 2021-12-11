import warnings
import pywt
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math

import data.dataLoader as dl
import SGAN.SQL.sqlList as sqlList

#el.read_weather_data_from_excel(base_dir, None)

base_dir="./DataFiles/"


def dataFileExists(target):
    result = False
    file_list = os.listdir(base_dir)

    for filename in file_list:
        if filename.endswith('.csv'):
            if filename == target:
                result = True
    return result


def windDataFileExists():
    result = False
    file_list = os.listdir(base_dir)

    for filename in file_list:
        if filename.endswith('.csv'):
            if filename == "WindData.csv":
                result = True

    return result


def solarDataFileExists():
    result = False
    file_list = os.listdir(base_dir)

    for filename in file_list:
        if filename.endswith('.csv'):
            if filename == "SolarData.csv":
                result = True
    return result

# init funciton, DownaLoad DB
def init():

    if windDataFileExists() is False:
        print(">Download Wind data From DB...")
        dl.dataLoadSQL(sqlList.sqlTextWindBelgium, base_dir + "WindData.csv", indexSet=False)

    if dataFileExists("WindAllData.csv") is False:
        print(">Download Wind data From Wind All DB...")
        dl.dataLoadSQL(sqlList.sqlTextWindBelgiumAll, base_dir + "WindAllData.csv", indexSet=False)
    ###

    if dataFileExists("WindAllDataM3.csv") is False:
        print(">Download Wind data From Wind All DB For 3-Month...")
        dl.dataLoadSQL(sqlList.sqlTextWindBelgiumAllM3, base_dir + "WindAllDataM3.csv", indexSet=False)

    if dataFileExists("WindAllDataM6.csv") is False:
        print(">Download Wind data From Wind All DB For 6-Month...")
        dl.dataLoadSQL(sqlList.sqlTextWindBelgiumAllM6, base_dir + "WindAllDataM6.csv", indexSet=False)

    if dataFileExists("WindAllDataY1.csv") is False:
        print(">Download Wind data From Wind All DB For 1-Year...")
        dl.dataLoadSQL(sqlList.sqlTextWindBelgiumAllY1, base_dir + "WindAllDataY1.csv", indexSet=False)

    if dataFileExists("WindAllDataY2.csv") is False:
        print(">Download Wind data From Wind All DB For 2 Year...")
        dl.dataLoadSQL(sqlList.sqlTextWindBelgiumAllY2, base_dir + "WindAllDataY2.csv", indexSet=False)


    ###
    if solarDataFileExists() is False:
        print(">Download Solar data From DB...")
        dl.dataLoadSQL(sqlList.sqlTextSolarBelgium, base_dir + "SolarData.csv", indexSet=False)

    if dataFileExists("SolarAllData.csv") is False:
        print(">Download Solar data From Solar All DB...")
        dl.dataLoadSQL(sqlList.sqlTextSolarBelgiumAll, base_dir + "SolarAllData.csv", indexSet=False)

    ###
    if dataFileExists("SolarAllDataM3.csv") is False:
        print(">Download Solar data From Solar All DB For 3-Month...")
        dl.dataLoadSQL(sqlList.sqlTextSolarBelgiumAllM3, base_dir + "SolarAllDataM3.csv", indexSet=False)

    if dataFileExists("SolarAllDataM6.csv") is False:
        print(">Download Solar data From Solar All DB For 6-Month...")
        dl.dataLoadSQL(sqlList.sqlTextSolarBelgiumAllM6, base_dir + "SolarAllDataM6.csv", indexSet=False)

    if dataFileExists("SolarAllDataY1.csv") is False:
        print(">Download Solar data From Solar All DB For 1-Year...")
        dl.dataLoadSQL(sqlList.sqlTextSolarBelgiumAllY1, base_dir + "SolarAllDataY1.csv", indexSet=False)

    if dataFileExists("SolarAllDataY2.csv") is False:
        print(">Download Solar data From Solar All DB For 2-Year...")
        dl.dataLoadSQL(sqlList.sqlTextSolarBelgiumAllY2, base_dir + "SolarAllDataY2.csv", indexSet=False)

    ####

    ####
    if dataFileExists("BelgiumAllData.csv") is False:
        print(">Download Wind+Solar data From Belgium Energy All DB...")
        dl.dataLoadSQL(sqlList.sqlTextBelgiumEnergyAll, base_dir + "BelgiumAllData.csv", indexSet=False)
    #####


def fftSave(fileName):
    if dataFileExists(fileName+".csv"):
        dfData = pd.read_csv(base_dir + fileName+".csv")
        fftDataProcess(dfData, base_dir + fileName+"FFT.csv")

def main():
    ## import data
    #df = pd.read_csv('windData.csv')
    figsizeSet = (15, 5)
    PlotShow = False

    if windDataFileExists():
        dfWind = pd.read_csv(base_dir + 'WindData.csv')

        #print(dfWind.head())
        #print(dfWind.tail())
        #print(dfWind.shape)
        #print(dfWind.columns)

        if PlotShow is True:
            fig, ax = plt.subplots(figsize=figsizeSet)
            ax.plot(dfWind.index, dfWind['PW'], label='Wind Power')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Wind Power Belgium(2016-2021)")
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(figsize=figsizeSet)
            ax.plot(dfWind.index, dfWind['PW'], label='Wind Power')
            ax.plot(dfWind.index, dfWind['CAPA'], label='Power Installation')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Wind Power Belgium(2016-2021) && Installation")
            plt.tight_layout()
            plt.show()


            fig, ax = plt.subplots(figsize=figsizeSet)
            sampleData = dfWind[2500:4000]
            ax.plot(sampleData.index, sampleData['PW'], label='Solar+Wind Power')
            # ax.plot(dfData.index, dfData['CAPA'], label='Installed Solar+Wind Power')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Wind  Power Belgium(2week)")
            plt.tight_layout()
            plt.show()

        fftDataProcess(dfWind, base_dir+"WindDataFFT.csv")


    if solarDataFileExists():
        dfSolar = pd.read_csv(base_dir + 'SolarData.csv')

        if PlotShow is True:
            fig, ax = plt.subplots(figsize=figsizeSet)
            ax.plot(dfSolar.index, dfSolar['PW'], label='Solar Power')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Solar Power Belgium(2016-2021)")
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(figsize=figsizeSet)
            ax.plot(dfSolar.index, dfSolar['PW'], label='Solar Power')
            ax.plot(dfSolar.index, dfSolar['CAPA'], label='Power Installation')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Solar Power Belgium(2016-2021) && Installation")
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(figsize=figsizeSet)
            sampleData = dfSolar[2500:4000]
            ax.plot(sampleData.index, sampleData['PW'], label='Solar+Wind Power')
            # ax.plot(dfData.index, dfData['CAPA'], label='Installed Solar+Wind Power')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Solar Power Belgium(2week)")
            plt.tight_layout()
            plt.show()

        fftDataProcess(dfSolar, base_dir+"SolarDataFFT.csv")



    if dataFileExists("BelgiumAllData.csv"):
        dfData = pd.read_csv(base_dir + 'BelgiumAllData.csv')

        if PlotShow is True & windDataFileExists() & solarDataFileExists():
            fig, ax = plt.subplots(figsize=figsizeSet)
            ax.plot(dfData.index, dfData['PW'], label='Solar+Wind Power')
            ax.plot(dfSolar.index, dfSolar['PW'], label='Solar Power')
            ax.plot(dfWind.index, dfWind['PW'], label='Wind Power')
            ax.plot(dfData.index, dfData['CAPA'], label='Installed Solar+Wind Power')
            ax.plot(dfSolar.index, dfSolar['CAPA'], label='Installed Solar Power')
            ax.plot(dfWind.index, dfWind['CAPA'], label='Installed Wind Power')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Solar+Wind  Power Belgium(2016-2021)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        if PlotShow is True:
            fig, ax = plt.subplots(figsize=figsizeSet)
            ax.plot(dfData.index, dfData['TEMP_MAX'], label='Temperatures Max')
            ax.plot(dfData.index, dfData['TEMP_MIN'], label='Temperatures Min')

            # ax.plot(dfSolar.index, dfSolar['TEMP_MAX'], label='Temp Max')
            ax.set(xlabel="Days(15min Points)",
                   ylabel="degree",
                   title="Belgium Temperature(2016-2021)")
            # date_form = DateFormatter("%Y%m%d%H%M")
            # ax.xaxis.set_major_formatter(date_form)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if PlotShow is True:
            fig, ax = plt.subplots(figsize=figsizeSet)
            ax.plot(dfData.index, dfData['RADIATION'], label='RADIATION MAX PER DAY')
            # ax.plot(dfSolar.index, dfSolar['TEMP_MAX'], label='Temp Max')
            ax.set(xlabel="Days(15min Points)",
                   ylabel="W/M",
                   title="Belgium RADIATION(2016-2021)")
            # date_form = DateFormatter("%Y%m%d%H%M")
            # ax.xaxis.set_major_formatter(date_form)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if PlotShow is True & windDataFileExists() & solarDataFileExists():
            fig, ax = plt.subplots(figsize=figsizeSet)
            sampleData = dfData[2500:4000]
            # print(sampleData.DATE)
            ax.plot(sampleData.index, sampleData['PW'], label='Solar+Wind Power')
            sampleData = dfWind[2500:4000]
            ax.plot(sampleData.index, sampleData['PW'], label='Wind Power')
            sampleData = dfSolar[2500:4000]
            ax.plot(sampleData.index, sampleData['PW'], label='Solar Power')
            # ax.plot(dfData.index, dfData['CAPA'], label='Installed Solar+Wind Power')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Solar+Wind  Power Belgium(2week)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        if PlotShow is True & windDataFileExists() & solarDataFileExists():
            fig, ax = plt.subplots(figsize=figsizeSet)
            sampleData = dfData[5000:6500]
            # print(sampleData.DATE)
            ax.plot(sampleData.index, sampleData['PW'], label='Solar+Wind Power')
            sampleData = dfWind[5000:6500]
            ax.plot(sampleData.index, sampleData['PW'], label='Wind Power')
            sampleData = dfSolar[5000:6500]
            ax.plot(sampleData.index, sampleData['PW'], label='Solar Power')
            # ax.plot(dfData.index, dfData['CAPA'], label='Installed Solar+Wind Power')
            ax.set(xlabel="15min",
                   ylabel="MW",
                   title="Solar+Wind  Power Belgium(2week)")
            plt.tight_layout()
            plt.legend()
            plt.show()

        # db1 = pywt.Wavelet('db1')
        # (cA2, cD2), (cA1, cD1) = pywt.swt(dfData['PW'] / 6000 , db1, level=2)
        # ax.plot(dfData.index, cA2, label='Wavelet cA2')
        # ax.plot(dfData.index, cD2, label='Wavelet cD2')
        # ax.plot(dfData.index, cA1, label='Wavelet cA1')
        # ax.plot(dfData.index, cD1, label='Wavelet cD1')
        # plt.show()
        fftDataProcess(dfData, base_dir+"BelgiumAllDataFFT.csv")

    fftSave("SolarAllData")
    fftSave("SolarAllDataM3")
    fftSave("SolarAllDataM6")
    fftSave("SolarAllDataY1")
    fftSave("SolarAllDataY2")

    fftSave("WindAllData")
    fftSave("WindAllDataM3")
    fftSave("WindAllDataM6")
    fftSave("WindAllDataY1")
    fftSave("WindAllDataY2")



def fftDataProcess(df, filename):

    warnings.filterwarnings(action='ignore') # waring ignore
    df_technicalIndicated = get_technical_indicators(df)
    warnings.filterwarnings(action='default') # waring ignore

    dataset = df_technicalIndicated.iloc[20:, :].reset_index(drop=True)
    datasetFFT = get_fourier_transfer(dataset)

    dataJoined = pd.concat([dataset, datasetFFT], axis=1)
    # print(Final_data.head())
    dataJoined.to_csv(filename, index=False)

    # show indicator plot
    # plot_technical_indicators(df_technicalIndicated, 17000)
    # show ifft plot
    warnings.filterwarnings(action='ignore') # waring ignore
    # plot_Fourier(dataset, filename)
    warnings.filterwarnings(action='default') # waring ignore

    # print(len(dataset))
    A = list()
    D = list()
    if len(dataset) % 2 == 0:
        A, D = convert_Wavelet(dataset, filename)
    else:
        A, D = convert_Wavelet(dataset[1:], filename)
    listA = pd.DataFrame(A,columns=['DWT_A'])
    listD = pd.DataFrame(D,columns=['DWT_D'])

    totl = len(dataJoined)
    tmp_dwt = pd.concat([listA[:totl], listD[:totl]], axis=1)
    dwt_df = pd.concat([dataJoined, tmp_dwt], axis=1)
    dwt_df['DATE'].astype(int)
    dwt_df.to_csv(filename[:-4]+"_DWT.csv", index=False)
    #dataJoined['DWT_A'] = A
    #dataJoined['DWT_D'] = D


def get_technical_indicators(data):
    # # Create 7 and 21 days Moving Average
    # data['MA7'] = data.iloc[:, 1].rolling(window=7).mean()
    # data['MA21'] = data.iloc[:, 1].rolling(window=21).mean()
    # # Create MACD
    # data['MACD'] = data.iloc[:, 1].ewm(span=26).mean() - data.iloc[:,1].ewm(span=12,adjust=False).mean()
    # # Create Bollinger Bands
    # data['20SD'] = data.iloc[:, 1].rolling(20).std()
    # data['upper_band'] = data['MA21'] + (data['20SD'] * 2)
    # data['lower_band'] = data['MA21'] - (data['20SD'] * 2)

    data['MA1H'] = data.iloc[:, 1].rolling(window=4).mean()
    data['MA1D'] = data.iloc[:, 1].rolling(window=96).mean()
    # Create MACD
    data['MACD'] = data.iloc[:,1].ewm(span=2496).mean() - data.iloc[:,1].ewm(span=1152,adjust=False).mean()
    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 1].rolling(1920).std()
    data['upper_band'] = data['MA1D'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA1D'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:,1].ewm(com=0.5).mean()
    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:,1] - 1)

    return data


def get_fourier_transfer(dataset):
    # Get the columns for doing fourier
    #data_FT = dataset[['IDX', 'PW']]
    dataset["IDX"] = dataset.index
    dataIndexed = dataset[["IDX", 'PW']]
    pw_fft = np.fft.fft(np.asarray(dataIndexed['PW'].tolist()))
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
        fft_com['ABS_' + str(num_) + 'COMP'] = fft_com['fft'].apply(lambda x: np.abs(x))
        fft_com['ANGLE_' + str(num_) + 'COMP'] = fft_com['fft'].apply(lambda x: np.angle(x))
        fft_com = fft_com.drop(columns='fft')
        fft_com_df = pd.concat([fft_com_df, fft_com], axis=1)
    return fft_com_df


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    dataset = dataset.iloc[-last_days:, :]
    x_ = list(dataset.index)

    plt.plot(dataset['PW'], label='Power', color='b')

    # # Plot first subplot
    # plt.plot(dataset['MA7'], label='MA 7', color='g', linestyle='--')
    # plt.plot(dataset['MA21'], label='MA 21', color='r', linestyle='--')

    # Plot first subplot
    plt.plot(dataset['MA1H'], label='MA 1Hour', color='g', linestyle='--')
    plt.plot(dataset['MA1D'], label='MA 1Day', color='r', linestyle='--')

    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Power Band'.format(last_days))
    plt.ylabel('Power MW')
    plt.legend()
    plt.legend()
    plt.show()


def plot_Fourier(dataset, plotname):
    data_FT = dataset[['IDX', 'PW']]
    #print(data_FT.info())
    PW_fft = np.fft.fft(np.asarray(data_FT['PW'].tolist()))
    fft_df = pd.DataFrame({'fft': PW_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())
    figsizeSet = (15, 5)
    plt.figure(figsize=figsizeSet, dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    # plt.plot(data_FT['PW'], label='Real')
    plt.xlabel('Time Line(15min)')
    plt.ylabel('MW')
    if plotname is None:
        plotname = "Belgium Power Data: "
    else:
        plotname = "Belgium Power Data: " + plotname
    plt.title(plotname + 'Time Series Data(FFT> IFFT Components)')
    plt.tight_layout()
    plt.legend()
    plt.show()


def convert_Wavelet(dataset, filename):
    # dataset['row'] = dataset.reset_index().index # 인덱스를 읽어서 새로운 row 생성
    # dataset.set_index('row', inplace=True) # row를 새로운 인덱스로 설정
    print("Convering Wavelet Data....")
    PlotShow = False
    SWT_LEVEL2 = False

    if filename is not None:
        infoFile = filename
    else:
        infoFile = ''
    figsizeSet = (15, 5)

    # dataset.to_csv('origin.csv')
    # FFT 처리
    data_power = dataset[['IDX', 'PW']]
    npData_power = np.asarray(data_power['PW'].tolist())

    # Single Level dwt
    coeffs = pywt.dwt(npData_power, 'db1', 'smooth')
    cA, cD = coeffs
    A = pywt.idwt(cA, None, 'db1', 'smooth')
    D = pywt.idwt(None, cD, 'db1', 'smooth')

    if PlotShow is True:
        plt.figure(figsize=figsizeSet)
        plt.plot(A, label='Wavelet A')
        plt.plot(D, label='Wavelet D')
        plt.title(infoFile + "DWT db1/smooth")
        plt.tight_layout()
        plt.legend()
        plt.show()

    # MultiLevel 2
    coeffsMultiLevel2 = pywt.wavedec(npData_power, 'db1', level=2)
    cA2, cD2, cD1 = coeffsMultiLevel2

    if PlotShow is True:
        plt.figure(figsize=figsizeSet)
        plt.plot(cA2, label='Wavelet cA2')
        plt.plot(cD2, label='Wavelet cD2')
        plt.plot(cD1, label='Wavelet cD1')
        plt.title(infoFile + " DWT MultiLevel 2")
        plt.tight_layout()
        plt.legend()
        plt.show()

    # MultiLevel 3
    coeffsMultiLevel3 = pywt.wavedec(npData_power, 'db1', level=3)
    cA4, cD32, cD22, cD12 = coeffsMultiLevel3

    if PlotShow is True:
        plt.figure(figsize=figsizeSet)
        plt.plot(cA4, label='Wavelet cA4', color='blue')
        plt.plot(cD32, label='Wavelet cD32', color='g')
        plt.plot(cD22, label='Wavelet cD22', color='r')
        plt.plot(cD12, label='Wavelet cD12', color='c')
        plt.title(infoFile + "DWT MultiLevel 3")
        plt.tight_layout()
        plt.legend()
        plt.show()

    if PlotShow is True:
        plt.figure(figsize=figsizeSet)
        plt.plot(cA4[2500:4000], label='Wavelet cA4 Sample', color='blue')
        plt.plot(cD32[2500:4000], label='Wavelet cD32 Sample', color='g')
        plt.plot(cD22[2500:4000], label='Wavelet cD22 Sample', color='r')
        plt.plot(cD12[2500:4000], label='Wavelet cD12 Sample', color='c')
        plt.title(infoFile + "DWT MultiLevel MultiLevel 3 Sample")
        plt.tight_layout()
        plt.legend()
        plt.show()

    if PlotShow is True:
        plt.figure(figsize=figsizeSet)
        plt.plot(cA4[5000:6500], label='Wavelet cA4 Sample', color='blue')
        plt.plot(cD32[5000:6500], label='Wavelet cD32 Sample', color='g')
        plt.plot(cD22[5000:6500], label='Wavelet cD22 Sample', color='r')
        plt.plot(cD12[5000:6500], label='Wavelet cD12 Sample', color='c')
        plt.title(infoFile + "DWT MultiLevel MultiLevel 3 Sample")
        plt.tight_layout()
        plt.legend()
        plt.show()

    # SWT MultiLevel 2
    if SWT_LEVEL2 is True:
        db1 = pywt.Wavelet('db1')
        print("hellow")
        print(len(npData_power))
        (cA2, cD2), (cA1, cD1) = pywt.swt(npData_power, db1, level=2)

        if PlotShow is True:
            plt.figure(figsize=figsizeSet)
            plt.plot(cA2, label='Wavelet cA2', color='blue')
            plt.plot(cD2, label='Wavelet cD2', color='g')
            plt.plot(cA1, label='Wavelet cA1', color='r')
            plt.plot(cD1, label='Wavelet cD1', color='c')
            plt.title(infoFile + "SWT(Stationary wavelet transform) MultiLevel 2")
            plt.tight_layout()
            plt.legend()
            plt.show()

            plt.figure(figsize=figsizeSet)
            plt.plot(cA2[2500:4000], label='Wavelet cA4 Sample', color='blue')
            plt.plot(cD2[2500:4000], label='Wavelet cD32 Sample', color='g')
            plt.plot(cA1[2500:4000], label='Wavelet cD22 Sample', color='r')
            plt.plot(cD1[2500:4000], label='Wavelet cD12 Sample', color='c')
            plt.title(infoFile + "SWT(Stationary wavelet transform) MultiLevel 2 Sample")
            plt.tight_layout()
            plt.legend()
            plt.show()


        plt.figure(figsize=figsizeSet)
        plt.plot(cA2[5000:6500], label='Wavelet cA4 Sample', color='blue')
        plt.plot(cD2[5000:6500], label='Wavelet cD32 Sample', color='g')
        plt.plot(cA1[5000:6500], label='Wavelet cD22 Sample', color='r')
        plt.plot(cD1[5000:6500], label='Wavelet cD12 Sample', color='c')
        plt.title(infoFile + " SWT(Stationary wavelet transform) MultiLevel 2 Sample")
        plt.tight_layout()
        plt.legend()
        plt.show()



    return A, D



# 프로그램 시작처리
if __name__ == '__main__':
    print('################ LoadData To CSV #########################')
    init()
    main()
