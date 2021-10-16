import data.dataLoader as dataLoader
import DataPreProcessor
import GRU_Model
import GRUPredict

import os
import platform

def show_Power_chart():
    # Use a breakpoint in the code line below to debug your script.
    print('############### wind power graph ###############')
    # Press Ctrl+F8 to toggle the breakpoint.

    dataLoader.dataLoaderInit('Testing Code')
    # dataLoader.dataLoad_WindPowerChart()
    dataLoader.dataLoad_SolarPowerChart()
    # windData = dataLoader.load_WindPower('20210501', '20210501')
    # return windData

# From DB to Dataframe 4 Main API
def decodeWindFFT():
    dataLoader.dataLoaderInit('>main.py/decodeWindFFT/Processing...')
    # dataLoader.dataLoad_WindPowerChart()
    fileName =dataLoader.windDataFFTSave('20200301', '20210301')
    return fileName


def decodeSolarFFT():
    dataLoader.dataLoaderInit('>main.py/decodeSolarFFT/Processing...')
    # dataLoader.dataLoad_WindPowerChart()
    fileName = dataLoader.solarDataFFTSave('20060301', '20070425')
    return fileName


def decodeWindDWT():
    dataLoader.dataLoaderInit('>main.py/decodeWindDWT/Processing...')
    fileName =dataLoader.windDataDWTSave('20200301', '20210301')
    return fileName


def decodeSolarDWT():
    dataLoader.dataLoaderInit('>main.py/decodeSolarDWT/Processing...')
    fileName =dataLoader.solarDataDWTSave('20060301', '20070425')
    return fileName


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print('This is running on ' + os.name + 'at ' + platform.system())
    # Setting For Test
    initOpt = 1
    mode = 'Solar'
    savedFileExist = True

    if initOpt is not None:
        mode = 'Wind'
        FFT = True
        Wavelet = False
        savedFileExist = False
    else:
        FFT = False
        Wavelet = True

    # decodeWindFFT()
    if FFT:
        if savedFileExist:
            fileName = 'result-iSolar20210815161603.csv'
        else:
            if mode == 'Solar':
                fileName = decodeSolarFFT()
            elif mode == 'Wind':
                fileName = decodeWindFFT()
            else:
                exit()

    if Wavelet:
        if mode == 'Solar':
            fileName = decodeSolarDWT()
        elif mode == 'Wind':
            fileName = decodeWindDWT()
        else:
            exit()
    exit()

    x_train_scaled = []
    y_train_scaled = []
    x_test_scaled = []
    y_test_scaled = []
    num_x_signals = 0
    num_y_signals = 0
    num_train = 0

    # Save Data to CSV , Load np fo np format data return
    x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled,\
    num_x_signals, num_y_signals, num_train, xScaler, yScaler = DataPreProcessor.load_data(fileName)

    if False:
        GRU_Model.build_model(mode, x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled,
                                     num_x_signals, num_y_signals, num_train)

    target_names = ['1st', '2st', '3st']
    start_idx = 0
    if mode == 'Solar':
        length_test = 50000
    elif mode == 'Wind':
        length_test = 30000
    else:
        exit()

    if False:
        GRUPredict.plot_comparison(mode, start_idx, target_names,
                                   x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, xScaler, yScaler,
                                   length=length_test, train=True)


