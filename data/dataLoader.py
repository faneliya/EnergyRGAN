import data.dbmanager as db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import wavedec
import os
from scipy import fftpack
from datetime import datetime


def dataLoaderInit(opt):
    print('############### dataLoader Class ########################')
    print(opt)


def dataLoadSQL(sqlText, fileName):
    """
    DATE 2021-11-01
    DESC
    """
    if sqlText is not None and fileName is not None:
        conn = db.connectMariaDB()
        # date format coding check!
        dataOut = pd.read_sql(sqlText, conn)
        dataOut.to_csv(fileName)

        db.closeMariaDB(conn)
    return


def dataLoad_WindPower():
    """
    DATE
    DESC
    """
    conn = db.connectMariaDB()
    cur = conn.cursor()
    sql = "SELECT" \
          " REG_ID," \
          " REG_YMD," \
          " REG_HH24," \
          " REG_MM," \
          " PW_P," \
          " CAPACITY_MW " \
          "FROM MELCHIOR.AST0102 WHERE REG_ID='EU_BG_EA_W'"
    cur.execute(sql)

    dfSample = pd.DataFrame(index=range(0, 3), columns=['power', 'capacity'])

    while True:
        row = cur.fetchone()
        if row == None:
            break;
        daoRegion = row[0]
        daoYmd = row[1]
        daoHH = row[2]
        daoMi = row[3]
        daoPower = row[4]
        daoMax = row[5]
        print("%s %4s %2s %2s %5.2f %5.2f" % (daoRegion, daoYmd, daoHH, daoMi, daoPower, daoMax))

    sql2 = "SELECT" \
           " CONCAT(REG_YMD, REG_HH24, REG_MM) AS DT," \
           " PW_P," \
           " CAPACITY_MW " \
           "FROM MELCHIOR.AST0102 WHERE REG_ID='EU_BG_EA_W' " \
           "AND REG_YMD BETWEEN '20210301' AND '20210302' "

    # date format coding check!
    daoOut = pd.read_sql(sql2, conn, parse_dates={'DT'}, index_col='DT')
    daoOut.info()

    daoOut.PW_P.resample('D').sum().plot(title='EURO Beligium 2021 Wind Power')
    plt.tight_layout()
    plt.show()

    db.closeMariaDB(conn)


def load_WindPower(start_day, end_day):
    """
    DATE
    DESC
    """
    conn = db.connectMariaDB()

    sql2 = "SELECT" \
           " CONCAT(REG_YMD, REG_HH24, REG_MM) AS DT," \
           " PW_P," \
           " CAPACITY_MW " \
           "FROM MELCHIOR.AST0102 WHERE REG_ID='EU_BG_EA_W' " \
           "AND REG_YMD BETWEEN '" + start_day + "' AND '" + end_day + "'"

    # date format coding check!
    windData = pd.read_sql(sql2, conn, parse_dates={'DT'}, index_col='DT')
    db.closeMariaDB(conn)
    return windData


def dataLoad_SolarPowerChart():
    conn = db.connectMariaDB()

    sqlWind = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
          "PW_P AS power " \
          "FROM MELCHIOR.AST0203 " \
          "AND REG_YMD BETWEEN '20200301' AND '20200305' "

    sqlSolar = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
          "PW_P AS power " \
          "FROM MELCHIOR.AST0203 WHERE REG_ID = 'US_AL_NRE_S' " \
          "AND REG_YMD BETWEEN '20060301' AND '20060305'"

    # date format coding check!
    daoOut = pd.read_sql(sqlSolar, conn, parse_dates={'DATE_STAMP'}, index_col='DATE_STAMP')

    daoOut.plot()

    plt.grid(True)
    plt.show()

    # fft analysis
    time_step = 0.1
    sig_np = daoOut.to_numpy()
    sig_fft = fftpack.fft(sig_np)

    power = np.abs(sig_fft)

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(sig_np.size, d=time_step)

    # Plot the FFT power
    plt.figure(figsize=(18, 8))
    plt.plot(sample_freq, power)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('plower')
    plt.show()

    # daoOut.plot(x='DT', y='PW_P')

    # daoOut.PW_P.resample('D').sum().plot(title='USA Alibama')
    # plt.tight_layout()
    # plt.show()
    plot_Fourier(daoOut)

    db.closeMariaDB(conn)


def dataLoad_WindPowerChart():
    conn = db.connectMariaDB()

    sqlWind = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
          "PW_P AS power " \
          "FROM MELCHIOR.AST0102 WHERE REG_ID='EU_BG_EA_W' " \
          "AND REG_YMD BETWEEN '20200301' AND '20210425' "

    sql = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
          "PW_P AS power " \
          "FROM MELCHIOR.AST0203 WHERE REG_ID='US_AL_NRE_S' " \
          "AND REG_YMD BETWEEN '20060101' AND '20060107' "

    # date format coding check!
    daoOut = pd.read_sql(sqlWind, conn, parse_dates={'DATE_STAMP'}, index_col='DATE_STAMP')

    daoOut.plot()

    plt.grid(True)
    plt.show()

    # fft analysis
    time_step = 0.1
    sig_np = daoOut.to_numpy()
    sig_fft = fftpack.fft(sig_np)

    power = np.abs(sig_fft)

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(sig_np.size, d=time_step)

    # Plot the FFT power
    plt.figure(figsize=(18, 8))
    plt.plot(sample_freq, power)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('plower')
    plt.show()

    # daoOut.plot(x='DT', y='PW_P')

    # daoOut.PW_P.resample('D').sum().plot(title='USA Alibama')
    # plt.tight_layout()
    # plt.show()
    plot_Fourier(daoOut)

    db.closeMariaDB(conn)


def plot_Fourier(dataset):

    dataset['row'] = dataset.reset_index().index # 인덱스를 읽어서 새로운 row 생성
    dataset.set_index('row', inplace=True) # row를 새로운 인덱스로 설정
    data_FT = dataset[['power']]

    pw_fft = np.fft.fft(np.asarray(data_FT['power'].tolist()))
    fft_df = pd.DataFrame({'fft': pw_fft})
    # fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    # fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())

    plt.figure(figsize=(40, 10))

    for num_ in [3, 6, 12]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        result = np.fft.ifft(fft_list_m10) # i-FFT
        plt.plot(np.abs(result), label='FFT {} Days'.format(int(96/num_)))

    plt.xlabel('15min')
    plt.ylabel('MW')
    plt.title('Power Output FFT Analysis')
    plt.legend()
    plt.show()

    plt.figure(figsize=(40, 10))
    for num_ in [24, 48, 96]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        result = np.fft.ifft(fft_list_m10)
        plt.plot(np.abs(result), label='FFT {} Days'.format(int(96/num_)))

    plt.xlabel('15min')
    plt.ylabel('MW')
    plt.title('Power Output FFT Analysis')
    plt.legend()
    plt.show()

    plt.figure(figsize=(40, 10))
    for num_ in [192, 288, 384]:
        fft_list_m10 = np.copy(fft_list);
        fft_list_m10[num_:-num_] = 0
        result = np.fft.ifft(fft_list_m10)
        # print(rslt)
        plt.plot(np.abs(result), label='FFT {} components'.format(96/num_))

    plt.xlabel('15min')
    plt.ylabel('MW')
    plt.title('Power Output FFT Analysis')
    plt.legend()
    plt.show()


def convert_Fourier(dataset, mode):

    # 숫자열로 인덱스 처리 0,1,2,.....
    dataset['row'] = dataset.reset_index().index # 인덱스를 읽어서 새로운 row 생성
    dataset.set_index('row', inplace=True) # row를 새로운 인덱스로 설정

    # 저장
    # 턴dataset.to_csv('origin.csv')

    # FFT 처리
    data_power = dataset[['power']]
    data_FT = dataset[['power']]
    # pw_fft = np.fft.fft(np.asarray(data_FT['power'].tolist()))
    pw_fft = np.fft.fft(np.asarray(data_power['power'].tolist()))
    fft_df = pd.DataFrame({'fft': pw_fft})
    ifft_df = pd.DataFrame({'fft': np.abs(pw_fft)})
    # ifft_df['date'] = dateStamp
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())

    # plt.figure(figsize=(40, 10))
    i = 1
    for num_ in [192, 288, 384]:
        fft_list_filtered = np.copy(fft_list)
        fft_list_filtered[num_:-num_] = 0
        result = np.fft.ifft(fft_list_filtered) # i-FFT

        ifft_df[str(i)+'st'] = np.abs(result)
        i += 1
        # plt.plot(np.abs(result), label='FFT {} Days'.format(int(96/num_)))

    ifft_df['row'] = dataset.reset_index().index # 인덱스를 읽어서 새로운 row 생성
    ifft_df.set_index('row', inplace=True) # row를 새로운 인덱스로 설정
    timeStr = str(datetime.today().strftime("%Y%m%d%H%M%S"))

    # 파일명 정리해서 저장처리 금일날짜별
    if mode == 'Solar':
        modeName = 'Solar'
    elif mode == 'Wind':
        modeName = 'Wind'
    else:
        modeName = 'Gen'
    saveFileName = 'result-i' + modeName + timeStr + '.csv'
    ifft_df.to_csv(saveFileName, sep=',', na_rep='NaN', index = True)
    # ifft_df.to_excel('result-i.xlsx', na_rep='NaN', index = True)

    # 그래프 표 표시 후 리
    plot_Fourier(dataset)
    return saveFileName


def convert_Wavelet(dataset, mode):
    dataset['row'] = dataset.reset_index().index # 인덱스를 읽어서 새로운 row 생성
    dataset.set_index('row', inplace=True) # row를 새로운 인덱스로 설정

    dataset.to_csv('origin.csv')
    # FFT 처리
    data_power = dataset[['power']]
    npData_power = np.asarray(data_power['power'].tolist())

    # Single Level dwt
    coeffs = pywt.dwt(npData_power, 'db1', 'smooth')
    cA, cD = coeffs
    A = pywt.idwt(cA, None, 'db2', 'smooth')
    D = pywt.idwt(None, cD, 'db2', 'smooth')

    # MultiLevel 2
    coeffsMulti = wavedec(npData_power, 'db1', level=2)
    cA2, cD2, cD1 = coeffsMulti
    coeffsMulti1 = (cA2, None, None)
    coeffsMulti2 = (None, cD2, cD1)
    # coeffsMulti3 = (None, None, cD1)
    m1 = pywt.waverec(coeffsMulti1, 'db1')
    m2 = pywt.waverec(coeffsMulti2, 'db1')
    # m3 = pywt.waverec(coeffsMulti3, 'db1')

    # MultiLevel 3
    coeffsMultiLevel3 = wavedec(npData_power, 'db1', level=3)
    cA4, cD32, cD22, cD12 = coeffsMultiLevel3
    plt.figure(figsize=(40, 10))
    plt.ylim(-20, 150)
    plt.plot(cA4, label='Wavelet', color='blue')
    plt.show()
    plt.figure(figsize=(40, 10))
    plt.ylim(-20, 150)
    plt.plot(cD32, label='Wavelet', color='g')
    plt.show()
    plt.figure(figsize=(40, 10))
    plt.ylim(-20, 150)
    plt.plot(cD22, label='Wavelet', color='r')
    plt.show()
    plt.figure(figsize=(40, 10))
    plt.ylim(-20, 150)
    plt.plot(cD12, label='Wavelet', color='c')
    plt.show()

    #plt.figure(figsize=(40, 10))
    #plt.plot(cA2, label='Wavelet A Days1'
    #plt.show()

    #plt.figure(figsize=(40, 10))
    #plt.plot(cD1, label='Wavelet D Days2')
    #plt.show()

    #plt.figure(figsize=(40, 10))
    #plt.plot(cD2, label='Wavelet D Days2')
    #plt.show()

    plt.figure(figsize=(40, 10))
    plt.plot(m1, label='Wavelet A Days1')
    plt.plot(m2, label='Wavelet D Days2')
    # plt.plot(m3, label='Wavelet D Days3')
    plt.show()

    plt.figure(figsize=(40, 10))
    plt.plot(A, label='Wavelet A Days')
    plt.plot(D, label='Wavelet D Days')
    plt.show()
    return 'test'


def windDataFFTSave(startYmd, endYmd):
    print(__name__)
    conn = db.connectMariaDB()
    startYmdStr = str(startYmd)
    endYmdStr = str(endYmd)
    sqlWind = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
              " PW_P AS power " \
              "FROM MELCHIOR.AST0102 WHERE REG_ID='EU_BG_EA_W' " \
              "AND REG_YMD BETWEEN '" + startYmdStr + "' AND '" + endYmdStr + "'"
    # date format coding check!
    dfResult = pd.read_sql(sqlWind, conn, parse_dates={'DATE_STAMP'}, index_col='DATE_STAMP')
    # fft analysis
    fileName = convert_Fourier(dfResult, 'Wind')
    db.closeMariaDB(conn)
    return fileName


def solarDataFFTSave(startYmd, endYmd):
    conn = db.connectMariaDB()
    startYmdStr = str(startYmd)
    endYmdStr = str(endYmd)

    sqlSolar = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
               "PW_P AS power " \
               "FROM MELCHIOR.AST0203 WHERE REG_ID = 'US_AL_NRE_S' " \
               "AND REG_YMD BETWEEN '" + startYmdStr + "' AND '" + endYmdStr + "'"

    # date format coding check!
    dfResult = pd.read_sql(sqlSolar, conn, parse_dates={'DATE_STAMP'}, index_col='DATE_STAMP')
    # fft analysis
    fileName = convert_Fourier(dfResult, 'Solar')
    db.closeMariaDB(conn)
    return fileName


def windDataDWTSave(startYmd, endYmd):
    conn = db.connectMariaDB()
    startYmdStr = str(startYmd)
    endYmdStr = str(endYmd)
    sqlWind = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
              " PW_P AS power " \
              "FROM MELCHIOR.AST0102 WHERE REG_ID='EU_BG_EA_W' " \
              "AND REG_YMD BETWEEN '" + startYmdStr + "' AND '" + endYmdStr + "'"
    # date format coding check!
    dfResult = pd.read_sql(sqlWind, conn, parse_dates={'DATE_STAMP'}, index_col='DATE_STAMP')
    # fft analysis
    fileName = convert_Wavelet(dfResult, 'Wind')
    db.closeMariaDB(conn)
    return fileName


def solarDataDWTSave(startYmd, endYmd):
    conn = db.connectMariaDB()
    startYmdStr = str(startYmd)
    endYmdStr = str(endYmd)
    sqlSolar = "SELECT CONCAT(REG_YMD,REG_HH24,REG_MM) AS DATE_STAMP," \
               "PW_P AS power " \
               "FROM MELCHIOR.AST0203 WHERE REG_ID = 'US_AL_NRE_S' " \
               "AND REG_YMD BETWEEN '" + startYmdStr + "' AND '" + endYmdStr + "'"
    # date format coding check!
    dfResult = pd.read_sql(sqlSolar, conn, parse_dates={'DATE_STAMP'}, index_col='DATE_STAMP')
    # fft analysis
    fileName = convert_Wavelet(dfResult, 'Solar')
    db.closeMariaDB(conn)
    return fileName


