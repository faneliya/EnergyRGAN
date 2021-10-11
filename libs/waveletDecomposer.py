import numpy as np
import pandas as pd
import pywt
import keras
import os
import sys

from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
from matplotlib import pyplot as plt
from scipy import fftpack as fp

sys.path.append(os.path.dirname(os.path.abspath((os.path.dirname(__file__)))))

uci_har_signals_train = []
uci_har_labels_train = []
uci_har_signals_test = []
uci_har_labels_test = []
x_train = []
y_train = []
x_test = []
y_test = []


# Fast Fourier Transform
def get_fft_values(y_values: np.array, data_per_time: float, num_of_data: int):
    f_values = np.linspace(0.0, 1.0 / (2.0 * data_per_time), num_of_data // 2)
    fft_values_ = fp.fft(y_values)
    fft_values = 2.0 / num_of_data * np.abs(fft_values_[0:num_of_data // 2])
    return f_values, fft_values


# get_ave_values
def get_ave_values(y_values: np.array, data_per_time: float, num_of_data: int):
    f_values = np.linspace(0.0, 1.0 / (2.0 * data_per_time), num_of_data // 2)
    fft_values_ = fp.fft(y_values)
    fft_values = 2.0 / num_of_data * np.abs(fft_values_[0:num_of_data // 2])
    return f_values, fft_values


def show_fft_cases() -> None:
    timeVal = 1.0
    num_of_data = 100000
    data_per_time = timeVal / num_of_data
    f_s = 1 / data_per_time

    xa = np.linspace(0, timeVal, num=num_of_data)
    xb = np.linspace(0, timeVal / 4, num=int(num_of_data / 4))

    frequencies = [4, 30, 60, 90]
    y1a, y1b = np.sin(2 * np.pi * frequencies[0] * xa), np.sin(2 * np.pi * frequencies[0] * xb)
    y2a, y2b = np.sin(2 * np.pi * frequencies[1] * xa), np.sin(2 * np.pi * frequencies[1] * xb)
    y3a, y3b = np.sin(2 * np.pi * frequencies[2] * xa), np.sin(2 * np.pi * frequencies[2] * xb)
    y4a, y4b = np.sin(2 * np.pi * frequencies[3] * xa), np.sin(2 * np.pi * frequencies[3] * xb)

    composite_signal1 = y1a + y2a + y3a + y4a
    composite_signal2 = np.concatenate([y1b, y2b, y3b, y4b])

    f_values1, fft_values1 = get_fft_values(composite_signal1, data_per_time, num_of_data)
    f_values2, fft_values2 = get_fft_values(composite_signal2, data_per_time, num_of_data)

    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axarr[0, 0].plot(xa, composite_signal1)
    axarr[1, 0].plot(xa, composite_signal2)
    axarr[0, 1].plot(f_values1, fft_values1)
    axarr[1, 1].plot(f_values2, fft_values2)

    plt.tight_layout()
    plt.show()
    return


def show_wavelet_case():
    print(pywt.families(short=False))

    discrete_wavelets = ['db5', 'sym5', 'coif5', 'bior2.4']
    continuous_wavelets = ['mexh', 'morl', 'cgau5', 'gaus5']

    list_list_wavelets = [discrete_wavelets, continuous_wavelets]
    list_funcs = [pywt.Wavelet, pywt.ContinuousWavelet]

    fig, axarr = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    for ii, list_wavelets in enumerate(list_list_wavelets):
        func = list_funcs[ii]
        row_no = ii
        for col_no, waveletname in enumerate(list_wavelets):
            wavelet = func(waveletname)
            family_name = wavelet.family_name
            biorthogonal = wavelet.biorthogonal
            orthogonal = wavelet.orthogonal
            symmetry = wavelet.symmetry
            if ii == 0:
                _ = wavelet.wavefun()
                wavelet_function = _[0]
                x_values = _[-1]
            else:
                wavelet_function, x_values = wavelet.wavefun()
            if col_no == 0 and ii == 0:
                axarr[row_no, col_no].set_ylabel("Discrete Wavelets", fontsize=16)
            if col_no == 0 and ii == 1:
                axarr[row_no, col_no].set_ylabel("Continuous Wavelets", fontsize=16)
            axarr[row_no, col_no].set_title("{}".format(family_name
                                                        + "b: " + str(biorthogonal)
                                                        + "o: " + str(orthogonal)
                                                        + "s: " + str(symmetry)), fontsize=16)
            axarr[row_no, col_no].plot(x_values, wavelet_function)
            axarr[row_no, col_no].set_yticks([])
            axarr[row_no, col_no].set_yticklabels([])

    plt.tight_layout()
    plt.show()
    return


def show_coefficient():
    x = np.linspace(0, 1, num=2048)
    chirp_signal = np.sin(250 * np.pi * x ** 2)

    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_title("Original Chirp Signal: ")
    ax.plot(chirp_signal)
    plt.show()

    data = chirp_signal
    waveletname = 'sym5'

    fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6, 6))
    for ii in range(5):
        (data, coeff_d) = pywt.dwt(data, waveletname)
        axarr[ii, 0].plot(data, 'r')
        axarr[ii, 1].plot(coeff_d, 'g')
        axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
        axarr[ii, 0].set_yticklabels([])
        if ii == 0:
            axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
            axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
        axarr[ii, 1].set_yticklabels([])
    plt.tight_layout()
    plt.show()
    return


def show_loadedData():
    dataset = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
    df_nino = pd.read_table(dataset)
    N = df_nino.shape[0]
    t0 = 1871
    dt = 0.25
    time = np.arange(0, N) * dt + t0
    signal = df_nino.values.squeeze()

    scales = np.arange(1, 128)
    plot_signal_plus_average(time, signal)
    plot_fft_plus_power(time, signal)
    plot_wavelet(time, signal, scales)
    return


# data loading
def read_signals_ucihar(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data


def read_labels_ucihar(filename):
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities


def get_ucihar_data():
    folder_ucihar = './waveletData/UCI_HAR/'
    train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data(folder_ucihar)
    return train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar



def load_ucihar_data(folder):
    train_folder = folder + 'train/Inertial Signals/'
    test_folder = folder + 'test/Inertial Signals/'
    labelfile_train = folder + 'train/y_train.txt'
    labelfile_test = folder + 'test/y_test.txt'
    train_signals, test_signals = [], []
    print(os.listdir('waveletData'))
    print(os.listdir('./waveletData/UCI_HAR/train/Inertial Signals/'))
    print(train_folder)

    for input_file in os.listdir(train_folder):
        signal = read_signals_ucihar(train_folder + input_file)
        train_signals.append(signal)

    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    for input_file in os.listdir(test_folder):
        signal = read_signals_ucihar(test_folder + input_file)
        test_signals.append(signal)

    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    train_labels = read_labels_ucihar(labelfile_train)
    test_labels = read_labels_ucihar(labelfile_test)
    return train_signals, train_labels, test_signals, test_labels


def execute_load_uci():

    folder_ucihar = './waveletData/UCI_HAR/'
    train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data(folder_ucihar)

    uci_har_signals_train = train_signals_ucihar
    uci_har_signals_test = test_signals_ucihar
    uci_har_labels_train = train_labels_ucihar
    uci_har_labels_test = test_labels_ucihar

    scales = range(1, 128)
    waveletname = 'morl'
    train_size = 5000
    test_size = 500

    train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 9))

    for ii in range(0, train_size):
        if ii % 1000 == 0:
            print(ii)
        for jj in range(0, 9):
            signal = uci_har_signals_train[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:, :127]
            train_data_cwt[ii, :, :, jj] = coeff_

    test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 9))
    for ii in range(0, test_size):
        if ii % 100 == 0:
            print(ii)
        for jj in range(0, 9):
            signal = uci_har_signals_test[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:, :127]
            test_data_cwt[ii, :, :, jj] = coeff_

    uci_har_labels_train = list(map(lambda x: int(x) - 1, uci_har_labels_train))
    uci_har_labels_test = list(map(lambda x: int(x) - 1, uci_har_labels_test))

    x_train = train_data_cwt
    y_train = list(uci_har_labels_train[:train_size])
    x_test = test_data_cwt
    y_test = list(uci_har_labels_test[:test_size])

    history = History()

    img_x = 127
    img_y = 127
    img_z = 9
    input_shape = (img_x, img_y, img_z)

    num_classes = 6
    batch_size = 16
    num_classes = 7
    epochs = 10

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    return

#######################################################################################################################

def plot_wavelet(time, signal, scales,
                 waveletname='cmor',
                 cmap=plt.cm.seismic,
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='Period (years)',
                 xlabel='Time'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()


def plot_signal_plus_average(time, signal, average_over=5):
    fig, ax = plt.subplots(figsize=(15, 3))
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label='time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Signal Amplitude', fontsize=18)
    ax.set_title('Signal + Time Average', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)
    ax.legend()
    plt.show()


def plot_fft_plus_power(time, signal):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1 / dt

    fig, ax = plt.subplots(figsize=(15, 3))
    variance = np.std(signal) ** 2
    f_values, fft_values = get_fft_values(signal, dt, N)
#    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
    ax.plot(f_values, fft_values, 'r-', label='Fourier Transform')
    ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT Power Spectrum')
    ax.set_xlabel('Frequency [Hz / year]', fontsize=18)
    ax.set_ylabel('Amplitude', fontsize=18)
    ax.legend()
    plt.show()

