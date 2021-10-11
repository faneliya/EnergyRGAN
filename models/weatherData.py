import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data() -> object:
    # load file to dataset
    df = pd.read_csv('daily-climate-train.csv', index_col='date', parse_dates=['date'])
    df.head()
    print(df.values.shape)
    target_names = ['temperature', 'humidity', 'windspeed']
    shift_days = 1
    # shift_steps = shift_days * 24  # Number of hours.
    shift_steps = shift_days * 1  # Number of hours.
    df_targets = df[target_names].shift(-shift_steps)
    df[target_names].head(shift_steps + 5)

    df_targets.head(5)
    df_targets.tail()

    # ######################  TRAIN DATA ####################################
    x_data = df.values[0:-shift_steps]
    y_data = df_targets.values[:-shift_steps]

    print(type(x_data))
    print("Shape:", x_data.shape)
    print(type(y_data))
    print("Shape:", y_data.shape)

    num_data = len(x_data)
    train_split = 0.9
    num_train = int(train_split * num_data)
    # num_test = num_data - num_train

    x_train = x_data[0:num_train]
    y_train = y_data[0:num_train]
    x_test = x_data[num_train:]
    y_test = y_data[num_train:]

    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]
    print('total Data Size = ' + str(len(x_train) + len(x_test)) + 'num of x signal =' + str(num_x_signals))
    print('total Data Size = ' + str(len(y_train) + len(x_test)) + 'num of y signal =' + str(num_y_signals))
    print("Min:", np.min(x_train))
    print("Max:", np.max(x_train))

    xScaler = MinMaxScaler()
    yScaler = MinMaxScaler()

    x_train_scaled = xScaler.fit_transform(x_train)
    y_train_scaled = yScaler.fit_transform(y_train)
    x_test_scaled = xScaler.transform(x_test)
    y_test_scaled = yScaler.transform(y_test)
    print(x_train_scaled.shape)
    print(y_train_scaled.shape)

    return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, \
           num_x_signals, num_y_signals, num_train, xScaler, yScaler
