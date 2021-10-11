import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(fileName) -> object:
    # load file to dataset
    try:
        df = pd.read_csv(fileName, index_col='row')
    except Exception as error:
        print("Error load file.")
        print(error)
        return

    df.head()
    print(df.values.shape)
    target_names = ['1st', '2st', '3st']
    shift_days = 1
    # shift_steps = shift_days * 24  # Number of hours.
    shift_steps = shift_days * 4 * 24  # Number of hours.
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

    np.save("X_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", x_test)
    np.save("y_test.npy", y_test)
    np.save("yc_train.npy", yc_train)
    np.save("yc_test.npy", yc_test)
    np.save('index_train.npy', index_train)
    np.save('index_test.npy', index_test)
    np.save('train_predict_index.npy', index_train)
    np.save('test_predict_index.npy', index_test)

    return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, \
           num_x_signals, num_y_signals, num_train, xScaler, yScaler
