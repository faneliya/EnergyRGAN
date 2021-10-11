import tensorflow as tf

def trainData():
    n_train_time = 365*24

    train = values[:n_train_time, :]
    test = values[n_train_time:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
    model = tf.model.Sequential()
    model.add(tf.model.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
