import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean


def loss_mse_warmup(y_true, y_pred,  warmup_steps):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    y_true is the desired output.
    y_pred is the model's output.
    """
    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].
    warmup_steps = 50
    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the Mean Squared Error and use it as loss.
    mse = mean(square(y_true_slice - y_pred_slice))

    return mse


def load_data():
    # load file to dataset
    df = pd.read_csv('daily-climate-train.csv', index_col='date', parse_dates=['date'])

    df.head()
    # df['Esbjerg']['Pressure'].plot();
    # df['Roskilde']['Pressure'].plot();
    df.values.shape
    # df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
    # df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)
    # df.values.shape
    # df.head(1)
    # df['Odense']['Temp']['2006-05':'2006-07'].plot();
    # df['Aarhus']['Temp']['2006-05':'2006-07'].plot();
    # df['Roskilde']['Temp']['2006-05':'2006-07'].plot();
    # df['Various', 'Day'] = df.index.dayofyear
    # df['Various', 'Hour'] = df.index.hour
    #
    # target_city = 'Odense'
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

    return x_train_scaled, y_train_scaled,  x_test_scaled, y_test_scaled, num_x_signals, num_y_signals, num_train


def batch_generator(batch_size, sequence_length,
                    x_train_scaled, y_train_scaled, num_x_signals, num_y_signals, num_train):
    """
    Generator function for creating random batches of training-data.
    """
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]
        yield x_batch, y_batch
    # return x_batch, y_batch


def build_model(x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, num_x_signals, num_y_signals, num_train):

    batch = 0  # First sequence in the batch.
    batch_size = 256
    # sequence_length = 24 * 7 * 8
    sequence_length = 1 * 7 * 8
    # buid batch Data and Function
    generator = batch_generator(x_train_scaled, y_train_scaled, num_x_signals, num_y_signals, num_train,
                                batch_size=batch_size, sequence_length=sequence_length)
    x_batch, y_batch = next(generator)
    print(x_batch.shape)
    print(y_batch.shape)

    signal = 0  # First signal from the 20 input-signals.
    seq = x_batch[batch, :, signal]
    plt.plot(seq)

    validation_data = (np.expand_dims(x_test_scaled, axis=0), np.expand_dims(y_test_scaled, axis=0))

    # ###################### CREATE MODEL ####################################
    makeModel = True
    model_metrics = False
    addMoreModel = True
    optimizer = RMSprop(lr=1e-3)

    if makeModel:
        model = Sequential()
        model.add(GRU(units=512, return_sequences=True, input_shape=(None, num_x_signals)))
        model.add(Dense(num_y_signals, activation='sigmoid'))
        if addMoreModel:
            from tensorflow.keras.initializers import RandomUniform

            # Maybe use lower init-ranges.
            init = RandomUniform(minval=-0.05, maxval=0.05)
            model.add(Dense(num_y_signals, activation='linear', kernel_initializer=init))
        model.compile(loss=loss_mse_warmup, optimizer=optimizer)
        model.summary()

        # ###################### CALLBACK FUNCTION ####################################
        path_checkpoint = '23_checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                              verbose=1, save_weights_only=True, save_best_only=True)
        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        callback_tensorboard = TensorBoard(log_dir='./23_logs/', histogram_freq=0, write_graph=False)
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1)
        callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard,
                     callback_reduce_lr]
        model.fit(x=generator, epochs=20, steps_per_epoch=100, validation_data=validation_data, callbacks=callbacks)
        try:
            model.load_weights(path_checkpoint)
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

        result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                                y=np.expand_dims(y_test_scaled, axis=0))

        print("loss (test-set):", result)

        # If you have several metrics you can use this instead.
        if model_metrics:
            for res, metric in zip(result, model.metrics_names):
                print("{0}: {1:.3e}".format(metric, res))

    # show graph

    model.save_weights('weather_gru_weight.h5')
    model.save('weather_gru.h5')
