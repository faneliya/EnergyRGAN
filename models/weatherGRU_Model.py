import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean


def loss_mse_warmup(y_true, y_pred):
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
    generator = batch_generator(batch_size, sequence_length,
                                x_train_scaled, y_train_scaled, num_x_signals, num_y_signals, num_train)
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
    customizedLoss = False
    optimizer = RMSprop(lr=1e-3)
    model = Sequential()
    if makeModel:
        model.add(GRU(units=512, return_sequences=True, input_shape=(None, num_x_signals)))
        model.add(Dense(num_y_signals, activation='sigmoid'))
        if addMoreModel:
            from tensorflow.keras.initializers import RandomUniform

            # Maybe use lower init-ranges.
            init = RandomUniform(minval=-0.05, maxval=0.05)
            model.add(Dense(num_y_signals, activation='linear', kernel_initializer=init))
        if customizedLoss:
            model.compile(loss=loss_mse_warmup, optimizer=optimizer)
        else:
            model.compile(loss='mse', optimizer=optimizer)
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

    if customizedLoss:
        model.save_weights('weather_gru_weight.h5')
        model.save('weather_gru.h5')
    else:
        model.save_weights('weather_gru_weight_normal.h5')
        model.save('weather_gru_normal.h5')
