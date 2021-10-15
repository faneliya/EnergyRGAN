import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import GRU_Model
from sklearn.preprocessing import MinMaxScaler


def load_model(mode, customizedLoss):
    # 훈련된 모델을 불려온다
    if customizedLoss: # 새로 정의된 loss 함수 사용여부
        if mode == 'Solar':
            model = tf.keras.models.load_model('solar_gru2.h5',
                                               custom_objects={'loss_mse_warmup': GRU_Model.loss_mse_warmup})
            model.load_weights('solar_gru_weight2.h5')
        elif mode == 'Wind':
            model = tf.keras.models.load_model('wind_gru2.h5',
                                               custom_objects={'loss_mse_warmup': GRU_Model.loss_mse_warmup})
            model.load_weights('wind_gru_weight2.h5')
    else:
        if mode == 'Solar':
            model = tf.keras.models.load_model('solar_gru_normal2.h5')
            model.load_weights('solar_gru_weight_normal2.h5')
        elif mode == 'Wind':
            model = tf.keras.models.load_model('solar_gru_normal2.h5')
            model.load_weights('solar_gru_weight_normal2.h5')
    return model


def plot_comparison(mode, start_idx, target_names, x_train_scaled, x_test_scaled, y_train, y_test, xScaler, yScaler, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    :param target_names:
    :param y_test:
    :param y_train:
    :param x_test_scaled:
    :param x_train_scaled:
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    #xScaler = MinMaxScaler()
    #yScaler = MinMaxScaler()
    warmup_steps = length
    customizedLoss = False
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    model = load_model(mode, customizedLoss)
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_pred
    y_pred_rescaled = yScaler.inverse_transform(y_pred[0])
    y_true_rescaled = yScaler.inverse_transform(y_true)
    # y_true_rescaled = y_true
    print(y_pred)
    print('-----------------------------------')
    print('-----------------------------------')
    print(y_true)
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]

        # Get the true output-signal from the data-set.
        # signal_true = y_true[:, signal]
        signal_true = y_true_rescaled[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15, 5))

        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()




