import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from main.feature import get_all_features
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Sequential
from pickle import load
from sklearn.metrics import mean_squared_error


# Constant Settings
DataFilesDir="./DataFiles/"
ProcessedFilesDir="./ProcessedFiles/"
ModelFileDir="./ModelSaves/"

######################################################################################################################
#TrainCaseName = 'SolarAllDataM3FFT'
#TrainCaseName = 'SolarAllDataM3FFT_DWT'
#TrainCaseName = 'WindAllDataM3FFT'
#TrainCaseName = 'WindAllDataM3FFT_DWT'
#TrainCaseName = 'BeligumAllDataM3FFT'
TrainCaseName = 'BeligumAllDataM3FFT_DWT'

if TrainCaseName is not None:
    #Train Data 8 objects
    X_train     = np.load(ProcessedFilesDir + TrainCaseName + "_" + "X_train.npy", allow_pickle=True)
    y_train     = np.load(ProcessedFilesDir + TrainCaseName + "_" + "y_train.npy", allow_pickle=True)
    X_test      = np.load(ProcessedFilesDir + TrainCaseName + "_" + "X_test.npy", allow_pickle=True)
    y_test      = np.load(ProcessedFilesDir + TrainCaseName + "_" + "y_test.npy", allow_pickle=True)
    yc_train    = np.load(ProcessedFilesDir + TrainCaseName + "_" + "yc_train.npy", allow_pickle=True)
    yc_test     = np.load(ProcessedFilesDir + TrainCaseName + "_" + "yc_test.npy", allow_pickle=True)
    yScaler     = ProcessedFilesDir + TrainCaseName + "_" + "TargetValueScaler.pkl"
    xScaler     = ProcessedFilesDir + TrainCaseName + "_" + "BaseValueScaler.pkl"
else:
    X_train = np.load("X_train.npy", allow_pickle=True)
    y_train = np.load("y_train.npy", allow_pickle=True)
    X_test = np.load("X_test.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    yc_train = np.load("yc_train.npy", allow_pickle=True)
    yc_test = np.load("yc_test.npy", allow_pickle=True)
    yScaler = 'y_scaler.pkl'
    xScaler = 'X_scaler.pkl'


def make_generator_model(input_dim, output_dim, feature_size) -> tf.keras.models.Model:

    model = Sequential()
    model.add(GRU(units=1024, return_sequences = True, input_shape=(input_dim, feature_size),
                  recurrent_dropout=0.2))
    model.add(GRU(units=512, return_sequences = True, recurrent_dropout=0.2)) # 256, return_sequences = True
    model.add(GRU(units=256, recurrent_dropout=0.2)) #, recurrent_dropout=0.1
    # , recurrent_dropout = 0.2
    model.add(Dense(128))
    # model.add(Dense(128))
    model.add(Dense(64))
    #model.add(Dense(16))
    model.add(Dense(units=output_dim))
    return model


def make_discriminator_model():

    cnn_net = tf.keras.Sequential()
    cnn_net.add(Conv1D(32, input_shape=(4, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(64, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(128, kernel_size=5, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Flatten())
    cnn_net.add(Dense(220, use_bias=False))
    cnn_net.add(LeakyReLU())
    cnn_net.add(Dense(220, use_bias=False, activation='relu'))
    cnn_net.add(Dense(1, activation='sigmoid'))
    return cnn_net



class GAN:
    def __init__(self, generator, discriminator, opt):
        self.opt = opt
        self.lr = opt["lr"]
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.batch_size = self.opt['bs']
        self.checkpoint_dir = '../training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_x, real_y, yc):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(real_x, training=True)
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat([tf.cast(generated_data_reshape, tf.float64), yc], axis=1)
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
            d_real_input = tf.concat([real_y_reshape, yc], axis=1)

            # Reshape for MLP
            # d_fake_input = tf.reshape(d_fake_input, [d_fake_input.shape[0], d_fake_input.shape[1]])
            # d_real_input = tf.reshape(d_real_input, [d_real_input.shape[0], d_real_input.shape[1]])

            real_output = self.discriminator(d_real_input, training=True)
            fake_output = self.discriminator(d_fake_input, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': gen_loss}

    def train(self, real_x, real_y, yc, opt):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []

        epochs = opt["epoch"]
        for epoch in range(epochs):
            start = time.time()

            real_value, fake_price, loss = self.train_step(real_x, real_y, yc)

            G_losses = []
            D_losses = []

            RealValue = []
            PredictedValue = []

            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())

            PredictedValue.append(fake_price.numpy())
            RealValue.append(real_value.numpy())

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                tf.keras.models.save_model(generator, 'gen_model_3_1_%d.h5' % epoch)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix + f'-{epoch}')
                print('epoch', epoch + 1, 'd_loss', loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())
            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)

        ###########################################################################
        tf.keras.models.save_model(generator, ModelFileDir + TrainCaseName + '_' + 'GanGeneratorModel.h5')
        ##########################################################################

        # Reshape the predicted result & real
        PredictedValue = np.array(PredictedValue)
        PredictedValue = PredictedValue.reshape(PredictedValue.shape[1], PredictedValue.shape[2])
        RealValue = np.array(RealValue)
        RealValue = RealValue.reshape(RealValue.shape[1], RealValue.shape[2])

        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return PredictedValue, RealValue, np.sqrt(mean_squared_error(RealValue, PredictedValue)) / np.mean(RealValue)


## TRAIN DATA
def plot_traindataset_result(RealValue, PredictedValue):
    # Rescale back the real dataset
    X_scaler = load(open(xScaler, 'rb'))
    y_scaler = load(open(yScaler, 'rb'))
    train_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                  + "train_predict_index.npy",  allow_pickle=True)
    test_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_"
                                 + "test_predict_index.npy", allow_pickle=True)
    # dataset_train = pd.read_csv('dataset_train.csv', index_col=0)

    rescaled_RealValue = y_scaler.inverse_transform(RealValue)
    rescaled_PredictedValue = y_scaler.inverse_transform(PredictedValue)

    print("----- rescaled predicted price -----", rescaled_PredictedValue)
    print("----- SHAPE rescaled predicted price -----", rescaled_PredictedValue.shape)

    predict_result = pd.DataFrame()
    for i in range(rescaled_PredictedValue.shape[0]):
        y_predict = pd.DataFrame(rescaled_PredictedValue[i], columns=["PREDICTED_VALUE"],
                                 index=train_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
    #
    real_value = pd.DataFrame()
    for i in range(rescaled_RealValue.shape[0]):
        y_train = pd.DataFrame(rescaled_RealValue[i], columns=["REAL_VALUE"],
                               index=train_predict_index[i:i + output_dim])
        real_value = pd.concat([real_value, y_train], axis=1, sort=False)

    predict_result['PREDICTED_MEAN'] = predict_result.mean(axis=1)
    real_value['REAM_MEAN'] = real_value.mean(axis=1)

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_value["REAM_MEAN"])
    plt.plot(predict_result["PREDICTED_MEAN"], color='r')
    plt.xlabel("DATE")
    plt.ylabel("Real Value")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title(TrainCaseName + " : result of Training", fontsize=20)
    plt.show()

    # Calculate RMSE
    predicted = predict_result["PREDICTED_MEAN"]
    real = real_value["REAM_MEAN"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Train RMSE -- ', RMSE)


if __name__ == '__main__':

    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]

    ## For Bayesian
    opt = {"lr": 0.00016, "epoch": 165, 'bs': 128}

    print('#GAN-Discriminator Model------------------------------------')
    showDiscrimModel = make_discriminator_model()
    print(showDiscrimModel.summary())
    print('#GAN-Generator Model------------------------------------')
    showGanModel = make_generator_model()
    print(showGanModel.summary())

    ##################### Building Models #########################################
    generator = make_generator_model(X_train.shape[1], output_dim, X_train.shape[2])
    discriminator = make_discriminator_model()
    gan = GAN(generator, discriminator, opt)

    PredictedValue, RealValue, RMSPE = gan.train(X_train, y_train, yc_train, opt)
    ###############################################################################

    GanModel = tf.keras.models.load_model(ModelFileDir + TrainCaseName + '_' + 'GanGeneratorModel.h5')
    print(GanModel.summary())
    print("PREDICT PROCESSING......" + TrainCaseName)
    PredictedValue = GanModel.predict(X_train, verbose=0)
    plot_traindataset_result(y_train, PredictedValue)
    #################### Prediction Model #########################################
