import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import GRU, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, ELU, ReLU
from tensorflow.keras import Sequential, regularizers
from tensorflow.python.client import device_lib

######################################################################################################################
# Constant Settings
DataFilesDir="./DataFiles/"
ProcessedFilesDir="./ProcessedFiles/"
ModelFileDir="./ModelSaves/"

TrainCaseName = 'SolarAllDataM3FFT'
#TrainCaseName = 'SolarAllDataM3FFT_DWT'
#TrainCaseName = 'WindAllDataM3FFT'
#TrainCaseName = 'WindAllDataM3FFT_DWT'
#TrainCaseName = 'BelgiumAllDataM3FFT'
#TrainCaseName = 'BelgiumAllDataM3FFT_DWT'

DataVersion = 'simple_'

if TrainCaseName is not None:
    #Train Data 8 objects
    X_train     = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "X_train.npy", allow_pickle=True)
    y_train     = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "y_train.npy", allow_pickle=True)
    X_test      = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "X_test.npy", allow_pickle=True)
    y_test      = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "y_test.npy", allow_pickle=True)
    yc_train    = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "yc_train.npy", allow_pickle=True)
    yc_test     = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "yc_test.npy", allow_pickle=True)
    yScaler     = ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "TargetValueScaler.pkl"
    xScaler     = ProcessedFilesDir + TrainCaseName + "_" + DataVersion + "BaseValueScaler.pkl"
else:
    X_train = np.load("X_train.npy", allow_pickle=True)
    y_train = np.load("y_train.npy", allow_pickle=True)
    X_test = np.load("X_test.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    yc_train = np.load("yc_train.npy", allow_pickle=True)
    yc_test = np.load("yc_test.npy", allow_pickle=True)
    yScaler = 'y_scaler.pkl'
    xScaler = 'X_scaler.pkl'

# Define the generator
def Generator(input_dim, output_dim, feature_size) -> tf.keras.models.Model:
    model = Sequential()
    model.add(GRU(units=256,
                  return_sequences=True,
                  input_shape=(input_dim, feature_size),
                  recurrent_dropout=0.02,
                  recurrent_regularizer=regularizers.l2(1e-3)))
    model.add(GRU(units=128,
                  #return_sequences=True,
                  recurrent_dropout=0.02,
                  recurrent_regularizer=regularizers.l2(1e-3)))
    #model.add(Dense(128,
    #              kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dense(64, kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dense(32, kernel_regularizer=regularizers.l2(1e-3)))
    #model.add(Dense(16, kernel_regularizer=regularizers.l2(1e-3)))
    #model.add(Dense(8, kernel_regularizer=regularizers.l2(1e-3)))
    model.add(Dense(units=output_dim))
    return model


# Define the discriminator
def Discriminator() -> tf.keras.models.Model:
    model = tf.keras.Sequential()
    if DataFilesDir is not None:
        model.add(
            Conv1D(32, input_shape=(14, 1), kernel_size=3, strides=2, padding='same', activation=LeakyReLU(alpha=0.01)))
    else:
        model.add(Conv1D(32, input_shape=(4, 1), kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    #model.add(Conv1D(32, input_shape=(4, 1), kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Conv1D(64, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Conv1D(128, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(alpha=0.01)))
    model.add(Flatten())
    model.add(Dense(220, use_bias=True))
    model.add(LeakyReLU())
    model.add(Dense(220, use_bias=True))
    model.add(ReLU())
    model.add(Dense(1))
    return model


# Train WGAN-GP model
class GAN():
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.d_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.g_optimizer = tf.keras.optimizers.Adam(0.0001)
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = 128
        self.checkpoint_dir = './training_checkpoints' + TrainCaseName + "_" + DataVersion + 'WGAN'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.g_optimizer,
                                              discriminator_optimizer=self.d_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

    def gradient_penalty(self, batch_size, real_output, fake_output):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interpolated data
        if DataFilesDir is not None:
            alpha = tf.random.normal([batch_size, 14, 1], 0.0, 1.0) # check data shape
        else:
            alpha = tf.random.normal([batch_size, 4, 1], 0.0, 1.0)
        diff = fake_output - tf.cast(real_output, tf.float32)
        interpolated = tf.cast(real_output, tf.float32) + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]

        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))

        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_input, real_value, yc = data
        batch_size = tf.shape(real_input)[0]
        for _ in range(1):
            with tf.GradientTape() as d_tape:
                # Train the discriminator
                # generate fake output
                generated_data = self.generator(real_input, training=True)
                # reshape the data
                generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
                fake_output = tf.concat([generated_data_reshape, tf.cast(yc, tf.float32)], axis=1)
                real_y_reshape = tf.reshape(real_value, [real_value.shape[0], real_value.shape[1], 1])
                real_output = tf.concat([tf.cast(real_y_reshape, tf.float32), tf.cast(yc, tf.float32)], axis=1)
                # Get the logits for the fake images
                D_real = self.discriminator(real_output, training=True)
                # Get the logits for real images
                D_fake = self.discriminator(fake_output, training=True)
                # Calculate discriminator loss using fake and real logits
                real_loss = tf.cast(tf.reduce_mean(D_real), tf.float32)
                fake_loss = tf.cast(tf.reduce_mean(D_fake), tf.float32)
                d_cost = fake_loss-real_loss
                # Calculate the gradientjiu penalty
                gp = self.gradient_penalty(batch_size, real_output, fake_output)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * 10

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        for _ in range(3):
            with tf.GradientTape() as g_tape:
                # Train the generator
                # generate fake output
                generated_data = self.generator(real_input, training=True)
                # reshape the data
                generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
                fake_output = tf.concat([generated_data_reshape, tf.cast(yc, tf.float32)], axis=1)
                # Get the discriminator logits for fake images
                G_fake = self.discriminator(fake_output, training=True)
                # Calculate the generator loss
                g_loss = -tf.reduce_mean(G_fake)
            g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return real_value, generated_data, {'d_loss': d_loss, 'g_loss': g_loss}

    def train(self, X_train, y_train, yc, epochs):
        data = X_train, y_train, yc
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []


        for epoch in range(epochs):
            start = time.time()

            real_value, fake_value, loss = self.train_step(data)

            G_losses = []
            D_losses = []

            RealValue = []
            PredictedValue = []

            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())

            PredictedValue.append(fake_value)
            RealValue.append(real_value)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                tf.keras.models.save_model(generator, 'gen_GRU_model_%d.h5' % epoch)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print('epoch', epoch+1, 'd_loss', loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())

            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)

        ###########################################################################
        print("#WGAN SAVING MODEL....")
        tf.keras.models.save_model(generator, ModelFileDir + TrainCaseName + '_' + 'WGanGeneratorModel.h5')
        ##########################################################################
            
        # Reshape the predicted result & real
        PredictedValue = np.array(PredictedValue)
        PredictedValue = PredictedValue.reshape(PredictedValue.shape[1], PredictedValue.shape[2])
        RealValue = np.array(RealValue)
        RealValue = RealValue.reshape(RealValue.shape[1], RealValue.shape[2])

        # Plot the loss
        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(TrainCaseName + " : WGAN result of Loss(Generator/Discriminator)")
        plt.tight_layout()
        plt.legend()
        plt.savefig('./PICS/' + TrainCaseName + '_WGAN_Loss.png')
        plt.show()

        print("REAL", RealValue.shape)
        print(RealValue)
        print("PREDICTED", PredictedValue.shape)
        print(PredictedValue)

        return PredictedValue, RealValue, np.sqrt(mean_squared_error(RealValue, PredictedValue)) / np.mean(RealValue)

# %% --------------------------------------- Plot the result -----------------------------------------------------

## TRAIN DATA
def plot_traindataset_result(RealValue, PredictedValue):
    X_scaler = load(open('X_scaler.pkl', 'rb'))
    y_scaler = load(open('y_scaler.pkl', 'rb'))
    train_predict_index = np.load("index_train.npy", allow_pickle=True)
    test_predict_index = np.load("index_test.npy", allow_pickle=True)
    
    print("----- predicted price -----", PredictedValue)
    
    rescaled_RealValue = y_scaler.inverse_transform(RealValue)
    rescaled_PredictedValue = y_scaler.inverse_transform(PredictedValue)
    
    print("----- rescaled predicted price -----", rescaled_PredictedValue)
    print("----- SHAPE rescaled predicted price -----", rescaled_PredictedValue.shape)
    
    predict_result = pd.DataFrame()
    for i in range(rescaled_PredictedValue.shape[0]):
        y_predict = pd.DataFrame(rescaled_PredictedValue[i], columns=["predicted_value"], index=train_predict_index[i:i+output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
    
    real_value = pd.DataFrame()
    for i in range(rescaled_RealValue.shape[0]):
        y_train = pd.DataFrame(rescaled_RealValue[i], columns=["real_value"], index=train_predict_index[i:i+output_dim])
        real_value = pd.concat([real_value, y_train], axis=1, sort=False)
    
    predict_result['PREDICTED_MEAN'] = predict_result.mean(axis=1)
    real_value['REAM_MEAN'] = real_value.mean(axis=1)
    
    # Calculate RMSE
    predicted = predict_result["PREDICTED_MEAN"]
    real = real_value["REAM_MEAN"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- RMSE -- ', RMSE)

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_value["REAM_MEAN"])
    plt.plot(predict_result["PREDICTED_MEAN"], color='r')
    plt.xlabel("DATE")
    plt.ylabel("Real Value")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title(TrainCaseName + " : GAN result of Training, RMSE=" + str(RMSE), fontsize=16)
    plt.tight_layout()
    plt.savefig('./PICS/'+TrainCaseName + '_wGAN_traindataset.png')
    plt.show()
    
    
    
    
def plot_testdataset_result(X_test, y_test, model):

    print(TrainCaseName + "Predicting Data...START")
    test_yhat = model.predict(X_test, verbose=0)
    print(TrainCaseName + "Predicting Data...END")

    y_scaler = load(open(yScaler, 'rb'))
    test_predict_index = np.load(ProcessedFilesDir + TrainCaseName + "_" + DataVersion +
                                 "test_predict_index.npy", allow_pickle=True)

    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(test_yhat)

    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["PREDICT_VALUE"],
                                 index=test_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_value = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["REAL_VALUE"],
                               index=test_predict_index[i:i + output_dim])
        real_value = pd.concat([real_value, y_train], axis=1, sort=False)

    predict_result["PREDICTED_MEAN"] = predict_result.mean(axis=1)
    real_value["REAL_MEAN"] = real_value.mean(axis=1)

    # Calculate RMSE
    predicted = predict_result["PREDICTED_MEAN"]
    real = real_value["REAL_MEAN"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_value["REAL_MEAN"])
    plt.plot(predict_result["PREDICTED_MEAN"], color='r')
    plt.xlabel("DATE")
    plt.ylabel("Real Value")
    plt.legend(("Real Value", "Predicted Value"), loc="upper left", fontsize=16)
    plt.title(TrainCaseName + " wGAN : result of Training, RMSE=" + str(RMSE), fontsize=16)
    plt.tight_layout()
    plt.savefig('./PICS/'+TrainCaseName + '_wGAN_testdataset.png')
    plt.show()
    print('-- Test RMSE -- ', RMSE)
    return RMSE


if __name__ == '__main__':
    input_dim = X_train.shape[1]
    feature_size = X_train.shape[2]
    output_dim = y_train.shape[1]
    epoch = 100

    ##################### Building Models #########################################
    generator = Generator(X_train.shape[1], output_dim, X_train.shape[2])
    discriminator = Discriminator()
    gan = GAN(generator, discriminator)
    PredictedValue, RealValue, RMSPE = gan.train(X_train, y_train, yc_train, epoch)

'''
    ###############################################################################
    print('#Loading GAN Model-----------------')
    GanModel = tf.keras.models.load_model(ModelFileDir + TrainCaseName + '_' + 'WGanGeneratorModel.h5')
    # print(GanModel.summary())

    #################### plot train Model #########################################
    print("PREDICT DATASET PROCESSING......" + TrainCaseName)
    PredictedValue = GanModel.predict(X_train, verbose=0)
    plot_traindataset_result(y_train, PredictedValue)

    #################### plot test Model #########################################
    print("PREDICT TESTSET PROCESSING......" + TrainCaseName)
    plot_testdataset_result(X_test, y_test, GanModel)
'''

