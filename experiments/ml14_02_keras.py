import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(0)


def make_random_data():
    x = np.random.uniform(low=-2, high=2, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t * t / 3), size=None)
        y.append(r)
    return x, 1.726 * x - 0.84 + np.array(y)


x, y = make_random_data()
plt.plot(x, y, 'o')
plt.show()

# train, test set
x_train, y_train = x[:150], y[:150]
x_test, y_test = x[150:], y[150:]

# model1 #############################################
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))

model.compile(optimizer='sgd', loss='mse')
history = model.fit(x_train, y_train, epochs=500, validation_split=0.3)

epochs = np.arange(1, 500 + 1)
plt.plot(epochs, history.history['loss'], label='Traiing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save_weights('simple_weights.h5')
model.save('simple_model.h5')

# model2 #############################################

inputLayer = tf.keras.Input(shape=(1,))
# output = tf.keras.layers.Dense(1)(input)
dense = tf.keras.layers.Dense(1)
output = dense(inputLayer)
model2 = tf.keras.Model(inputLayer, output)
model2.summary()
model2.compile(optimizer='sgd', loss='mse')

history2 = model2.fit(x_train, y_train, epochs=500, validation_split=0.3)
plt.plot(epochs, history2.history['loss'], label='Traiing Loss')
plt.plot(epochs, history2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model2.evaluate(x_test, y_test)

# model3 #############################################
callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='simple_weights.h5',
                                                    monitor='val_loss',
                                                    save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(patience=5)]

history3 = model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=callback_list)

epochs3 = np.arange(1, len(history3.history['loss'])+1)
plt.plot(epochs3, history3.history['loss'], label='Training Loss')
plt.plot(epochs3, history3.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs3')
plt.ylabel('Loss')
plt.legend()
plt.show()

# model4 #############################################

model4 = tf.keras.models.load_model('simple_model.h5')
model4.load_weights('simple_weights.h5')
model4.evaluate(x_test, y_test)

x_arr = np.arange(-2, 2, 0.1)
y_arr = model.predict(x_arr)

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', alpha=0.3)
plt.plot(x_arr, y_arr, '-r', lw=3)
plt.show()
