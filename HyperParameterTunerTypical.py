import keras_tuner as kt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical

IMAGE_H=224
IMAGE_W=224
BATCH_SIZE=32

x_train = np.load("train.npy", allow_pickle=True)
y_train = np.load("train_labels.npy", allow_pickle=True)

x_val = np.load("val.npy", allow_pickle=True)
y_val = np.load("val_labels.npy", allow_pickle=True)

x_test = np.load("test.npy", allow_pickle=True)
y_test = np.load("test_labels.npy", allow_pickle=True)

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)

print(x_test.shape)
print(y_test.shape)


def model_builder(hp):
    model = tf.keras.Sequential()
    input_shape = (IMAGE_H, IMAGE_W, 3)
    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])

    model.add(tf.keras.layers.Conv2D(hp.Int("conv_1", min_value=32, max_value=256, step=32), (3, 3), padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.Activation(hp_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(hp.Int("conv_2", min_value=32, max_value=256, step=32), (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation(hp_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(hp.Int("conv_3", min_value=32, max_value=256, step=32), (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation(hp_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(hp.Int("conv_4", min_value=32, max_value=256, step=32), (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation(hp_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(hp.Int("conv_5", min_value=32, max_value=256, step=32), (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation(hp_activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    hp_layer_2 = hp.Int('layer_2', min_value=64, max_value=512, step=32)

    model.add(Dense(units=hp_layer_2, activation=hp_activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=30, factor=3, directory='dir', project_name='x')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train, epochs=20, verbose=1, validation_data=(x_val, y_val), shuffle=True, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units:
Conv_1  {best_hps.get('conv_1')},
Conv_2  {best_hps.get('conv_2')},
Conv_3  {best_hps.get('conv_3')},
Conv_4  {best_hps.get('conv_4')},
Conv_5  {best_hps.get('conv_5')},
DenseLayer2  {best_hps.get('layer_2')},
Activation  {best_hps.get('activation')},
LearningRate  {best_hps.get('learning_rate')}
""")

model = tuner.hypermodel.build(best_hps)

model.summary()
