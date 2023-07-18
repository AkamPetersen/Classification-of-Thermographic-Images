
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, InputLayer, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os
import cv2
import glob

from alibi_detect.od import OutlierAE, OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
"""
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""

trainset = np.load("survived.npy", allow_pickle=True)
#trainset2 = np.load("survived.npy", allow_pickle=True)
testset1 = np.load("survived_test.npy", allow_pickle=True)
testset2 = np.load("not_survived.npy", allow_pickle=True)
#testset1 = np.load("augmented_survived_test.npy", allow_pickle=True)
#testset2 = np.load("augmented_not_survived.npy", allow_pickle=True)

print(trainset.shape)
#print(trainset2.shape)
print(testset1.shape)
print(testset2.shape)

encoding_dim = 1024  #Dimension of the bottleneck encoder vector.
dense_dim = [10, 10, 512] #Dimension of the last conv. output. This is used to work our way back in the decoder.

#Define encoder
encoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=trainset[0].shape),
      Conv2D(32, 3, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(64, 3, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(128, 3, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(512, 3, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
      Dense(encoding_dim,)
  ])

print(encoder_net.summary())

#Define the decoder.
decoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=(encoding_dim,)),
      Dense(np.prod(dense_dim)),
      Reshape(target_shape=dense_dim),
      Conv2DTranspose(128, 3, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
  ])

print(decoder_net.summary())

latent_dim = 1024  #(Same as encoding dim. )

# initialize outlier detector
od = OutlierVAE(threshold=0.0001,  # threshold for outlier score above which the element is flagged as an outlier.
                score_type='mse',  # use MSE of reconstruction error for outlier detection
                encoder_net=encoder_net,  # can also pass VAE model instead
                decoder_net=decoder_net,  # of separate encoder and decoder
                latent_dim=latent_dim,
                samples=2)

print("Current threshold value is: ", od.threshold)

# Train

adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

od.fit(trainset,
       optimizer = adam,
       epochs=200,
       batch_size=4,
       verbose=True)

print("Current threshold value is: ", od.threshold)

from alibi_detect.utils import save_detector, load_detector
#from alibi_detect.utils.saving import save_detector, load_detector
save_detector(od, "saved_SVS_models.h5")
#od = load_detector("saved_SVS_models.h5")


X = testset2
od_preds2 = od.predict(X,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)

print(list(od_preds2['data'].keys()))

#Scatter plot of instance scores. using the built-in function for the scatterplot.
target2 = np.ones(X.shape[0],).astype(int)  # Ground truth (all ones for bad images)
labels2 = ['normal', 'outlier']
plot_instance_score(od_preds2, target2, labels2, od.threshold) #pred, target, labels, threshold

Y = testset1
od_preds1 = od.predict(Y,
                      outlier_type='instance',    # use 'feature' or 'instance' level
                      return_feature_score=True,  # scores used to determine outliers
                      return_instance_score=True)

print(list(od_preds1['data'].keys()))

#Scatter plot of instance scores. using the built-in function for the scatterplot.
target1 = np.ones(Y.shape[0],).astype(int)  # Ground truth (all ones for bad images)
labels1 = ['normal', 'outlier']
plot_instance_score(od_preds1, target1, labels1, od.threshold) #pred, target, labels, threshold



