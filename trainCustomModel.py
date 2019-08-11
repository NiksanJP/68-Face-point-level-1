import tensorflow as tf 
from tensorflow.keras.regularizers import l2
import numpy as np 
import os
import gc        
import cv2

import matplotlib.pyplot as plt     

model = tf.keras.models.Sequential([
    tf.keras.layers.Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(100,100,1)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(32, (3,3), padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(64, (3,3), padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(64, (3,3), padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(96, (3,3), padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(96, (3,3), padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(128, (3,3),padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(128, (3,3),padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(256, (3,3),padding='same',use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(256, (3,3),padding='same',use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(512, (3,3), padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Convolution2D(512, (3,3), padding='same', use_bias=False, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.LeakyReLU(alpha = 0.1),
    tf.keras.layers.BatchNormalization(),
    

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.Dense(136)                           
])

optimizer = tf.compat.v1.train.AdamOptimizer()
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
print(model.summary())

checkPointDIR = os.path.dirname("training/training/cp.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkPointDIR,
                                                     save_weights_only=True,
                                                     verbose=0)

try:
    model.load_weights('136Points.h5')
    print("MODEL LOADED")
except Exception:
    pass

X = np.load('testX.npy')
Y = np.load('testY.npy')

num = 0 

prediction = model.predict(np.array([X[num]]))
print(prediction)
print(Y[num])

print(X.shape)
plt.scatter(x=prediction[0][0::2], y=prediction[0][1::2], c='r', s=1)
plt.imshow(X[0].reshape(X.shape[1], X.shape[2]), cmap='gray')
plt.waitforbuttonpress()
    