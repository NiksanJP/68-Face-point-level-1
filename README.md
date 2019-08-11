# 68-Face-point-level-1
Detect 68 points on face live. This is running two neural networks which are very light as it runs on 30% of Intel i7-7700k CPU. Please run liveDetection.py for live detection on face.

Hire me: https://www.linkedin.com/in/nikheil-malakar-20a2b7166/

# Before you start
This project is still in development stage which will have level 2 and level 3 keypoint detection because the person's face maybe far or
close, so the full version should be out soon.

# How it Works
This has two neural networks working at the same time running on high FPS. The first neural network is Tensorflow mobile net ssd which detects theface.
Another neural network takes in the picture of the face detected and detects the face keypoints

# Face Detection Neural Network - NN1
This can be downloaded via: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# Face Keypoint Neural Network - NN2
This Neural Network takes in an image of a face and turns it into a 100X100 grayscale image and then detects it. Therefore when the person is
too close the points are not that perfect. Updates should be made on the level-2 version. The model is

```
self.model = tf.keras.models.Sequential([
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
```

# Data Sets
It is part of the challenge for ibug: https://ibug.doc.ic.ac.uk/resources/300-W/
