import tensorflow as tf 
import matplotlib as plt 
import numpy as np 
import PIL
import cv2 
import sys 
import os 
import time

from tensorflow.keras.regularizers import l2
from skimage.transform import resize  
from skimage.color import rgb2gray

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 


class faceDetection:
    def __init__(self):
        self.PATH_TO_CKPT = 'D:/Python/facialPointv2/mobileNetFaceDetector/inference_graph/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = 'D:/Python/facialPointv2/mobileNetFaceDetector/labelmap.pbtxt'
        self.PATH_TO_IMAGE = 'D:/Python/facialPointv2/mobileNetFaceDetector/pic.png'
        
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=1, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        
        self.sess = self.sessionMaker()
        
        self.img_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
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
        
        self.model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='mse', metrics=['mae'])
        print(self.model.summary())
        self.model.load_weights('training/training')
        print("MODEL LOADED")
    
    def sessionMaker(self):
        self.detection_graph = tf.compat.v1.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.compat.v1.Session(graph=self.detection_graph)
        return sess 
    
    def detectLiveVideo(self):
        camera = cv2.VideoCapture(0)
        width = camera.get(3)
        height = camera.get(4)
        
        while True:
            ret, frame = camera.read()
            
            cv2.imwrite(self.PATH_TO_IMAGE, frame)
            img = cv2.imread(self.PATH_TO_IMAGE)
            img_expand = np.expand_dims(frame, axis=0)
                
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.img_tensor: img_expand})
                
            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=1,
                min_score_thresh=0.85)
            
            [ymin, xmin, ymax, xmax] = boxes[0][0]
            ymin *= height
            ymax *= height
            xmin *= width
            xmax *= width
            
            ymin, xmin, ymax, xmax = int(ymin) - 20, int(xmin) - 20, int(ymax) + 20, int(xmax) + 20
            face = frame[ ymin:ymax, xmin:xmax, :]
            
            maskface = resize(face, (100,100))
            maskface = rgb2gray(maskface)
            maskface = maskface.reshape(maskface.shape[0], maskface.shape[1], 1)
            n = np.array([maskface])
            
            prediction = self.model.predict(n)
            
            nX = face.shape[0]/100
            nY = face.shape[1]/100
            
            prediction[0][0::2] = (prediction[0][0::2] * nY) + xmin
            prediction[0][1::2] = (prediction[0][1::2] * nX) + ymin
                   
            for i in range(0,136,2):
                frame = cv2.circle(frame, (prediction[0][i], prediction[0][i+1]), 2, (0,0,255), 1)
 
            cv2.imshow('Detection', frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        camera.release()
        cv2.destroyAllWindows()
            
a = faceDetection()
a.detectLiveVideo()            
    
        


