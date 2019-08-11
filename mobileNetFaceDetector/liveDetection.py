import tensorflow as tf 
import matplotlib as plt 
import numpy as np 
import PIL
import cv2 
import sys 
import os 

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 


class faceDetection:
    def __init__(self):
        self.PATH_TO_CKPT = 'D:/Python/mobileNetFaceDetector/inference_graph/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = 'D:/Python/mobileNetFaceDetector/labelmap.pbtxt'
        self.PATH_TO_IMAGE = 'D:/Python/mobileNetFaceDetector/images/train/pic.png'
        
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=1, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        
        self.sess = self.sessionMaker()
        
        self.img_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


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
            img_expand = np.expand_dims(img, axis=0)
                
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
                line_thickness=8,
                min_score_thresh=0.80)
            
            cv2.imshow('Detection', img)
            
            [ymin, xmin, ymax, xmax] = boxes[0][0]
            ymin *= height
            ymax *= height
            xmin *= width
            xmax *= width
            
            ymin, xmin, ymax, xmax = int(ymin) - 20, int(xmin) - 20, int(ymax) + 20, int(xmax) + 20
            face = frame[ ymin:ymax, xmin:xmax, :]
            
            cv2.imshow('face',face)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        camera.release()
        cv2.destroyAllWindows()
            
a = faceDetection()
a.detectLiveVideo()            
    
        


