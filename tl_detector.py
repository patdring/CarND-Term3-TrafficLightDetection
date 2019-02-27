# This codes bases on https://pythonprogramming.net/tensorflow-object-detection-api-self-driving-car/ resp. https://github.com/tensorflow/models/tree/master/research/object_detection

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from os import path
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
import cv2


MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'

# What model to download
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS =  os.path.join('data', 'mscoco_label_map.pbtxt')

# number of classes for COCO dataset
NUM_CLASSES = 90

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = None
        self.label_map = None
        self.categories = None
        self.category_index = None
                
        # Download model
        if path.isdir(MODEL_NAME) is False:
            opener = urllib.request.URLopener()
            opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
            tar_file = tarfile.open(MODEL_FILE)
            for file in tar_file.getmembers():
              file_name = os.path.basename(file.name)
              if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                    max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor =self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = self.load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                #vis_util.visualize_boxes_and_labels_on_image_array(
                #        image_np,
                #        np.squeeze(boxes),
                #        np.squeeze(classes).astype(np.int32),
                #        np.squeeze(scores),
                #        self.category_index,
                #        use_normalized_coordinates=True,
                #        line_thickness=2)
                
                #result = Image.fromarray(image_np.astype(np.uint8))

                w, h = image.size

                np_boxes = np.squeeze(boxes) 
                np_scores = np.squeeze(scores) 
                np_classes = np.squeeze(classes).astype(np.int32)

                redFlag = False

                for i in range(min(5, np_boxes.shape[0])):
                   if np_classes[i] == 10 and np_scores[i] > 0.75:
                       yMin, xMin, yMax, xMax = tuple(np_boxes[i].tolist())
                       (left, right, top, bottom) = (xMin*w, xMax*w, yMin*h, yMax*h)
                       crop_img = image.crop((left, top, right, bottom)) 
                       
                       w_c, h_c = (40, 100) 
                       total_pixels = w_c * h_c
                       res = cv2.resize(np.array(crop_img), (w_c, h_c), interpolation=cv2.INTER_LINEAR)
                       hsv=cv2.cvtColor(res, cv2.COLOR_RGB2HSV)

                       lower_range_0 = np.array([0,75,55])
                       upper_range_0 = np.array([10,255,255])
                       mask_0 = cv2.inRange(hsv, lower_range_0, upper_range_0)

                       lower_range_1 = np.array([160,75,55])
                       upper_range_1 = np.array([180,255,255])
                       mask_1 = cv2.inRange(hsv, lower_range_1, upper_range_1)

                       mask = mask_0+mask_1

                       ratio =  np.sum(mask != 0) / total_pixels

                       if ratio > 0.05:
                           #save_at_template = os.path.join("./test_images_output", "image"+str(i)+"_cropped_{:02}_stop.png")
                           redFlag = True        
                       #crop_img.save(save_at_template.format(i))
                #result.save('./test_images_output/image' + str(idx) +'.png')
        return redFlag

if __name__ == "__main__":

    Num_images = 17
    PATH_TO_TEST_IMAGES_DIR = './test_images_input'
    # path to test images
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'img_{}.jpg'.format(i)) for i in range(1, Num_images+1) ]

    tlc = TLClassifier()    
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path) 
        result = tlc.get_classification(image)
        output_msg = image_path+" "+str(result)
        print(output_msg)

