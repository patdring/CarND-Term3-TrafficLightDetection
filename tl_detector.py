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
from glob import glob
import cv2

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def tl_detector(PATH_TO_TEST_IMAGES_DIR):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = 'model/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS =  os.path.join('data', 'label_map.pbtxt')

    # number of classes for COCO dataset
    NUM_CLASSES = 4

    #--------Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    #----------Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print(category_index)

    PATH_TO_IMGS = r'test_images_input'
    TEST_IMGS = glob(os.path.join(PATH_TO_IMGS, r'*.jpg'))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detect_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detect_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detect_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            for idx, img_path in enumerate(TEST_IMGS):
                image = Image.open(img_path)
                image_np = load_image_into_numpy_array(image)
                image_expanded = np.expand_dims(image_np, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                    [detect_boxes, detect_scores, detect_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})
                
                print('---------------')
                print('SCORES')
                print(scores[0][0])
                print('CLASSES')
                print(classes[0][0])
                
                if (scores[0][0] >= 0.5):
                    if(classes[0][0] == 1):
                        print('RED')
                    if(classes[0][0] == 2):
                        print('YELLOW')
                    if(classes[0][0] == 3):
                        print('GREEN')
                    if(classes[0][0] == 4):
                        print('UNKNOWN')
                else:
                    print('UNKNOWN')
                print('---------------')
                
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np, 
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=3,
                    line_thickness=8)

                result = Image.fromarray(image_np.astype(np.uint8))
                result.save('./test_images_output/img_' + str(idx) +'.jpg')
               

if __name__ == "__main__":
    PATH_TO_TEST_IMAGES_DIR = './test_images_input'
    tl_detector(PATH_TO_TEST_IMAGES_DIR)
    print("Finished traffic light detection on test images!!!")
