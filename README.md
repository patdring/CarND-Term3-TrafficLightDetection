# CarND-Term3-TrafficLightDetection
Outsourced  repository for traffic light detection (tests). It's part of the [capstone project](https://github.com/patdring/CarND-Term3-Capstone).

Here the approach of **Transfer Learning** is implemented. The understanding and code (tools to (re-)train, extract and freeze graphs) is based on the great work that can be found at https://github.com/tensorflow/models/tree/master/research/object_detection.
This was done to teach the CNN not only traffic lights, but also their status red, yellow or green:

a) Create a **labeld dataset**. I stored a lot of camera images from simulator and put dem in a dir. structur. You can find them [here](https://github.com/patdring/CarND-Term3-TrafficLightDetection/tree/master/transfer_learning/sim_dataset). Now you have to label them. This can be done with [LabelImg](https://github.com/tzutalin/labelImg). Result is a XML file which contains boxes with coords. of images traffic lights and its color.

b) Create a **TFRecord **file to retrain a TensorFlow model. A TFRecord is a binary file format which stores your images and ground truth annotations. Create also label_map.pbtxt file which contains the labels for red, yellow, green and unknown.

c) Choose a model and configure it. I choose a SSD Inception V2 mode trained against Coco dataset. From TensoFlows repo. you have to copy the according config file. And adjust there some paths and of epochs. The example there uses 200.000 epochs to retrain a network to detect pets. Way too much in my opionion. The simulator generates "perfect" and not so different images. I was statisfied with the already after ~1500 epochs. It's also important to keep in mind to keep your model (file) small.

d) Train and freeze your model. Include it in your project like I did [here] (https://github.com/patdring/CarND-Term3-Capstone/blob/master/ros/src/tl_detector/light_classification/tl_classifier.py)
