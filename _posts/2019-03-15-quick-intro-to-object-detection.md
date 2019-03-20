---
layout: post
title: "Quick intro to Object detection"
date: 2019-03-15
categories: [Deep Learning, Computer Vision]
---

Object detection deals with the detection of object instances in an image. There are a number of methods to accomplish it. The following post summarizes few important object detection methods.

<a href="https://commons.wikimedia.org/wiki/File:Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg#/media/File:Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg"><img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg" alt="Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg" width="640" height="412" style="display: block; margin: auto; width: auto; max-width: 100%;"></a>

## Classification and Localization

This method treats object detection as a regression problem. Given a number of objects in an image, localization deals with classification of all the objects in the image by drawing a bounding box by finding its location in the image.

The loss may consists of softmax loss for classification and L2 regression loss for bounding box. The problem with this approach is that an image may contain different number of objects thus each image need different number of outputs, which creates a problem.

## Sliding window

This method treats object detection as a classification problem. The sliding window deals with sliding the window through the image and passing the cropped image to a convolutional neural network and classifying it as object or background.

The problem with this approach is that it's needed to apply CNN to a huge number of windows.

## Region proposals

The region proposal methods deals with using selective search to propose regions (boxes) that likely contain objects thus avoiding the computationally expensive method of appplying CNN to every possible window in the image.

The various region based methods i.e. region-proposals are:  

* **R-CNN:** First, the Region of Interest (ROI) is suggested by a region proposal method. These regions are then fed into CNN and support vector machines is used to classify them.
* **Fast R-CNN:** Instead of passing each region through CNN, in Fast R-CNN the entire image is passed once generating convolutional feature maps, using which the regions are proposed. The ROI pooling layer is then used to convert these regions into a fixed size, finally feeding it into a fully connected network.
* **Faster R-CNN:** Unlike Fast R-CNN which uses selective search for ROI, Faster R-CNN uses Region Proposal Network (RPN) to predict proposals from features.
* **Mask R-CNN:** It extends Faster R-CNN. Mask R-CNN is used for instance segmentation which not only does object detection but also predicts object masks.

## Detection without proposals

There are other object detection methods that use detection without proposals.

* YOLO (You Only Look Once)
* SSD (Single Shot Detection)


**Further Readings:**  
1. [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) 
2. [A Step-by-Step Introduction to the Basic Object Detection Algorithms](https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/)  
3. [Introduction to Object Detection](https://www.hackerearth.com/blog/machine-learning/introduction-to-object-detection/)
