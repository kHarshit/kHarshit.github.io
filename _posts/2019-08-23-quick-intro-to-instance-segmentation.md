---
layout: post
title: "Quick intro to Instance segmentation: Mask RCNN"
date: 2019-08-23
categories: [Deep Learning, Computer Vision]
---

*This is the third post in the Quick intro series: [object detection (I)]({% post_url 2019-03-15-quick-intro-to-object-detection %}), [semantic segmentation (II)]({% post_url 2019-08-09-quick-intro-to-semantic-segmentation %})*.

The instance segmentation combines *object detection*, where the goal is to classify individual objects and localize them using a bounding box, and *semantic segmentation*, where the goal is to classify each pixel into the given classes. In instance segmentation, we care about detection and segmentation of the instances of objects separately.

<img src="/img/segmentation.png" style="display: block; margin: auto; width: 90%; max-width: 100%;">

## Mask R-CNN

Mask R-CNN is a state-of-the-art model for instance segmentation. It extends Faster R-CNN, the model used for object detection, by adding a parallel branch for predicting segmentation masks.

<img src="/img/seg_mask_rcnn.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

[Source](http://lernapparat.de/static/artikel/pytorch-jit-android/thomas_viehmann.pytorch_jit_android_2018-12-11.pdf)

The following Mask R-CNN implementation is from `facebookresearch/maskrcnn-benchmark`<sup id="a1">[1](#myfootnote1)</sup>. First, install it as follows.

{% highlight python %}
# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ../../

# install apex
rm -rf apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git pull
python setup.py install --cuda_ext --cpp_ext
cd ../

# install PyTorch Detection
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with symbolic links, so that you can modify
# the files if you want and won't need to re-build it
python setup.py build develop

# download predictor.py, which contains necessary utility functions
wget https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/demo/predictor.py

# download configuration file
wget https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml
{% endhighlight %}

Here, for inference, we'll use model pretrained on MS COCO dataset.

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
from PIL import Image
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)

# a helper class `COCODemo`, which loads a model from the config file, and performs pre-processing, model prediction and post-processing for us
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

pil_image = Image.open('cats.jpg').convert("RGB")
# convert to BGR format
image = np.array(pil_image)[:, :, [2, 1, 0]]

# compute predictions
predictions = coco_demo.run_on_opencv_image(image)

# plot
f, ax = plt.subplots(1, 2, figsize=(15, 4))
ax[0].set_title('input image')
ax[0].axis('off')
ax[0].imshow(pil_image)
ax[1].set_title('segmented output')
ax[1].axis('off')
ax[1].imshow(predictions[:, :, [2, 1, 0]])
plt.savefig("segmented_output.png", bbox_inches='tight')
{% endhighlight %}


<img src="/img/segmentation_cat_output_instance.png" style="display: block; margin: auto; width: auto; max-width: 100%;">


**References:**  
<a></a>1: [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)  
<a name="myfootnote1"></a>2: [facebookresearch/maskrcnn-benchmark - Faster R-CNN and Mask R-CNN in PyTorch 1.0](https://github.com/facebookresearch/maskrcnn-benchmark) [↩](#a1)  
<a></a>3: 