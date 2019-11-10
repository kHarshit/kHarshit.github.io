---
layout: post
title: "Introduction to Panoptic Segmentation: A Tutorial"
date: 2019-10-18
categories: [Deep Learning, Computer Vision]
---

In semantic segmentation, the goal is to classify each pixel into the given classes. In instance segmentation, we care about segmentation of the instances of objects separately. The panoptic segmentation combines semantic and instance segmentation such that all pixels are assigned a class label and all object instances are uniquely segmented.

*Read about [semantic segmentation]({% post_url 2019-08-09-quick-intro-to-semantic-segmentation %}), and [instance segmentation]({% post_url 2019-08-23-quick-intro-to-instance-segmentation %})*.

<img src="/img/college_semantic.png" style="width: 304px; max-width: 100%">
<img src="/img/college_instance.png" style="width: 304px; max-width: 100%">
<img src="/img/college_panoptic.png" style="width: 304px; max-width: 100%">
<figcaption style="text-align: center;">Left: semantic segmentation, middle: instance segmentation, right: panoptic segmentation</figcaption>

## Introduction

The goal in panoptic segmentation is to perform a unified segmentation task. In order to do so, let's first understand few basic concepts.

A *thing* is a countable object such as people, car, etc, thus it's a category having instance-level annotation. The *stuff* is amorphous region of similar texture such as road, sky, etc, thus it's a category without instance-level annotation. Studying thing comes under object detection and instance segmentation, while studying stuff comes under semantic segmentation.

The label encoding of pixels in panoptic segmentation involves assigning each pixel of an image two labels -- one for semantic label, and other for instance id. The pixels having the same label are considered belonging to the same class, and instance id for stuff is ignored. Unlike instance segmentation, each pixel in panoptic segmentation has only one label corresponding to instance i.e. there are no overlapping instances.

For example, consider the following set of pixel values in a naive encoding manner:

```
26000, 260001, 260002, 260003, 19, 18
```

Here, `pixel // 1000` gives the semantic label, and `pixel % 1000` gives the instance id. Thus, the pixels `26000, 26001, 260002, 26003` corresponds to the same object and represents different instances. And, the pixels `19`, and `18` represents the semantic labels belonging to the non-instance stuff classes.

In COCO, the panoptic annotations are stored in the following way:

> Each annotation struct is a per-image annotation rather than a per-object annotation. Each per-image annotation has two parts: (1) a PNG that stores the class-agnostic image segmentation and (2) a JSON struct that stores the semantic information for each image segment.

{% highlight python %}
annotation{
    "image_id": int,
    "file_name": str,      # per-pixel segment ids are stored as a single PNG at annotation.file_name
    "segments_info": [segment_info],
}

segment_info{
    "id": int,             # unique segment id for each segment whether stuff or thing
    "category_id": int,    # gives the semantic category
    "area": int,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,     # indicates whether segment encompasses a group of objects (relevant for thing categories only).
}

categories[{
    "id": int,
    "name": str,
    "supercategory": str,
    "isthing": 0 or 1,     # stuff or thing
    "color": [R,G,B],
}]
{% endhighlight %}

## Datasets

The available panoptic segmentation datasets include [MS-COCO](http://cocodataset.org/#panoptic-2019), [Cityscapes](https://www.cityscapes-dataset.com/), [Mapillary Vistas](https://research.mapillary.com/eccv18/#panoptic), [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/), and [Indian Driving Dataset](https://idd.insaan.iiit.ac.in/).

## Evaluation

In semantic segmentation, `IoU` and per-pixel accuracy is used as a evaluation criterion. In instance segmentation, average precision over different `IoU` thresholds is used for evaluation. For panoptic segmentation, a combination of `IoU` and `AP` can be used, but it causes asymmetry for classes with or without instance-level annotations. That is why, a new metric that treats all the categories equally, called **Panoptic Quality (`PQ`)**, is used.

*Read more about [evaluation metrics]({% post_url 2019-09-20-evaluation-metrics-for-object-detection-and-segmentation %}).*

As in the calculation of `AP`, `PQ` is also first calculated independently for each class, then averaged over all classes. It involves two steps: matching, and calculation.

Step 1 (matching): The predicted and ground truth segments are considered to be matched if their `IoU > 0.5`. It, with non-overlapping instances property, results in a unique matching i.e. there can be at most one predicted segment corresponding to a ground truth segment.

<img src="/img/pq.png" style="display: block; margin: auto; max-width: 100%;">

Step 2 (calculation): Mathematically, for a ground truth segment `g`, and for predicted segment `p`, PQ is calculated as follows.

$$
\begin{align}
\mathrm{PQ} &= \frac{\sum_{(p, g) \in T P} \operatorname{IoU}(p, g)}{|T P|+\frac{1}{2}|F P|+\frac{1}{2}|F N|}\\

&= \underbrace{\frac{\sum_{(p, g) \in T P} \operatorname{loU}(p, g)}{|T P|}}_{\text {segmentation quality (SQ) }} \times \underbrace{\frac{|T P|}{|T P|+\frac{1}{2}|F P|+\frac{1}{2}|F N|}}_{\text {recognition quality (RQ) }}
\end{align}
$$

Here, in the first equation, the numerator divided by `TP` is simply the average `IoU` of matched segments, and `FP` and `FN` are added to penalize the non-matched segments. As shown in the second equation, `PQ` can divided into segmentation quality (`SQ`), and recognition quality (`RQ`). `SQ`, here, is the average `IoU` of matched segments, and `RQ` is the `F1` score.

## Model

One of the ways to solve the problem of panoptic segmentation is to combine the predictions from semantic and instance segmentation models, e.g. [Fully Convolutional Network (FCN)]({% post_url 2019-08-09-quick-intro-to-semantic-segmentation %}) and [Mask R-CNN]({% post_url 2019-08-23-quick-intro-to-instance-segmentation %}), to get panoptic predictions. In order to do so, the overlapping instance predictions are first need to be converted to non-overlapping ones using a NMS-like (Non-max suppression) procedure.

<img src="/img/fpn_approach.png" style="display: block; margin: auto; max-width: 100%;">

A better way is to use a unified **Panoptic FPN** (Feature Pyramid Network) framework. The idea is to use FPN for multi-level feature extraction as backbone, which is to be used for region-based instance segmentation as in case of Mask R-CNN, and add a parallel dense-prediction branch on top of same FPN features to perform semantic segmentation.

<img src="/img/panoptic_fpn.png" style="display: block; margin: auto; max-width: 100%;">

During training, the instance segmentation branch has three losses $$L_{cls}$$ (classification loss), $$L_{bbox}$$ (bounding-box loss), and $$L_{mask}$$ (mask loss). The semantic segmentation branch has semantic loss, $$L_s$$, computed as the per-pixel cross-entropy between the predicted and the ground truth labels.

$$L = \lambda_i(L_{cls} + L_{bbox} + L_{mask}) + \lambda_s L_s$$

In addition, a weighted combination of the semantic and instance loss is used by adding two tuning parameters $$\lambda_i$$ and $$\lambda_s$$ to get the panoptic loss.

## Implementation

Facebook AI Research recently released [Detectron2](https://github.com/facebookresearch/detectron2) written in PyTorch. In order to test panoptic segmentation using Mask R-CNN FPN, follow the below steps.

{% highlight bash %}
# install pytorch (https://pytorch.org) and opencv
pip install opencv-python
# install dependencies
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# install detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python setup.py build develop

# test on an image (using `MODEL.DEVICE cpu` for inference on CPU)
python demo/demo.py --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml --input ~/Pictures/image.jpg --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl MODEL.DEVICE cpu
{% endhighlight %}

<img src="/img/panoptic_example.png" style="display: block; margin: auto; max-width: 100%;">

**References & Further Readings:**  
1. [Panoptic Segmentation paper](https://arxiv.org/pdf/1801.00868.pdf)  
2. [Panoptic data format](http://cocodataset.org/#format-data)  
3. [Panoptic FPN](https://arxiv.org/pdf/1901.02446.pdf)  
4. [Panoptic segmentation slides (also image source)](https://www.dropbox.com/s/t6tg87t78pdq6v3/cvpr19_tutorial_alexander_kirillov.pdf?dl=0)  
