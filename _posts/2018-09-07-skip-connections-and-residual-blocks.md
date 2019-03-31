---
layout: post
title: "Skip connections and Residual blocks"
date: 2018-09-07
categories: [Deep Learning, Computer Vision]
---

Deep neural networks are difficult to train. They also have vanishing or exploding gradient problems. Batch normalization helps, but with the increase in depth, the network has trouble reaching convergence.

> When deeper networks are able to start converging, a *degradation problem* has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.

One solution to this problem was proposed by Kaiming He et. al in *Deep Residual Learning for Image Recognition*<sup id="a1">[1](#myfootnote1)</sup> to use Resnet blocks, which connect the output of one layer with the input of an earlier layer. These skip connections are implemented as follows.

<img src="/img/resnet_block.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

Usually, a deep learning model learns the mapping, M, from an input x to an output y i.e.

$$M(x) = y$$ 

Instead of learning a direct mapping, the residual function uses the difference between a mapping applied to x and the original input, x i.e.

$$F(x) = M(x) - x$$

The skip layer connection is used i.e. 

$$M(x) = F(x) + x$$ 

$$\text{or}$$ 

$$y = F(x) + x$$ 

It is easier to optimize this residual function F(x) compared to the original mapping M(x).

The Microsoft Research team won the ImageNet 2015 competition using these deep residual layers, which use skip connections. They used ResNet-152 convolutional neural network architecture, which consists of 152 layers.


<img src="/img/resnet_50.png" style="display: block; margin: auto; width: auto; max-width: 100%;">


**References:**  
<a name="myfootnote1"></a>1. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) [↩](#a1)   
<a name="myfootnote2"></a>2. [Neural network with skip-layer connections](https://stats.stackexchange.com/questions/56950/neural-network-with-skip-layer-connections)  
