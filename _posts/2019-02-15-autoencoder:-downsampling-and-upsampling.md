---
layout: post
title: "Autoencoder: Downsampling and Upsampling"
date: 2019-02-15
categories: [Deep Learning]
---

An autoencoder is a neural network that learns data representations in an unsupervised manner. Its structure consists of *Encoder*, which learn the compact representation of input data, and *Decoder*, which decompresses it to reconstruct the input data. A similar concept is used in [generative models]({% post_url 2018-09-28-generative-models-and-generative-adversarial-networks %}).

For example, in case of MNIST dataset,

<img src="/img/autoencoder_1.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

## Convolutional autoencoder

In Convolutional autoencoder, the Encoder consists of convolutional layers and pooling layers, which downsamples the input image. The Decoder upsamples the image. The structure of convolutional autoencoder looks like this:

<img src="/img/autoencoder_3.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

### Downsampling

The normal convolution operation gives the same size output image as input image e.g. 3x3 convolution with stride 1 and padding 1.

<img src="/img/downsampling1.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

But strided convolution results in downsampling i.e. reduction in size of input image e.g. 3x3 convoltional with stride 2 and padding 1 convert image of size 4x4 to 2x2.

<img src="/img/downsampling.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

### Upsampling

One of the ways to upsample the compressed image is by Unpooling *(the reverse of pooling)* using Nearest Neighbor or by max unpooling.

<img src="/img/upsampling1.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

Another way is to use transpose convolution. The convolution operation with strides results in downsampling. The transpose convolution is reverse of the convolution operation. Here, the kernel is placed over the input image pixels. The pixel values are multiplied successively by the kernel weights to produce the upsampled image. In case of overlapping, the values are summed. The kernel weights in upsampling are learned the same way as in convolutional operation that's why it's also called learnable upsampling.

<img src="/img/upsampling2.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

One other way is to use nearest-neighbor upsampling and convolutional layers in Decoder instead of transpose convolutional layers. This method prevents checkerboard artifacts in the images, caused by transpose convolution.

## Denoising autoencoders

The denoising autoencoder recovers de-noised images from the noised input images. It utilizes the fact that the higher-level feature representations of image are relatively stable and robust to the corruption of the input. During training, the goal is to reduce the regression loss between pixels of original un-noised images and that of de-noised images produced by the autoencoder.

<img src="/img/autoencoder_denoise.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

There are many other types of autoencoders such as Variational autoencoder (VAE).

**References:**  
1. [Autoencoder - Wikipedia](https://en.wikipedia.org/wiki/Autoencoder)  
2. [PyTorch Deep Learning Nanodegree - Udacity (also image source)](https://in.udacity.com/course/deep-learning-nanodegree--nd101)  
3. [CS231n (also image source)](http://cs231n.stanford.edu/) 
4. [Deconvolution and Checkerboard Artifacts - Distill](https://distill.pub/2016/deconv-checkerboard/)  