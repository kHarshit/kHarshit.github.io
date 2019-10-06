---
layout: post
title: "Converting FC layers to CONV layers"
date: 2019-08-02
categories: [Deep Learning, Computer Vision]
---

> It is worth noting that the only difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters.

Suppose, the 7x7x512 activation volume output of the conv layer is fed into a 4096 sized fc layer. This fc layer can be replaced with a conv layer having 4096 filters (kernel) of size 7x7x512, where each filter gives 1x1x1 output  which are concatenated to give output of 1x1x4096, which is equal to what we get in fc layer.

As a general rule, replace `K` sized fc layer *with* a conv layer having `K` number of filters of the same size that is input to the fc layer.  
For example, if a `conv1` layer outputs `HxWxC` volume, and it's fed to a `K` sized `fc` layer. Then, the `fc` layer can be replaced with a `conv2` layer having `K HxW` filters. In PyTorch, it'd be 

{% highlight python %}
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
{% endhighlight %}

Before:  
*`nn.Conv2d(...)`*  
image dim: 7x7x512  
*`nn.Linear(512 * 7 * 7, 4096)`*  
*`nn.Linear(4096, 1000)`*

After:  
*`nn.Conv2d(...)`*  
image dim: 7x7x512  
*`nn.Conv2d(512, 4096, 7)`*  
image dim: 1x1x4096  
*`nn.Conv2d(4096, 1000, 1)`*  
image dim: 1x1x1000

Using the above reasoning, you'd notice that all the further fc layers, *except the first one*, will require `1x1` convolutions as shown in the above example, it's because after the first conv layer, the feature maps are of size `1x1xC` where `C` is the number of channels.


**References:**  
1. [CS231n](http://cs231n.github.io/convolutional-networks/#convert)  