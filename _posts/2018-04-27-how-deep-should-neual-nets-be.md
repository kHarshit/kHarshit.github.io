---
layout: post
title: "How deep should neural nets be?"
date: 2018-04-27
categories: [Data Science, Deep Learning]
---

How do we decide on what architecture to use while solving a problem using neural networks? Should we use no hidden layers? One hidden layer? Two hidden layers? How large should each layer be?


## Types of layers

There are three types of layers in neural networks.

### Input layer

The input layer contains neurons equal to number of features.

### Output layer

There is one output layer. Most of the time, it has only one neuron.

<img src="/img/deep_neural_net.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

### Hidden layer

In most of the problems, one hidden layer would suffice. In practice, it's often the case that 3 layer hidden network would outperform 2 layer network but going deeper rarely helps.

Neural networks with more layers (simply, more neurons) can express more complicated functions. They will always work better than smaller networks, but they might overfit -- the network won't generalize well. The [overfitting]({% post_url 2017-10-13-overfitting-and-underfitting %}) can be controlled by [regularization]({% post_url 2018-01-12-regularization %}) such as L2 and dropout regularization.

> Add layers until you start to overfit your training set. Then you add dropout or another regularization method.
> &mdash; <cite>Geoff Hinton</cite>

What about the size (number of neurons) of hidden layer? The optimal size of the hidden layer is usually between the size of the input and size of the output layers.

We can also follow [Occam's razor]({% post_url 2017-12-22-simplicity-doesn't-imply-accuracy %}) that simple is better than complex. There are five approaches that people use to build *simple* neural networks.

* *Growing:* start with no neurons in the network and then add neurons until the performance is adequate.
* *Pruning:* start with large networks, which likely overfit, and then remove neurons (or weights) one at a time until the performance degrades significantly.
* *Global searches:* search the space of all possible network architectures to locate the simplest model that explains the data.
* *Regularization:* keep the network small by constraining the magnitude of the network weights, rather than by constraining the number of network weights.
* *Early stopping:* does same as regularization.

## Conclusion

The takeaway is that you should not be using smaller networks because you are afraid of overfitting. Instead, you should use a big neural network, and use regularization techniques to control overfitting. The reason being the smaller networks are harder to train with local methods such as gradient descent.

Start with one hidden layer and keep adding until you get the desired performance. Use regularization and cross-validation.

**References:**

1. [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/)
2. [Neural Network Design 2nd Edition](http://hagan.okstate.edu/NNDesign.pdf#page=469)
3. [Choosing no of hidden layers and nodes - Cross Validated](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
