---
layout: post
title: "The gradient problem in RNN"
date: 2019-01-04
categories: [Deep Learning, Natural Language Processing]
---

Much as Convolutional Neural Networks are used to deal with images, the Recurrent Neural Networks are used to process sequential data. The key idea in recurrent neural networks is *parameter sharing* across different parts of the model i.e. a RNN shares same weights across several time steps. As feedforward neural network can model any function, a recurrent neural network can model any function involving recurrence.

## RNN

The vanilla Recurrent Neural Network (RNN) has the following structure: 

$$h_t = f_W(h_{t-1}, x_t)$$

Here, the same function and same set of parameters are used at every time step. The $$h_t$$ denotes the *state*, which is a kind of summary of past sequence of inputs $$(x_t, x_{t-1}, x_{t-2}, ..., x_1)$$ upto $$t$$ mapped to fixed length vector $$h_t$$.

But, the gradient flow in RNNs often lead to the following problems:

* Exploding gradients
* Vanishing gradients  

<img src="/img/gradient_flow_rnn.png" style="display: block; margin: auto; width: auto; max-width: 100%;">


The gradient computation involves recurrent multiplication of $$W$$. This multiplying by $$W$$ to each cell has a bad effect. *Think like this:* If you a scalar (number) and you multiply gradients by it over and over again for say 100 times, if that number > 1, it'll explode the gradient and if < 1, it'll vanish towards 0.

The problem of exploding gradients can be solved by *gradient clipping* i.e. if gradient is larger than the threshold, scale it by dividing. LSTM solves the problem of vanishing gradients.

## LSTM

A Long Short Term Memory (LSTM) utilizes four gates that perform a specific function.

<img src="/img/lstm.png" style="display: block; margin: auto; width: auto; max-width: 100%;">


* Input gate ($$i$$): controls what to write to the LSTM cell
* Forget gate ($$f$$): controls whether to erase the cell
* Output gate ($$o$$): controls how much to reveal cell
* The fourth gate ($$g$$) controls how much to write to the cell

<img src="/img/gradient_flow_lstm.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

In LSTM, the backpropagation only involves element-wise multiplication by $$f$$, and no matrix multiplication by $$W$$ i.e. the cell state gradient is multiplied only by the forget gate element-wise (better than full matrix multiplication). Also, in vanilla RNN, we're multiplying by same $$W$$ over and over again leading to exploding or vanishing gradient problems. But, here this forget gate will vary at each time-step (e.g. values sometimes > 1 or < 1) thus avoiding these problems.

**References:**  
1. [CS231n: Convolutional Neural Networks for Visual Recognition (also image source)](http://cs231n.stanford.edu/)
