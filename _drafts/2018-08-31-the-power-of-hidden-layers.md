---
layout: post
title: "The power of hidden layers"
date: 2018-08-31
categories: [Deep Learning]
---

As discussed in an earlier [post]({% post_url 2018-04-27-how-deep-should-neual-nets-be %}), you should use as many hidden layers to your network as required to get optimum performance and use regularization techniques to avoid overfitting. But, why adding more layers generally help? What exactly do those hidden layers do? What's hidden in them?


Optimization algorithms that use only a single example at a time are sometimes called stochastic, as you mentioned. Optimization algorithms that use the entire training set are called batch or deterministic gradient methods.

Most algorithms used for deep learning fall somewhere in between, using more than one but fewer than all the training examples. These were traditionally called minibatch or minibatch stochastic methods, and it is now common to call them simply stochastic methods.



*Adam* is a _second order method_, which generally perform better than _first order methods_ like *SGD*, but takes more computation power


**References:**  

1. [What's hidden in the Hidden Layers?](https://www.cs.cmu.edu/~dst/pubs/byte-hiddenlayer-1989.pdf)  
