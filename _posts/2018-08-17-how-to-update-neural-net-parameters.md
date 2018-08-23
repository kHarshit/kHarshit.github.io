---
layout: post
title: "How to update Neural Net parameters"
date: 2018-08-17
categories: [Data Science, Deep Learning]
---

Building neural networks, we compute [gradients]({% post_url 2018-03-02-gradient-descent-the-core-of-neural-networks %}) with the [backpropagation]({% post_url 2018-03-09-computational-graphs-backpropagation %}) algorithm. These gradients are used to perform the parameter updates. The default way of doing this is to update gradient along the negative gradient direction.

{% highlight python %}
# SGD
while True:
    data_batch = sample_training_data(data, 1)  # sample training 1 example
    # gradient calculation
    weights_grad = evaluate_gradient(loss_fn, data_batch, weights)  
    # parameter update
    weights += -learning_rate * weights_grad  
{% endhighlight %}

<img src="/img/local_minima.png" style="float: right; display: block; margin: auto; width: auto; max-width: 100%;">

There are certain problems with SGD. If the loss function has a local minima or a saddle point, the gradient becomes zero (gradient descent get stuck).

But, there are many more methods to improve on this. Few of them are:

## Momentum update

Here, the gradient instead of directly influencing the position, it influences the velocity, which in turn has an effect on the position. The momentum update builds up the velocity in directions having a gentle but consistent gradient.

{% highlight python %}
# momentum update
velocity += mu * velocity - learning_rate * weights_grad  # integrate velocity
weights += velocity  # integrate position
{% endhighlight %}

Here, the effect of gradient is to increment the previous velocity. `velocity` is initialized to zero and value of `mu` is the rate by which `velocity` decays, typically taken slightly less than 1. The momentum method has better convergence rate than the basic update.

## Nesterov momentum

<img src="/img/nesterov_momentum.png" style="display: block; margin: auto; width: auto; max-width: 100%;">
<figcaption>
    <i style="color:blue;">blue vectors: standard momentum</i>,
    <i style="color:brown;">brown vectors: jump</i>,
    <i style="color: red">red vectors: correction</i>,
    <i style="color:green;">green vectors: accumulated gradient</i>
</figcaption>

The standard momentum method first computes the gradient at the current position then takes a big jump in the direction of the updated accumulated gradient. While, in Nesterov momentum, first make a big jump in the direction of the previous accumulated gradient then measure the gradient where you end up and make a correction. It has better convergence than the standard momentum.

{% highlight python %}
# momentum update
v_prev = v # back this up
v = mu * v - learning_rate * dx # velocity update stays the same # dx is gradient
x += -mu * v_prev + (1 + mu) * v # position update changes
{% endhighlight %}

## Adagrad

All the above methods manipulate the learning rate globally and equally for all the parameters. Adagrad is an adaptive learning rate method that adaptively tune the learning rates per parameter.

{% highlight python %}
# adagrad
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
{% endhighlight %}
`cache` keep track of per-parameter sum of squared gradients. The learning rate will reduce for the weights having high gradients and increase for the weights having small updates.

## RMSprop

RMSProp keeps a moving average of the root mean squared (rms) gradients, by which we divide the current gradient. This makes the learning process much better.

{% highlight python %}
# rmsprop
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
{% endhighlight %}

## Adam

Adam is almost similar to RMSProp. Instead of adapting the parameter learning rate based only on the first moment (mean) as in RMSProp, Adam also uses the second moment of the gradients (variance). That is, it calculates the exponential average of the gradient (*i.e. mean) and the squared gradient (*i.e. variance*). The Adam upate also includes the *bias correction* to compensate the bias at zero of `m` and `v` during initialization.

{% highlight python %}
# t is your iteration counter going from 1 to infinity
m = beta1*m + (1-beta1)*dx
mt = m / (1-beta1**t)
v = beta2*v + (1-beta2)*(dx**2)
vt = v / (1-beta2**t)
x += - learning_rate * mt / (np.sqrt(vt) + eps)
{% endhighlight %}

## Conclusion

Adam is recommended the default choice of optimization algorithm. However, it's also worth trying RMSProp and SGD+Nesterov momentum.


**References:**  
1. [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-3/#update)  
2. [Neural Networks for Machine Learning](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
