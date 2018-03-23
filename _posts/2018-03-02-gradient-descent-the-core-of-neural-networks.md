---
layout: post
title: "Gradient descent: The core of neural networks"
date: 2018-03-02
categories: [Data Science, Deep Learning]
---

As discussed in the post [linear algebra and deep learning]({% post_url 2018-02-16-linear-algebra-the-essence-behind-deep-learning %}), the **optimization** is the *third* and last step in solving image classification problem in deep learning. It helps us find the values of weights `W` and bias `b` that minimizes the loss function.


<img src="/img/gradient_descent_demystified.png" style="float: right; display: block; margin: auto; width: auto; max-width: 100%;">

<h3>The strategy</h3>
We use the concept of *following the gradient* to find the optimal pairs of `W` and `b` i.e. we can compute the best direction along which we should change our weight vectors that is mathematically guaranteed to be the direction of the steepest descent. This direction is found out by the <abbr title="gradient is a vector of derivatives of multi-variable functions i.e. vector of partial derivates in each dimension">*gradient*</abbr> of the loss function.



<h3>Computation</h3>

There are two ways to compute the gradient:  
* **Numerical gradient:**  The numerical gradient is very simple to compute using the finite difference approximation, but the downside is that it is approximate and that it is very computationally expensive to compute. We use the following expression to calculate numerical gradient:

$$\displaystyle{\frac{df(x)}{dx} = \lim \limits_{h \to 0} \frac{f(x + h) - f(x)}{h}}$$


* **Analytical gradient:**  The second way to compute the gradient is analytically using Calculus, which allows us to derive a direct formula for the gradient (no approximations) that is also very fast to compute.

Unlike the numerical gradient it can be more error prone to implement, which is why in practice it is very common to compute the analytic gradient and compare it to the numerical gradient to check the correctness of your implementation. This is called a *gradient check*.

Now that we can compute the gradient of the loss function, the procedure of repeatedly evaluating the gradient and then performing a parameter update is called **gradient descent**. 

{% highlight python %}
# basic gradient descent
while True:
    weights_grad = evaluate_gradient(loss_fn, data, weights)  # gradient calculation
    weights += -step_size * weights_grad  # parameter update
{% endhighlight %}

This simple loop is at the core of all Neural Network libraries.

<h3>Mini-batch gradient descent</h3>

When the training data is large, it's computationally expensive and wasteful to compute the full loss function over the entire training set to perform a single parameter update. Thus, a simple solution of using the <abbr title="a typical batch contains 256 examples from the entire training set of 1.2 million">batches</abbr> of training data is implemented. This batch is then used to perform parameter update.

{% highlight python %}
# mini-batch gradient descent
while True:
    data_batch = sample_training_data(data, 256)  # sample training 256 examples
    weights_grad = evaluate_gradient(loss_fn, data_batch, weights)
    weights += -step_size * weights_grad
{% endhighlight %}

The extreme case of this is a setting where the mini-batch contains only a single example. This process is called *Stochastic Gradient Descent (SGD)* (or *on-line gradient descent*).

<h3>Conclusion</h3>

Gradient descent is one of many optimization methods, namely first order optimizer, meaning, that it is based on analysis of the gradient of the objective. In order to calculate gradients efficiently, we use backpropagation. Consequently, in terms of neural networks it is often applied together with backprop to make an efficient updates.


**Read more:**  
[Understanding gradient descent algorithm](https://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html)  

**References:**  
[image source](http://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
