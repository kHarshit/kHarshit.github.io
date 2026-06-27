---
layout: post
title: "Loss functions"
date: 2018-08-24
categories: [Data Science, Machine Learning, Deep Learning]
mathjax: true
excerpt: "A survey of common loss functions MSE, cross-entropy, hinge loss, with background on entropy, KL divergence, and the MLE connection."
---

In machine learning, the difference between the predicted output and the actual output is used to tune the parameters of the algorithm. This error in prediction, so called loss, is a crucial part of designing a good model as it evaluates the performance of our model. For accurate predictions, one needs to minimize this loss. In neural networks, it is done using the [gradient descent]({% post_url 2018-03-02-gradient-descent-the-core-of-neural-networks %}). There are many types loss functions. Some of them are:

## L2 loss (Mean-squared error)

Mean squared error is the most common used loss function for regression loss. It minimizes the squared difference between the predicted value and the actual value. The residual sum of squares is defined as:

$$RSS = \sum_{i=1}^n(y_{i} - \hat{y_i})^2$$

{% highlight python %}
import numpy as np

def l2_loss(yhat, y):
    return np.sum((yhat - y)**2))
{% endhighlight %}


## L1 loss (Mean-absolute error)

L1 loss minimized the sum of absolute error between the predicted value and the actual value.

$$L(y, \hat{y}) = \sum_{i=1}^n \left| y_{i} - \hat{y_i} \right|$$

{% highlight python %}
def l1_loss(yhat, y):
    return np.sum(np.absolute(yhat - y))
{% endhighlight %}

*Note:* L1 and L2 are also used in [regularization]({% post_url 2018-01-12-regularization %}). Don't get confused.

## Hinge loss

The hinge loss is used for classification problems e.g. in case of suppport vector machines. It is defined as: 

$$L(y, \hat{y}) = max(0, 1-\hat{y} * y)$$

{% highlight python %}
def hinge(yhat, y):
    return np.max(0, 1 - yhat * y)
{% endhighlight %}

## Cross-entropy (log loss)

### Background: Entropy

Before understanding cross-entropy, it helps to understand **entropy**, a concept from information theory introduced by Claude Shannon. Entropy measures the amount of uncertainty or surprise in a probability distribution. For a discrete random variable X with probability distribution p(x), it is defined as:

$$H(X) = -\sum_{x} p(x) \log p(x)$$

The term $$-\log p(x)$$ represents the "surprise" of observing event x — the less likely an event, the more surprising it is. Entropy is the expected value of this surprise. It is high when probabilities are nearly uniform (high uncertainty) and low when the distribution is skewed (low uncertainty). For example, a fair coin has entropy of 1 bit, while a biased coin that always lands heads has zero entropy.

Cross-entropy extends this concept to measure the difference between two distributions.

### Definition

The cross-entropy loss is used in case of classification problems for estimating the accuracy of model whose output is a probability, `p`, which lies between 0 and 1. In case of binary classification, it can be written as:

$$L(y) = -{(y\log(p) + (1 - y)\log(1 - p))}$$

For multi-class classification problems, it is:

$$L(y) = -\sum_{c=1}^n y_{o,c}\log(p_{o,c})$$

Here, `y` is a binary indicator (0 or 1) if class label `c` is the correct classification for observation `o` and `p` is predicted probability observation s.t. `o` is of class `c`.

<img src="/img/blog/loss-functions/log_loss.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

{% highlight python %}
def cross_entropy(y, p):
    return -np.sum(np.multiply(y, np.log(p)) + np.multiply((1-y), np.log(1-p)))
{% endhighlight %}

### Relationship to KL Divergence

Cross-entropy is closely related to the **Kullback-Leibler (KL) divergence**, which quantifies the information lost when distribution Q is used to approximate another distribution P:

$$D_{KL}(P \parallel Q) = H(P, Q) - H(P)$$

where $$H(P, Q)$$ is the cross-entropy and $$H(P)$$ is the entropy of the true distribution. Since $$H(P)$$ is constant for a fixed dataset, **minimizing cross-entropy is equivalent to minimizing the KL divergence** between the true and predicted distributions. KL divergence is non-negative and non-symmetric; it equals zero only when $$P = Q$$.

### Connection to Maximum Likelihood Estimation

Minimizing cross-entropy is equivalent to maximizing the log-likelihood of the data under a Bernoulli (binary) or multinomial (multi-class) model. The negative log-likelihood for a Bernoulli distribution is exactly the binary cross-entropy loss:

$$\theta_{MLE} = \arg\max_{\theta} \log L(\theta \mid x) \equiv \arg\min_{\theta} \text{CE Loss}$$

This gives cross-entropy a solid statistical foundation. It is the loss function that yields the **Maximum Likelihood Estimate** of the model parameters.

For a detailed example using *log loss*, check logistic regression implementation on GitHub: [kHarshit/ML-py](https://github.com/kHarshit/ML-py/blob/master/logistic_regression.ipynb).

## Conclusion

The choice of loss function depends on the class of problem (regression / classification) as well as is sometimes specific to the problem.

**References:**

1. [Loss functions - ML cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
2. [Differences between L1 and L2 as Loss Function and Regularization](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)
3. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function)
