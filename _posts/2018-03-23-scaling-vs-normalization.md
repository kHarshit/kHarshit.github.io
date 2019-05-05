---
layout: post
title: "Scaling vs Normalization"
date: 2018-03-23
categories: [Data Science, Machine Learning]
---

**Feature scaling** (also known as data normalization) is the method used to standardize the range of features of data. Since, the range of values of data may vary widely, it becomes a necessary step in data preprocessing while using machine learning algorithms.


## Scaling

In scaling, you transform the data such that the features are within a specific range e.g. [0, 1].

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

where x' is the normalized value.

Scaling is important in the algorithms such as support vector machines (SVM) and k-nearest neighbors (KNN) where distance between the data points is important. For example, in the dataset containing prices of products; without scaling, SVM might treat 1 USD equivalent to 1 INR though 1 USD = 65 INR.

{% highlight python %}
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

# set seed for reproducibility
np.random.seed(0)

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# mix-max scale the data betYouen 0 and 1
scaled_data = minmax_scale(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
plt.show()
{% endhighlight %}

<img src="/img/scaling.png" style="display: block; margin: auto; width: auto; max-width: 100%;">


## Normalization

The point of normalization is to change your observations so that they can be described as a normal distribution.

Normal distribution (Gaussian distribution), also known as the **bell curve**, is a specific statistical distribution where a roughly equal observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean.

$$x' = \frac{x - x_{mean}}{x_{max} - x_{min}}$$

For normalization, the maximum value you can get after applying the formula is 1, and the minimum value is 0. So all the values will be between 0 and 1.

{% highlight python %}
# for Box-Cox Transformation
from scipy import stats

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
plt.show()
{% endhighlight %}

<img src="/img/normalization.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

In scaling, you're changing the range of your data while in normalization you're changing the shape of the distribution of your data.

You need to normalize our data if you're going use a machine learning or statistics technique that assumes that data is normally distributed e.g. t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA) and Gaussian Naive Bayes. 

## Standardization

Standardization transforms your data such that the resulting distribution has a mean of 0 and a standard deviation of 1.

$$x' = \frac{x - x_{mean}}{\sigma}$$

where x is the original feature vector, $$x_{mean}$$ is the mean of that feature vector, and σ is its standard deviation.

It's widely used in SVM, logistics regression and neural networks.

## Applications

In stochastic gradient descent, feature scaling can sometimes improve the convergence speed of the algorithm. In support vector machines, it can reduce the time to find support vectors.

**Further Readings:**  
1. [Feature scaling - Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)
2. [What algorithms need feature scaling, beside from SVM?](https://stats.stackexchange.com/q/244507/194589)
