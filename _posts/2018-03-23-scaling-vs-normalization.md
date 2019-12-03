---
layout: post
title: "Scaling vs Normalization"
date: 2018-03-23
categories: [Data Science, Machine Learning]
---

**Feature scaling** (also known as **data normalization**) is the method used to standardize the range of features of data. Since, the range of values of data may vary widely, it becomes a necessary step in data preprocessing while using machine learning algorithms.


## Scaling

In scaling *(also called **min-max scaling**)*, you transform the data such that the features are within a specific range e.g. [0, 1].

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

# mix-max scale the data between 0 and 1
scaled_data = minmax_scale(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0], color='y')
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
plt.show()
{% endhighlight %}

<img src="/img/scaling.png" style="display: block; margin: auto; width: auto; max-width: 100%;">


## Normalization and Standardization

The point of normalization is to change your observations so that they can be described as a normal distribution.

Normal distribution (Gaussian distribution), also known as the **bell curve**, is a specific statistical distribution where a roughly equal observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean.

<a href="https://commons.wikimedia.org/wiki/File:The_Normal_Distribution.svg#/media/File:The_Normal_Distribution.svg"><img src="https://upload.wikimedia.org/wikipedia/commons/2/25/The_Normal_Distribution.svg" alt="The Normal Distribution.svg" style="display:block; margin: auto; width:80%; max-width:100%"></a>

***Note:** The above definition is as per statistics. There are various types of normalization. In fact, min-max scaling can also be said to a type of normalization. In machine learning, the following are most commonly used.*

## #1

**Standardization** *(also called **z-score normalization**)* transforms your data such that the resulting distribution has a mean of 0 and a standard deviation of 1. It's the definition that we read in the last paragraph.

$$x' = \frac{x - x_{mean}}{\sigma}$$

where x is the original feature vector, $$x_{mean}$$ is the mean of that feature vector, and σ is its standard deviation.

The z-score comes from statistics, defined as 

$$z = \frac{x - \mu}{\sigma}$$

<img src="/img/standardization.gif" style="display: block; margin: auto; width: auto; max-width: 100%;">

where $$\mu$$ is the mean. By subtracting the mean from the distribution, we're essentially shifting it towards left or right by amount equal to mean i.e. if we have a distribution of mean 100, and we subtract mean 100 from every value, then we shift the distribution left by 100 without changing its shape. Thus, the new mean will be 0. When we divide by standard deviation $$\sigma$$, we're changing the shape of distribution. The new standard deviation of this standardized distribution is 1 which you can get putting the new mean, $$\mu = 0$$ in the z-score equation.

It's widely used in SVM, logistics regression and neural networks.

## #2

Simply called **normalization**, it's just another way of normalizing data. Note that, it's a different from min-max scaling in numerator, and from z-score normalization in the denominator.

$$x' = \frac{x - x_{mean}}{x_{max} - x_{min}}$$

For normalization, the maximum value you can get after applying the formula is 1, and the minimum value is 0. So all the values will be between 0 and 1.

{% highlight python %}
# for Box-Cox Transformation
from scipy import stats

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0], color='y')
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
plt.show()
{% endhighlight %}

<img src="/img/normalization.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

In scaling, you're changing the range of your data while in normalization you're mostly changing the shape of the distribution of your data.

You need to normalize our data if you're going use a machine learning or statistics technique that assumes that data is normally distributed e.g. t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA) and Gaussian Naive Bayes. 

## Applications

In stochastic gradient descent, feature scaling can sometimes improve the convergence speed of the algorithm. In support vector machines, it can reduce the time to find support vectors.

**Further Readings:**  
1. [Feature scaling - Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)
2. [Normalization](https://en.wikipedia.org/wiki/Normalization_(statistics)#Examples)
3. [What algorithms need feature scaling, beside from SVM?](https://stats.stackexchange.com/q/244507/194589)  
4. [Scaling and Normalize Data](https://www.kaggle.com/jfeng1023/data-cleaning-challenge-scale-and-normalize-data)  
5. [Standardization - scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)  
6. [Compare the effect of different scalers on data with outliers - scikit-learn](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)  
7. [Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)
