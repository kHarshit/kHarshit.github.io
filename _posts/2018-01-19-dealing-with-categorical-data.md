---
layout: post
title: "Dealing with categorical data"
date: 2018-01-19
categories: [Technical Fridays, Data Science, Python]
---

Often in our machine learning model, we encounter qualitative predictors.

One of the ways to deal with these factors is to create a *dummy variable*. If a factor has two levels or possible values, then we simply create a dummy variable that takes two possible numerical values. For example, if `gender` variable takes two values -- `Male` and `Female`.

$$x_{i} = 1\ if\ ith\ person\ is\ female\\
0\ if\ ith\ person\ is\ male$$

and include this new variable in our model. This results in the model:

$$y_{i} = \beta_{0} + \beta_{1}x_{i} + \epsilon_{i}$$

If a qualitative predictor has more than two levels, we create additional dummy variables. For example, if `quality` has three possible values -- `Bad`, `Medium`, and `Good`.

We deal with these factors in the following ways:

## One Hot encoding and dummy encoding

One hot encoding is one of the widespread method used to deal with the categorical values. Another method is dummy encoding. There is a slight difference between the two.

For example, if one categorical variable has n values. One-hot encoding converts it into n variables, while dummy encoding converts it into n-1 variables.


### Using pandas

{% highlight python %}
PyDev console: using IPython 6.1.0
Python 3.6.3 |Anaconda, Inc.| (default, Nov  3 2017, 19:19:16) 
[GCC 7.2.0] on linux
In[2]: import pandas as pd
In[3]: df = pd.DataFrame({'quality': ['bad', 'medium', 'good']})
In[4]: df
Out[4]: 
  quality
0     bad
1  medium
2    good
In[5]: df.info()
<class'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 1 columns):
quality    3 non-null object
dtypes: object(1)
memory usage: 104.0+ bytes
In[6]: df = pd.get_dummies(df)  # one hot encoding
In[7]: df
Out[7]: 
   quality_bad  quality_good  quality_medium
0            1             0               0
1            0             0               1
2            0             1               0

{% endhighlight %}

By default, python's `get_dummies` doesn't do dummy encoding but one-hot encoding. To produce dummy encoding, use `drop_first=True`.
{% highlight python %}
In[8]: df = pd.get_dummies(df, drop_first=True)  # dummy encoding
In[9]: df
Out[9]: 
   quality_good  quality_medium
0             0               0
1             0               1
2             1               0
{% endhighlight %}

### Using scikit-learn

`scikit-learn` also provides mehods to deal with categorical variables e.g. `sklearn.preprocessing.LabelEncoder()`. `LabelEncoder` is incremental encoding, such as 0,1,2,3,4,... We can also use scikit-learn's `sklearn.preprocessing.OneHotEncoder()`.


**References:**  
1. [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
2. [sklearn.preprocessing.OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
