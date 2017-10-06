---
layout: post
title: "Data leakage: A big problem"
date: 2017-10-06
categories: [Technical Fridays]
---

Let's say your machine learning model performs better than than you expect it to in the test set. You are happy. Well, you should be. Now, you release the model to the world, or say you apply the model on different data. But, then something awful happens. Oops! your model performs worse than it did in your test set. And not just worse, it does not make a single accurate prediction. The model which you trained for hours and gave a great R<sup>2</sup> value is abandoning you now. You spend hours in debugging. Then, you find the culprit: data leakage. 

Data Leakage is the creation of unexpected additional information in the training data, allowing a model or machine learning algorithm to make unrealistically good predictions.

Data leakage can happen when the data you are using to train a machine learning algorithm contains the information you are trying to predict.

## Hot dog / Not hot dog

For example, you want to train a machine learning algorithm to classify whether the image contains a hot dog or not. And you train your model with the following images: 
<img src="/img/hotdog1.jpg" style="display: block; margin: auto; width: auto; max-width: 100%;">  
<img src="/img/hotdog2.jpg" style="display: block; margin: auto; width: auto; max-width: 100%;">  
<img src="/img/hotdog3.jpg" style="display: block; margin: auto; width: auto; max-width: 100%;">  

Your classifier may seem to do well on most hot dogs but it may decide a hot dog is a thing which has yellow sausage on it. So, it may not recognize this as a hot dog.

<img src="/img/hotdog4.jpg" style="display: block; margin: auto; width: auto; max-width: 100%;">  

In this case, the yellow sausage gave your model too much predictive power and your model is using it to cheat on its predictions.

## How to solve the problem of data leakage?

You can minimize data leakage by:
* performing data preparation within each cross-validation fold separately, not using the entire dataset.
* splitting off a validation set for final sanity check of developed models.

Before building the model, look for features highly correlated with the target value. Also, if your model is performing too good to be true, it may be an indication of data leakage. Data leakage is one the most common machine learning problems. So, it must be taken care of.
