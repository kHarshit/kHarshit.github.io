---
layout: post
title: "Ensembling is the key"
date: 2018-03-16
categories: [Data Science, Machine Learning]
---

Most of us have our favourite machine learning algorithms. For some, it may be state-of-the-art algos like *Support Vector Machines* while for others it may be something simple like *Naive Bayes*.

But the truth is that   the performance of these algorithms vary from problem to problem. For some applications, it may be good to use *random forest* while for others, *logistic regression* might be the optimal choice.

What if instead of using a single algorithm, we use multiple ones together? Is it even possible? Yes, it is. *random forest* does that. It is called *model ensembles*. In the simplest technique called *bagging*, we take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions. 

*Random forest* provide an improvement over bagged trees. While building a number of decision trees, each time a split in a tree is considered, a random sample of `m (< p)` predictors is chosen from the set of `p` predictors. Random forest differs from bagging in the choice of predictor subset size m.

In *boosting*, decision trees are built sequentially i.e. each tree is grown using information from previously grown trees unlike in bagging where we create multiple copies of original training data and fit separate decision tree on each. In boosing, each tree is fitted on modified version of the original data set.

In *stacking*, the outputs of individual classifiers become the inputs of a *higher-level* learner that figures out how best to combine them.

<img src="/img/ensemble.jpg" style="float: right; display: block; margin: auto; width: auto; max-width: 100%;">

Many other techniques exist. In the Netflix prize, teams found that they obtained the best results by combining their learners with other teams’, and merged into larger and larger teams. The winner and runner- up were both stacked ensembles of over 100 learners, and combining the two ensembles further improved the results.

Thus, model ensembles are a key part of machine learning tookit though they are complex to interpret.


**References:**
1. Pedro Domingos, A Few Useful Things to Know about Machine Learning  
2. <a href="http://www-bcf.usc.edu/~gareth/ISL/">An Introduction to Statistical Learning</a> 
3. <a href="https://jamesmccaffrey.wordpress.com/2016/09/22/machine-learning-ensemble-model-averaging/">Image source</a> 
