---
layout: post
title: "The Curse of Dimensionality"
date: 2018-01-26
categories: [Data Science, Machine Learning]
---

While applying k nearest neighbors approach in solving a problem, we can sometimes notice that there is a deterioration in the kNN performance when the number of predictors, `p` is large. The reason for this can be the high number of dimensions. This problem is known as *the curse of dimensionality*.

It means that the test error tends to increase as the dimensionality of the problem (number of predictors) increases, unless the additional features are *truly* associated with the response. It is opposite to the thought one might have that as the number of predictors used to fit a model increases, the quality of the fitted model will increase as well.

A [quora answer](https://www.quora.com/What-is-the-curse-of-dimensionality/answer/Kevin-Lacker?share=f6be224e&srid=4Ozb) best explains this:
> Let's say you have a straight line 100 yards long and you dropped a penny somewhere on it. It wouldn't be too hard to find. You walk along the line and it takes two minutes.

> Now let's say you have a square 100 yards on each side and you dropped a penny somewhere on it. It would be pretty hard, like searching across two football fields stuck together. It could take days.  
Now a cube 100 yards across. That's like searching a 30-story building the size of a football stadium. Ugh.

> The difficulty of searching through the space gets a *lot* harder as you have more dimensions. You might not realize this intuitively when it's just stated in mathematical formulas, since they all have the same "width". That's the curse of dimensionality. It gets to have a name because it is unintuitive, useful, and yet simple.

<abbr title="These methods don't make explicit assumptions about the functional form of function f that explains the relationship b/w predictiors and response; instead they seek an estimate of f that gets as close to data points as possible.">Non-parametric approaches</abbr> like kNN often perform poorly when `p` is large. The decrease in performance of kNN arises from the fact that in higher dimensions there is effectively a reduction in sample size. Say, in a dataset, there are 100 training observations; when `p = 1`, this provides enough information for a good fit. However, spreading 100 observations over `p = 20` dimensions doesn't. The k observations that are nearest to a given test observation $$x_{0}$$ may be very far away from $$x_{0}$$ in p-dimensional space, if p is large, leading to poor prediction (a poor kNN fit).

### How to avoid the curse of dimensionality?

We can use various feature selection algorithms like best subset selection or stepwise selection to select a subset of predictors.

Another approach is to use PCA (principal component analysis), a dimension reduction method, which transforms the p predictors into M < p predictors. The model is fit using these M predictors. PCR (principal component regression), based on PCA, provides a way to perform regression using M < p predictors. PCA is a feature extraction method.

**References:**  
1. [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
2. [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)  
