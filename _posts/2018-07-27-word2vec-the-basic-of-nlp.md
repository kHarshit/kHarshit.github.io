---
layout: post
title: "word2vec: The foundation of NLP"
date: 2018-07-27
categories: [Data Science, Natural Language Processing]
---

**Natural Language Processing** deals with the task of making computer understand the human language. The computer understands only 0s and 1s. A language consists of words. So, the first task is to convert word into a combination of 0s and 1s. This is easy. But, a word is nothing without its meaning. Hence, the task we need to solve is to represent the meaning of a word.

## Representing words as discrete symbols

One solution to this problem is to use *one-hot vector*. For example, in the sentence, "*I love machine learning*", the word *machine* can be represented as [0, 0, 1, 0] *where* 1 corresponds to *machine*. In the same way, we can represent other words such as *learning*, [0, 0, 0, 1].

The problem with this approach is that the combination of words has no meaning i.e. to match "machine learning" in a given document. we have got two separate vectors of *machine* and *learning*, which unfortunately are orthogonal. Thus, such a search will not yield any result. That is, there is no notion of similarity for one-hot vectors.

## Representing words by their contexts

The core idea in this approach is that the words are represented by the context they have i.e. a word's meaning is given by the words that frequently appear close-by.

word2vec<sup id="a1">[1](#myfootnote1)</sup> is a model for learning word vectors.


**To be continued...**


**References:**  
<a name="myfootnote1"></a>1: [word2vec:  Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) [↩](#a1) 
