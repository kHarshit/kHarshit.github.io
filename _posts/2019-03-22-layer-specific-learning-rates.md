---
layout: post
title: "Layer-specific learning rates"
date: 2019-03-22
categories: [Deep Learning]
---

***Note:*** *This post answers the question I asked on Cross-Validated: [What are the benefits of layer-specific learning rates?](https://stats.stackexchange.com/q/397848/194589)*

The layer-specific learning rates mean using different learning rates for different layers of neural networks instead of using the same global learning rate for each layer.

The layer-specific learning rates help in overcoming the slow learning *(thus slow training)* problem in deep neural networks. As stated in the paper titled Layer-Specific Adaptive Learning Rates for Deep Networks<sup id="a1">[1](#myfootnote1)</sup>:

> When the gradient descent methods are used to train **deep networks**, additional problems are introduced. As the number of layers in a network increases, the gradients that are propagated back to the initial layers get very small ([vanishing gradient problem]({% post_url 2019-01-04-the-gradient-problem-in-rnn %})). This dramatically slows down the rate of learning in the initial layers and slows down the convergence of the whole network.

> The learning rates specific to each layer in the network allows larger learning rates to compensate for the small size of gradients in shallow layers *(layers near the input layer)*.


The layer-specific learning rates also help in **transfer learning** -- check differential learning rates<sup id="a2">[2](#myfootnote2)</sup>, discriminative fine-tuning by Jeremy Howard<sup id="a3">[3](#myfootnote3)</sup> and post on CaffeNet<sup id="a4">[4](#myfootnote4)</sup>.

The intuition is that in the layers closer to the input layer are more likely to have learned more general features -- such as lines and edges, which we won’t want to change much. Thus, we set their learning rate low. On the other hand, in case of later layers of the model -- which learn the detailed features, we increase the learning rate -- to let the new layers learn fast.

**References:**  

<a name="myfootnote1"></a>1. [Layer-Specific Adaptive Learning Rates for Deep Networks](https://arxiv.org/pdf/1510.04609.pdf) [↩](#a1)  
<a name="myfootnote2"></a>2. [Differential Learning Rates](https://blog.slavv.com/differential-learning-rates-59eff5209a4f) [↩](#a2)  
<a name="myfootnote3"></a>3. [Discriminative fine-tuning by Jeremy Howard](https://arxiv.org/pdf/1801.06146.pdf) [↩](#a3)  
<a name="myfootnote4"></a>4. [Post on CaffeNet](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html) [↩](#a4)  