---
layout: post
title: "Attention"
date: 2019-03-08
categories: [Deep Learning, Computer Vision, Natural Language Processing]
---

Attention is the technique through which the model focuses itself on a certain region of the image or on certain words in a sentence just like the same way the humans do. For example, when looking at an image, humans shifts their attention to different parts of the image one at a time rather than focusing on all parts in equal amount at the same time.

## Encoder-Decoder architecture

A Recurrent Neural Network (RNN) can be used to map a varied-length input sequence to the varied-length output sequence e.g. machine translation. The simplest sequence-to-sequence architecture that makes it possible consists of two models, Encoder, and Decoder. In Encoder-Decoder architecture, the idea is to learn a fixed-size context vector that contains the essential information of the inputs.

* The **Encoder** processes the input sequence $$(x_1, x_2, x_3, ..., x_n)$$ and gives a context vector, `C` that summarizes the input sequence.
* The **Decoder** uses `C` to generate the output sequence $$(y_1, y_2, y_3, ..., y_m)$$, where `n` and `m` may not be equal.

<img src="/img/encoder_decoder_arch.png" style="display: block; margin: auto; width: auto; max-width: 100%;">

The final hidden state of the hidden state of encoder RNN is used to compute `C`, which is further provided as an input to the decoder.

The problem with encoder-decoder approach is that all the input information needs to be compressed in a fixed length context vector, `C`. It makes it difficult for the network to cope up with large amount of input information *(e.g. in text, large sentences)* and produce good results with only that context vector. With **attention mechanism**, the encoder RNN instead of producing a single context vector to summarize the input image or sequence, produces a grid of vectors. That is, attention makes `C` a variable-length sequence rather than a fixed-size vector.

## Additive Attention

## Multiplicative Attention

**Further Readings:**  
1. [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
2. [How Does Attention Work in Encoder-Decoder Recurrent Neural Networks](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/)
3. [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)