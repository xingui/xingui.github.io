---
layout: post
title: "Wavenet based keyword spotting"
date: 2018-11-27
categories: 论文笔记
tags: [PixelRNN, PixelCNN, WaveNet, Keyword Spotting, wakeup word detection]
grammar_cjkRuby: true
---

# 1. Overview
[Snips](https://www.baidu.com/link?url=71r7macqyrkGTnFqtQh42KCKS7oU6uaLTVUKXNLMR5e&wd=&eqid=fe9d06200000ca0f000000045bfd6681)团队在《Efficient keyword spotting using dilated convolutions and gating》这篇论文中采用 wavenet 这一生成模型来做 Keyword spotting(Wakeup word detection)，得到了非常不错的结果，如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/1_1.png" width="75%" height="75%" />
</p>
为此，我们追根溯源，从 PixelRNN 开始，一步步分析 Wavenet 的进化过程，以及它如何应用于 keyword spotting。

# 2. PixelRNN/PixelCNN
Google DeepMind团队在《pixel recurrent neural networks》中首次提出了生成模型 PixelRNN/PixelCNN。 所谓生成模型，它生成的图片的像素点的联合分布和训练集的图片像素点的联合分布相近，即给它一组图片作为训练集让模型进行学习，模型可以生成一组和训练集图片尽可能相近的图片。

若想要生成一张图片，直观的想法就是：逐行逐像素(pixel by pixel)的生成，并且每一个像素的生成都依赖于它前面的像素，因此这个问题就在数学上就可以表示为对该像素点的条件概率的求解：

$$
p(x) = \prod_{i=1}^{n^2}p(x_i|x_1, ..., x_{i-1})
$$

显然，**RNN 可以有效的对这种有序的序列建模，这就是 PixelRNN 被提出的直观想法**。对于图像而言，它有 RGB 三个通道，上述条件概率中，当前像素点的估计只依赖于之前的像素，如果将当前像素点已经估计的通道值用于其他通道的估计呢？为此将上述条件概率改写为：

$$
p(x_i|\mathbf{x}_{<i}) = p(x_{i,R}|\mathbf{x}_{<i}) p(x_{i,G}|\mathbf{x}_{<i},x_{i,R})p(x_{i,B}|\mathbf{x}_{<i},x_{i,R},x_{i,G})
$$

为了实现上述条件概率的建模，作者在模型卷积的过程中采用了两种 MASK，如下图
<p align="center">
<img src="/images/wavenet_keyword_spotting/1_2.png" width="50%" height="50%" />
</p>

在 PixelRNN 中，作者采用了12个二维 LSTM，

# 3. Gated PixelCNN

# 4. WaveNet

# 5. Wavenet for Keyword Spotting

# 6. Result & Analysis


# Reference
*[1] Oord A, Kalchbrenner N, Kavukcuoglu K. Pixel recurrent neural networks[J]. arXiv preprint arXiv:1601.06759, 2016.   
[2] van den Oord A, Kalchbrenner N, Espeholt L, et al. Conditional image generation with pixelcnn decoders[C]//Advances in Neural Information Processing Systems. 2016: 4790-4798.    
[3] Van Den Oord A, Dieleman S, Zen H, et al. WaveNet: A generative model for raw audio[C]//SSW. 2016: 125.   
[4] Paine T L, Khorrami P, Chang S, et al. Fast wavenet generation algorithm[J]. arXiv preprint arXiv:1611.09482, 2016.   
[5] Oord A, Li Y, Babuschkin I, et al. Parallel WaveNet: Fast high-fidelity speech synthesis[J]. arXiv preprint arXiv:1711.10433, 2017.   
[6] Coucke A, Chlieh M, Gisselbrecht T, et al. Efficient keyword spotting using dilated convolutions and gating[J]. arXiv preprint arXiv:1811.07684, 2018.*

