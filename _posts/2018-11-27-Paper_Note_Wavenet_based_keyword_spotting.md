---
layout: post
title: "Wavenet based keyword spotting"
date: 2018-11-27
categories: 论文笔记
tags: [PixelRNN, PixelCNN, WaveNet, Keyword Spotting, wakeup word detection]
grammar_cjkRuby: true
---

# 1. Overview
[Snips](https://snips.ai/)团队在《Efficient keyword spotting using dilated convolutions and gating》这篇论文中采用 wavenet 这一生成模型来做 Keyword spotting(Wakeup word detection)，得到了非常不错的结果，如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/1_1.png" width="75%" height="75%" />
</p>
为此，我们追根溯源，从 PixelRNN 开始，一步步分析 Wavenet 的进化过程，以及它如何应用于 keyword spotting。

# 2. PixelRNN/PixelCNN
Google DeepMind团队在《pixel recurrent neural networks》中首次提出了生成模型 PixelRNN/PixelCNN。 所谓生成模型，它生成的图片的像素点的联合分布和训练集的图片像素点的联合分布相近，即给它一组图片作为训练集让模型进行学习，模型可以生成一组和训练集图片尽可能相近的图片。

若想要生成一张图片，直观的想法就是：逐行逐像素(pixel by pixel)的生成，并且每一个像素的生成都依赖于它前面的像素，因此这个问题就在数学上就可以表示为对该像素点的条件概率的求解：

$$
p(x) = \prod_{i=1}^{n^2}p(x_i|x_1, ..., x_{i-1})    \tag{1}
$$

显然，**RNN 可以有效的对这种有序的序列建模，这就是 PixelRNN 被提出的直观想法**。对于图像而言，它有 RGB 三个通道，上述条件概率中，当前像素点的估计只依赖于之前的像素，如果将当前像素点已经估计的通道值用于其他通道的估计呢？为此将上述条件概率改写为：

$$
\begin{equation}
p(x_i|\mathbf{x}_{<i}) = p(x_{i,R}|\mathbf{x}_{<i}) p(x_{i,G}|\mathbf{x}_{<i},x_{i,R})p(x_{i,B}|\mathbf{x}_{<i},x_{i,R},x_{i,G})
\end{equation}
$$

为了实现上述条件概率的建模，作者在模型卷积的过程中采用了两种 MASK，如下图
<p align="center">
<img src="/images/wavenet_keyword_spotting/2_1.png" width="50%" height="50%" />
</p>

在 PixelRNN 中，作者采用了12个二维 LSTM layer，并且论文中对 LSTM layer 提出了两种不同的结构：Row LSTM 和 BiLSTM。
## 2.1 Row LSTM
Row LSTM 在结构上与一般的 LSTM 并没有区别，如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/2_2.png" width="55%" height="55%" />
</p>
只是其输入变成了当前行 feature map(input-to-state) 与 LSTM 上一行输出(state-to-state)的卷积和，即

$$
\begin{align*}
[o_i, f_i, i_i, g_i] &= \sigma (K^{ss} h_{i-1}) + K^{is} x_i) \\
c_i &= f_i \odot + i_i \odot g_i  \\
h_i &= o_i \odot \tanh(c_i)
\end{align*}
$$

其中对于\\(o_i, f_i, i_i\\), \\(\sigma\\) 为 sigmoid 函数，对于\\(g_i\\)，\\(\sigma\\)为 tanh 函数。

如果state-to-state 的卷积核和 input-to-state 的卷积核大小都为\\(3 \times 1\\)，那么 Row LSTM 的 receptive field 如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/2_3.png" width="20%" height="20%" />
</p>

**由于 receptive field 的范围限制，当前像素点的估计并没有用到之前所有的像素点的信息(即存在盲点，blind spot)， 无法很好的对公式(1)建模**，为此作者提出了 BiLSTM 的结构。

## 2.2 BiLSTM
如下图所示，**BiLSTM的 receptive field 完美的包含了之前所有点的信息。**

<p align="center">
<img src="/images/wavenet_keyword_spotting/2_4.png" width="20%" height="20%" />
</p>
那么它是怎么做到的呢？
 
首先它采用了\\(1\times 2\\)大小的卷积核，并且对于\\(n\times n\\)的图像，从上到下，每一行各往右移一个像素，图像大小由此变为\\(n \times 2n\\)，那么对于正向的 LSTM 而言，它的 receptive field 如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/2_5.png" width="50%" height="50%" />
</p>

卷积完成后，再把各行的像素各往左移一个像素，使其重新变为\\(n \times n\\)大小。

对于反向的 LSTM，采用同样的操作，那么就可以得到当前像素右边所有像素的信息，为了防止当前像素之后的像素信息参与到当前像素点的估计，在计算方向 LSTM 的时候，需要将图像往下移一行，然后再将卷积的结果叠加到正向 LSTM 中。

通过上述方法，可以将历史的全部像素点信息都参与到当前像素的估计中，解决了盲点的问题。**但是由于 RNN 的固有缺陷决定了 PixelRNN 只能做 sequence training**，为了加速训练过程，作者又提出了 PixelCNN。


## 2.3 PixelCNN
PixelCNN 中，只用当前的 feature map 参与当前像素点的估计，因此其训练过程可以很好的并行训练，加速训练过程。但其也存在盲点（尽管加大卷积核的大小以及增加卷积的层数，可以再一定程度上缓解这个问题），以\\(3\times 3\\)的卷积核为例，其 receptive field 如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/2_6.png" width="20%" height="20%" />
</p>

## 2.4  其他
为了训练更深的网络，作者借鉴了深度残差网络(ResNet) 的结构，第每一层都采用了 bypass，如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/2_7.png" width="50%" height="50%" />
</p>
论文中，对于上述三种网络，其结构如下图：
<p align="center">
<img src="/images/wavenet_keyword_spotting/2_8.png" width="50%" height="50%" />
</p>

总体性能上，**PixelCNN < PixelRNN(Row LSTM) < PixelRNN(BiLSTM)**。

# 3. Gated PixelCNN
DeepMind 团队在论文《Conditional image generation with pixelcnn decoders》中提出了 Gated PixelCNN，它是在 PixelCNN的基础上做了一些改进得到的。PixelCNN 主要存在两大缺陷：
* 表现性能不如 PixelRNN
* 存在 blind spot，即当前的像素值信息提取的过程中无论如何都不会包括到像素点，如下图中的灰色区域。
<p align="center">
<img src="/images/wavenet_keyword_spotting/3_1.png" width="40%" height="40%" />
</p>

作者针对上述两个方面，提出了相应的解决方。
## 3.1 表现性能
为什么 PixelRNN的效果更好，一个可能的原因是 每一个LSTM层都可以之前所有像素点的信息。对此，可以增加 CNN 的卷积核和层数来缓解。另一方面，PixelRNN 表现不错，是因为LSTM 的门结构可以对更复杂的结构建模，所以作者将masked convolutions 之后的 RELU 激活函数改为类似于门结构的激活函数：

$$
y = \tanh (W_{k,f}*x) \odot \sigma(W_{k,g}*x)
$$

其中，\\(*\\)表示卷积，\\(\odot\\)表示按元素相乘。

## 3.2 盲点

作者将之间的卷积分成两个卷积网络：horizontal stack 和 vertical stack，如下图所示。
<div align="center">
<img src="/images/wavenet_keyword_spotting/3_2.png" width="40%" height="40%" />
</div>
1. horizontal stack 对当前像素所在行进行卷积，所以其卷积核大小为\\(1 \times n\\)，当然也需要用 mask 来实现只对前面像素信息的提取。 
2. vertical stack 则是对当前行前面所有行像素的卷积，其卷积核大小为\\(n \times n\\)， 同样适用 mask 来实现只对当前像素之前像素的信息提取。
3. **将每一层中这两种 stack 信息结合，就可以提取到当前像素点前面所有的信息(结构如下图)，这样就解决了盲点问题**。
<p align="center">
<img src="/images/wavenet_keyword_spotting/3_3.png" width="80%" height="80%" />
</p>

## 3.3 Conditional PixelCNN
在图像生成的时候，我们可能已经有一些先验知识，那么该如何将这些先验知识应用到网络中呢？即如何对下面的分布进行建模：

$$
p(x|h) = \prod_{i=1}^{n^2} p(x_i|x_1,...,x_{i-1}, h)
$$

其中\\(h\\)为先验知识。

为此，作者提出了 conditional PixelCNN，将先验知识加入到每层中门结构的激活函数中：

$$
y = \tanh (W_{k,f}*x + V_{k,f}^{T}h) \odot \sigma(W_{k,g}*x + V_{k,g}^{T}h)
$$

其中\\(h\\)可以使一组类别标签(one-hot 向量)，也可以是其他网络生成的向量。

**应用上述优化后，Gated PixelCNN 的性能基本上就接近于 PixelRNN 了**。

# 4. WaveNet
图像上的成功应用之后，DeepMind 团队在《WaveNet: A generative model for raw audio》中将生成模型用于语音信号的生成，类似的，就是对以下分布进行建模：

$$
p(x) = \prod_{t=1}^{T}p(x_t|x_1, ..., x_t-1)
$$

将PixelCNN用于语音信号的生成存在一个问题：**图像大小有限，用一个与图像一样大小的卷积核就看到历史的所有信息，而对于长时语音信号来说，要得到历史的所有信息几乎是不可能的。**为此，论文中的gated PixelCNN做了两点改进：
1. 使用一维卷积，并且为了保证只提取到历史语音数据的信息，并没有采用mask的方式，而是采用了因果卷积(causal convolution)。
2. 为了看到尽可能多的历史数据，并且减小计算量，采用了空洞卷积(dilated convolution)。
因此其卷积过程如下图：
<div align="center">
<img src="/images/wavenet_keyword_spotting/4_1.gif" width="70%" height="60%" />
</div>
Gated PixeCNN中的门结构以及bypass connection在wavenet中也使用了，同时增加了skip connection用于语音信号的生成，最终的网络结构如下图：
<div align="center">
<img src="/images/wavenet_keyword_spotting/4_2.png" width="75%" height="70%" />
</div>
**通过多个dilated cnn layer 的级联，可以将非常长的历史数据用于当前语音数据的生成**。另外对于TTS来说，有text文本作为先验的数据，因此conditional wavenet也继承conditional gated PixelCNN中的实现，将先验知识加入到门结构激活函数的计算中：

$$
z = \tanh (W_{f,k}*x + V_{f,k}^{T}h) \odot \sigma(W_{g,k}*x + V_{g,k}^{T}h)
$$

Wavenet在生成语音数据时，依然只能串行生成，速度非常慢。因此后面又有两篇论文《Fast wavenet generation algorithm》和《Parallel WaveNet: Fast high-fidelity speech synthesis》对其进行了优化加速。

# 5. Wavenet for Keyword Spotting
Wavenet 在TTS领域内取得了巨大的成功，也被用在了其他一些领域，比如VAD(Voice activity detection)。Snips团队首次将其用于Keyword spotting，并得到了不错的结果。

## 5.1 改进
在具体的实现中，相比于Wavenet有以下几点调整：
* 模型结构基本上沿用了Wavenet的结构，只是在Dilated causal convolution layers的参数上有所调整
* 输入从语音的raw data变为了20维的MFCC特征(extracted from the input audio every 10ms over a window of 25ms)
* 输出直接是类别信息(1: keyword, 0: non-keyword)
* **在数据的label上，并没有把keyword所对应的frames全部标注成 1，而是以EOS(end of speech)为中心，左右两边\\(\Delta t\\)范围内的数据标注为 1， 这样可以显著的减少FA。**

## 5.2 Results & Analysis
在如下图所示的数据集上，
<div align="center">
<img src="/images/wavenet_keyword_spotting/5_1.png" width="40%" height="40%" />
</div>
相比于CNN 和 LSTM，性能得到了大幅提升。
<div align="center">
<img src="/images/wavenet_keyword_spotting/5_2.png" width="90%" height="90%" />
</div>


wavenet 在keyword spotting这个任务上的性能也能这么优越，我觉得主要有以下几个原因：
1. **多个Dilated causal convolution layers级联，它的left context非常长，甚至可以看到整个keyword**
2. **Gated activation 可以对更复杂的数据进行建模，使其在噪声环境下可以由更好的鲁棒性。**
3. **Label。精细调整的label，可以抑制FA，尤其是那种只包含keyword片段的speech导致的FA。**



# Reference
*[1] Oord A, Kalchbrenner N, Kavukcuoglu K. Pixel recurrent neural networks[J]. arXiv preprint arXiv:1601.06759, 2016.   
[2] van den Oord A, Kalchbrenner N, Espeholt L, et al. Conditional image generation with pixelcnn decoders[C]//Advances in Neural Information Processing Systems. 2016: 4790-4798.    
[3] Van Den Oord A, Dieleman S, Zen H, et al. WaveNet: A generative model for raw audio[C]//SSW. 2016: 125.   
[4] Paine T L, Khorrami P, Chang S, et al. Fast wavenet generation algorithm[J]. arXiv preprint arXiv:1611.09482, 2016.   
[5] Oord A, Li Y, Babuschkin I, et al. Parallel WaveNet: Fast high-fidelity speech synthesis[J]. arXiv preprint arXiv:1711.10433, 2017.   
[6] Coucke A, Chlieh M, Gisselbrecht T, et al. Efficient keyword spotting using dilated convolutions and gating[J]. arXiv preprint arXiv:1811.07684, 2018.   
[7] [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)    
[8] [High-fidelity speech synthesis with WaveNet](https://deepmind.com/blog/high-fidelity-speech-synthesis-wavenet/)    
*


