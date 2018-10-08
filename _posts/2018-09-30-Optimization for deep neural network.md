---
layout: post
title: "深度神经网络的优化"
date: 2018-09-30
categories: 深度学习
tags: [Feedforwrd, 神经网络, 量化, 压缩]
grammar_cjkRuby: true
---

# 1. 简介
深度学习作为机器学习的一个分支，近年来受到了广泛的追捧和应用，但其庞大的运算量一直是制约其发展的瓶颈，因此近年来也涌现出了一系列的神经网络优化方法，本文结合自身的一些工作心得，简单介绍一些深度神经网络的优化方法。

目前，主流的深度神经网络结构有MLP, CNN, RNN等，但针对它们的优化方法都是类似的，因此本文主要介绍FeedForward 网络的优化。

## 1.1 FeedForward
 ![Fig 1. Feed Forward 网络结构](/images/dnn_optimization/1.png)

 前向网络的推理计算大体可以分为两步：
 * 对输入做仿射变换：

$$
z_i = W_i h_{i-1} + b_i
$$

 * 激活函数做非线性变换(常用的激活函数有 Sigmoid, RELU，**分类任务中，输出层的激活函数则常采用Softmax**)：

$$
h_i = f(z_i)
$$

其中\\(h_i\\) 是第\\(i\\)个隐藏层的输出，\\(W_i\\)是对应的权值矩阵，\\(b_i\\) 是对应的偏置向量。

本文的内容也将围绕上述两步展开。

# 2. 计算优化
## 2.1 归一化处理
通常，机器学习的算法都要求对输入的特征向量做归一化处理，即

$$
\bar{x} = \frac{x - \mu}{\sigma}
$$

其中\\(\mu\\)为均值，\\(\sigma\\)为标准差。

为了减少计算量，可以将上述过程与第一层的仿射变换(\\(W_1 \bar{x} + b_1\\))合并，

$$
\begin{align*}
W_1 \bar{x} + b_1 &= W_1 (\frac{x - \mu}{\sigma}) + b_1 \\
                  &=\frac{W_1}{\sigma} x + (b_1 - \frac{W_1}{\sigma} \mu)
\end{align*}
$$

令\\(W_1 = \frac{W_1}{\sigma}, b_1 = b_1 - \frac{W_1}{\sigma} \mu\\)，则可以**保证在数学运算结果一致的情况下，节省了输入归一化带来的运算量**。

## 2.2 优化仿射变换
在给定 DNN 模型的情况下，计算的核心其实就是矩阵运算，其大部分的运算量都消耗在仿射变换上，因此对仿射变换进行优化至关重要。

### 2.2.1 FanIn & FanOut
![Fig 2. FanIn & FanOut](/images/dnn_optimization/2.png)

### 2.2.2 稀疏性处理

## 2.3 优化激活函数
激活函数的优化主要是针对 Softmax：

$$
    Softmax(z_i) = \frac{e^{z_i - z_{max}}}{\sum e^{z_j - z_{max}}}
$$

显然，如果

$$
Softmax(z_i) > Softmax(z_j)
$$

那么

$$
\log (Softmax(z_i)) > \log (Softmax(z_j))
$$

并且由于

$$
a - c > b -c \Leftrightarrow a > b
$$

因此对于

$$
\log (Softmax(z_i)) = z_i - z_{max} - \log (\sum e^{z_j - z_{max}})
$$

每次计算是否减去**常数项\\(z_{max} + \log (\sum e^{z_j - z_{max}})\\)** 并不会改变最终的竞争条件，即相互之间的大小关系，因此模型在推理计算时，如果不需要概率值，可以移除 Softmax 的计算，直接采用仿射变换的结果即可。

# 3. 量化与压缩
## 3.1 定点化

## 3.2 Exception List

# 4. 其他优化方法
* 模型剪枝
* 知识蒸馏
* SVD 分解
* Lazy Calculation(在语音识别领域，跳帧处理，既可以降低计算量，同时保持性能基本不变。)

# 5. 总结
近期在做一些神经网络的优化工作，使得神经网络可以在低功耗的嵌入式设备上运行，因此特将整个过程中的一些心得体会记录下来，若有不当之处，欢迎交流指正。
