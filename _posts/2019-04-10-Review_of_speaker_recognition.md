---
layout: post
title: 声纹识别的技术演变
date: 2019-04-10
categories: 声纹识别
tags: [声纹识别, GMM, i-vector, x-vector, DNN]
grammar_cjkRuby: true
---

# 1 简介
通俗的讲，声纹识别就是辨别某一句话是不是由某一个人讲的。理论上，每个人声腔结构参数（尺寸等）都是不一样的，因此只要能提取到可以充分表达人与人之间声腔差异的特征，声音就可以像指纹一样，将不同的人有效区分。

声纹识别技术自上世纪40年代末开始，经历了一系列的技术演变逐步的走向成熟，并在这两年呈现了井喷式的发展。

## 1.1 声纹识别的分类
按照判别方式可分为：
* 说话人辨认（1:N，辨认未知说话人是已记录说话人中的哪一位）
* 说话人确认（1:1, 判断未知说话人是否为某个指定人）

按照音频内容可分为：
* 文本相关（未知说话人的说话内容固定不变）
* 文本无关（未知说话人的说话内容不确定）

按照测试人员集合可分为
* 开集（未知说话人不一定在已记录的说话人中）
* 闭集（未知说话人一定在已记录的说话人中）

## 1.2 评价指标
* **1:1**
    1. **False Rejection Rate**

    $$
    \begin{align}
    False\ Rejection\ Rate &= Miss \ probability \\
    &=\frac{NO.\ of\ true\ speaker\ rejected}{Total\ No.\ of\ true\ speaker\ trials}
    \end{align}
    $$

    2. **False Acceptance Rate**

    $$
    \begin{align}
    False\ Acceptance\ Rate &= False alarm \ probability \\
    &=\frac{NO.\ of\ impostors\ accepted}{Total\ No.\ of\ impostor\ attempts} \\
    \end{align}
    $$

    3. **EER(Equal error rate, FAR==FRR)**

* **1:N**

$$
Recognition\ Rate = \frac{NO.\ of\ correct\ recognitions}{Total\ No.\ of\ trials}
$$


## 1.3 技术演变
按照时间先后的发展，声纹识别技术演变可以大致分为：**GMM-UBM -> GMM-SVM -> I-Vector PLDA -> DNN PLDA -> E2E(end to end)**

# 2 GMM-UBM
高斯混合模型（GMM）是一系列的高斯分布函数的线性组合，理论上GMM可以拟合出任意类型的分布。一个充分训练的GMM模型，可以比较好的表征说话人的空间。

在GMM-UBM方案的具体实现：
* Training
    1. 用大量不同说话人的数据EM算法训练UBM（Universal background model）模型（多个component的GMM模型）
    2. 针对每一个speaker，在UBM模型的基础上做MAP(Maximum a Posteriori Probability)，得到每一个speaker的模型
* Estimation(Speaker verification)
    1. 对于某条utterance \\(Y\\), 定义：
        * \\(H_0\\): \\(Y\\) 是Speaker \\(S\\) 说的
        * \\(H_1\\): \\(Y\\) 不是Speaker \\(S\\) 说的
    分别在speaker模型和UBM模型上计算似然度，按照如下准则进行判别

    $$
    Score = \frac{Y|H_0}{Y|H_1}\begin{cases}
    \geq \theta \quad accept\  H_0 \\
    < \theta \quad reject\  H_0
    \end{cases}
    $$

    其中 \\(\theta\\) 为判别阈值。
    <p align="center">
    <img src="/images/review_of_sre/1.png" width="75%" height="75%" />
    </p>
    2. 每个speaker的得分分布可能会不一致，因此对得分进行规整（Score Normalization），使得不同speaker服从一致的得分分布，可以进一步提升性能。

* Estimation(Speaker recognition)
与speaker verification类似，对score进行排序，得到nbest。

## Reference
1. [Speaker Verification Using Adapted Gaussian Mixture Models](http://speech.csie.ntu.edu.tw/previous_version/Speaker%20Verification%20Using%20Adapted%20Gaussain%20Mixture%20Models.pdf)


# 3 GMM-SVM
GMM-SVM（Support vector machine）的算法框图如下：
<p align="center">
<img src="/images/review_of_sre/2.png" width="75%" height="75%" />
</p>
其中，GSV(GMM supervector)的生成就是将GMM中所有component的均值堆叠得到。
<p align="center">
<img src="/images/review_of_sre/3.png" width="50%" height="50%" />
</p>

* **Training**
    1. EM训练得到UBM，然后对所有的speaker做MAP，得到每个speaker的GMM模型
    2. 提取每个speaker的supervector，训练SVM分类器
* **Evaluation**
    1. 针对utterance \\(Y\\)，MAP计算得到它对应的supervector
    2. SVM分类器分类得到判决结果

GMM-SVM的提出，主要基于以下考虑：
* GMM是生成模型，SVM是判别模型，而说话人识别本质上是一个判别任务，相对来说SVM更加适合
* 每个GMM都可以用均值、协方差、权重三部分表示，用GMM对说话人建模，其均值矢量必然包含了大量的说话人和信道信息，SVM通过特征的非线性变换，可以更好的提取并区分说话人之间的差异信息。

## Reference
1. [Support Vector Machines using GMM Supervectors for Speaker Verification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.604&rep=rep1&type=pdf)

# 4 JFA（Joint Factor Analysis）
GMM模型可以较好的对说话人进行建模，但它还存在几个问题：
* GMM Supervector 作为一个高维的特征向量，必然包含了很多冗余的信息（高维数据常见的问题）
* 在训练和测试环境一致的情况下，GMM可以取得很好的性能。但由于声音采集过程中，信道具有多样性且容易收到其它噪声的干扰，导致与训练环境不匹配，此时GMM的性能会急剧下降。

而因子分析方法可以很好的将高维冗余复杂的信息简化为较少的足够表征原有观测信息的几个因子，实现数据的降维。针对上述问题，大神 P. Kenny 提出JFA方法用于说话人识别，相比于之前的方案，进一步的提升了性能。

JFA认为，GMM Supervector中建模的差异信息由说话人差异信息和信道差异信息两部分组成，即

$$
M=s+c
$$

其中，\\(s\\) 为说话人相关的超矢量，表示说话人之间的差异。 \\(c\\)为信道相关的超矢量，表示信道之间的差异。 \\(M\\) 为 GMM 超矢量。
<p align="center">
<img src="/images/review_of_sre/4.png" width="25%" height="25%" />
</p>
JFA通过分别对说话人差异和信道差异的子空间建模，这样就可以去除信道的干扰，得到对说话人的更精确的描述。

## 4.1 JFA算法详解
对于给定的某个speaker的超矢量 \\(s\\)，将其按如下分解
<p align="center">
<img src="/images/review_of_sre/5.png" width="75%" height="75%" />
</p>
其中：
* \\(m\\)：说话人无关超矢量（UBM超矢量）
* \\(V\\): 本征音矩阵
* \\(U\\): 本征信道矩阵
* \\(D\\): 残差矩阵（且为对角阵）
* \\(y\\): 说话人因子, 服从\\(N(0,1)\\)先验分布
* \\(x\\): 信道因子，服从\\(N(0,1)\\)先验分布
* \\(z\\): 说话人相关的残差因子，服从\\(N(0,1)\\)先验分布

### 4.1.1 Training
JFA的训练过程主要分为以下三步：
1. 假设 \\(U\\) 和 \\(D\\) 都为 \\(0\\), 训练本征音矩阵 \\(V\\)
2. 根据训练好的本征音矩阵 \\(V\\)，假设 \\(D\\) 为 \\(0\\)，训练本征信道矩阵 \\(D\\)
3. 根据训练好的 \\(V\\) 和 \\(U\\)，训练残差矩阵 \\(D\\)


### 4.1.2 Score
Building

# More ...

##### Reference
1. [JFA tutorial](http://www1.icsi.berkeley.edu/Speech/presentations/AFRL_ICSI_visit2_JFA_tutorial_icsitalk.pdf)
2. [Eigenvoice Modeling With Sparse Training Data](https://www.crim.ca/perso/patrick.kenny/eigenvoices.PDF?)
3. [A Study of Inter-Speaker Variability in Speaker Verification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.494.6825&rep=rep1&type=pdf)
4. [Joint Factor Analysis of Speaker and Session Variability: Theory and Algorithms](https://www.crim.ca/perso/patrick.kenny/FAtheory.pdf)
5. [Joint Factor Analysis versus Eigenchannels in Speaker Recognition](https://www.crim.ca/perso/patrick.kenny/FASysJ.pdf)
