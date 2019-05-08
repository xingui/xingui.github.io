---
layout: post
title: 声纹识别的技术演变（三）
date: 2019-04-10
categories: 声纹识别
tags: [声纹识别, GMM, i-vector, PLDA]
grammar_cjkRuby: true
---

JFA方法的思想是使用GMM超矢量空间的子空间对说话人差异和信道差异分别建模，从而可以方便的分类出信道干扰。然而，Dehak注意到，在JFA模型中，信道因子中也会携带部分说话人的信息，在进行补偿时，会损失一部分说话人信息。

所以Dehak提出了全局差异空间模型，将说话人差异和信道差异作为一个整体进行建模。这种方法相比JFA具有以下优点：
1. 改善了JFA对训练语料的要求
2. 计算复杂度相比JFA更低，可以应对更大规模的数据
3. 性能与JFA相当，结合PLDA的信道补偿，其信道鲁棒性更强。

# 1 I-vector算法原理
对于给定的utterance，其GMM supervector可分解为：

$$
M=m+Tw
$$

其中：
* \\(M\\): GMM Supervector
* \\(m\\)：说话人无关超矢量（UBM超矢量）
* \\(T\\): 全局差异空间矩阵，低秩
* \\(w\\): 全局差异因子，它的后验均值即为i-vector矢量，服从\\(N(0,1)\\)先验分布
本征信道矩阵，低秩

## 1.1 Training
**对于全局差异空间矩阵的训练，与JFA中本征音矩阵的训练一致。**

## 1.2 Evalution
对于i-vector，可以采用余弦距离（cosine distance）对目标speaker和测试speaker的判决打分：

$$
Score(w_{target}, w_{test}) = \frac{<w_{target},w_{test}>}{||w_{target}||||w_{test}||}\begin{cases}\geq \theta \quad accept \\
< \theta \quad reject\end{cases}
$$


其识别框架如下：
<p align="center">
<img src="/images/review_of_sre/6.png" width="75%" height="75%" />
</p>
    
## 2 PLDA
Building

## Reference
1. Ivector理论：[Front-End Factor Analysis for Speaker Verification](http://habla.dc.uba.ar/gravano/ith-2014/presentaciones/Dehak_et_al_2010.pdf)
2. 训练算法：[A Straightforward and Efficient Implementation of the Factor Analysis Model for Speaker Verification](http://mistral.univ-avignon.fr/doc/publis/07_Interspeech_Matrouf.pdf)
3. [A Small Footprint i-Vector Extractor](https://www.isca-speech.org/archive/odyssey_2012/papers/od12_001.pdf)
4. PLDA理论：[Probabilistic Linear Discriminant Analysis for Inferences About Identity](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester2201112/prince-iccv07-plda.pdf)
5. PLDA打分：[Analysis of I-vector Length Normalization in Speaker Recognition Systems](https://isca-speech.org/archive/archive_papers/interspeech_2011/i11_0249.pdf)
6. HT-PLDA: [Bayesian Speaker Verification with Heavy-Tailed Priors](https://www.crim.ca/perso/patrick.kenny/kenny_Odyssey2010.pdf)

