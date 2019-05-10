---
layout: post
title: 声纹识别的技术演变（三）
date: 2019-05-09
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
PLDA是一个概率生成模型，为了解决人脸识别的问题而提出。后被广泛应用的声纹领域，并产生了多重变种。主要有以下三种：

* **Standard**

    $$
		\begin{align*}
		\phi_{ij} &= \mu + Vy_i + Ux_{ij} + \epsilon_{ij}	\\
		y_i &\sim N(0, I)	\\
		x_{ij} &\sim N(0, I)	\\
		\epsilon_{ij} &\sim N(0, \Lambda^{-1})	\\
		\end{align*}
    $$

* **Simplified**

    $$
		\begin{align*}
		\phi_{ij} &= \mu + Sy_i + \epsilon_{ij}	\\
		y_i &\sim N(0, I)	\\
		\epsilon_{ij} &\sim N(0, \Lambda_f^{-1})	\\
		\end{align*}
    $$

* **Two Covariance**

    $$
        \begin{align*}
        y_i &\sim N(y_i | \mu, B^{-1})	\\
        \psi_{ij}|y_i &\sim N(\psi_{ij} | y_i, W^{-1})	\\
        \end{align*}
    $$

Reference [6]中将上述三种PLDA统一到了同一框架中，并实验验证Two-Covariance对声纹具有最好的性能

本文的讨论的主要是Reference [6]中提出的PLDA(Two Covariance PLDA，kaldi中采用的版本). 在介绍PLDA前，先简单介绍下LDA.

### 2.1 LDA
LDA 假设数据服从高斯分布，并且各类的协方差相同。各类的先验概率为\\(\pi_k\\)，且

$$
	\sum_{k=1}^K \pi_k = 1
$$

各类的概率分布为

$$
	P(x|k) \sim N(\pi_k, \Sigma)
$$

对于观测数据，其类别的后验概率为

$$
	P(k|x) = \frac{P(x|k)\pi_k)}{P(x)}
$$

为了对数据进行分类，计算各类后验的似然比

$$
\begin{aligned}
	ln \frac{P(k|x)}{P(l|x)} &= ln \frac{P(x|k)}{P(x|l)} + ln \frac{\pi_k}{\pi_l} \\
	                                 &= ln \frac{\pi_k}{\pi_l} - \frac{1}{2}(\mu_k - \mu_l)^T\Sigma^{-1}(\mu_k + \mu_l) - x^T\Sigma^{-1}(\mu_k-\mu_l)
\end{aligned}
$$

由于假设协方差相同，似然比是关于输入 \\(x\\) 的线性函数。LDA 用一系列超平面划分数据空间，进而完成分类。 如下图
<p align="center">
<img src="/images/review_of_sre/7.jpg" width="75%" height="75%" />
</p>

LDA还可以对数据进行降维，但它无法出类训练数据中未出现的类别数据。

### 2.2 Two Covariance PLDA
如果把类别\\(\boldsymbol{y}\\)当作隐变量，对其进行建模，就可以处理未知类别的样本数据。

$$
\begin{aligned}
	P(x|y) &= N(x|y, \Phi_w) \\
	P(y) &= N(y|m, \Phi_b)
\end{aligned}
$$

其中
* \\(\Phi_w\\)为正定矩阵，反映了类内差异
* \\(\Phi_b\\)为半正定矩阵，反映了类间差异

此时可以对\\(\Phi_w\\)和\\(\Phi_b\\)做合同对角化，即

$$
\begin{aligned}
	V^T\Phi_b V &=\Psi \\
	V^T \Phi_w V &= I \\
\end{aligned}
$$

其中 \\(\Psi\\)为对角阵，\\(I\\)为单位阵。

令\\(A = V^{-1}\\), PLDA可以等价为

$$
\begin{aligned}
x &= m + Au \\
u &\sim N(v, I) \\
v &\sim N(0, \Psi)
\end{aligned}
$$

其中，\\(\Psi\\)反映类间差异，\\(I\\)反映类内差异

#### 2.2.1 Evaluation
对每个观测变量\\(x\\)，可以先做变换得到

$$
u = A^{-1}(x-m)
$$

对给定的一组同类观测数据\\(u_1, ..., u_n\\)，类别\\(v\\)的后验概率为

$$
	P(v|u_1, ..., u_n) = N(v|\frac{n\Psi}{n\Psi + I}\bar{u}, \quad\frac{\Psi}{n\Psi + I})
$$

其中\\(\bar{u} = \frac{1}{n}\sum_{i=1}^{n} u_i\\)。

因此，对于未知的数据点\\(u^p\\)和已知某类的若干数据点\\(u_1^g, ..., u_n^g\\), \\(u^p\\)的属于该类的概率为

$$
\begin{aligned}
P(u^p|u_1^g, ..., u_n^g) &= p(u^p|v)p(v|u_1^g, ..., u_n^g) \\
									&=N(u^p|v, I) N(v|\frac{n\Psi}{n\Psi + I}\bar{u}, \quad\frac{\Psi}{n\Psi + I}) \\
									& = N(\frac{n\Psi}{n\Psi + I}\bar{u}, \quad\frac{\Psi}{n\Psi + I}+I)
\end{aligned}
$$

$$
\ln P(u^p|u_1^g, ..., u_n^g) = C -\frac{1}{2}(u^p-\frac{n\Psi}{n\Psi+I}\bar{u}^g)^T(\frac{\Psi}{n\Psi+I}+I)^{-1}(u^p-\frac{n\Psi}{n\Psi+I}\bar{u}^g)-\frac{1}{2}\ln|\frac{\Psi}{n\Psi+I}+I|
$$

\\(u^p\\)不属于任何类的概率为

$$
P(u^p|\Phi) = N(u^p|0, \quad \Psi+I)
$$

$$
\ln P(u^p|\Phi) = C - \frac{1}{2}{u^p}^T (\Psi+I)^{-1}u^p - \frac{1}{2}\ln|\Psi+I|
$$

其中\\(C = - \frac{1}{2} d\ln 2 \pi\\)为常量，与数据无关，\\(d\\)为数据的维度。

因此，利用PLDA做说话人识别时，
* ***对于说话人识别，计算最大似然度***

	$$
		i = arg max_i ln P(u^p|u_1^{g_i}, ..., u_n^{g_i})
	$$

* **对于说话人验证，计算似然比**

$$
	ln R =  ln P(u^p|u_1^{g}, ..., u_n^{g}) - P(u^p|Phi)
$$

更多关于PLDA打分的内容可参考Reference [8]

#### 2.2.2 Training
PLDA中，需要估计的参数有\\(A\\), \\(\Psi\\), \\(m\\)。
* 对于每类样本数相同(若样本数不一致，可做上采样)的场景，可以直接求解。可参考【Reference 6】
* EM算法，参考【Reference 7】

**之后有时间再补上**

##### 2.3 Length Normalization
经过PLDA变换之后，我们一般还有在做Length Normalization，可以进一步提升性能，为何要做Length Normalization？主要基于以下两点考虑：
* PLDA基于高斯假设
* 样本较少时数据服从学生$t$分布，通过whithen和 Length Norm进行补偿。

那么Length Normalization 是如何做的呢？主要分为两步：
1. Centering and whitening(PLDA变换中已完成)

$$
			u = A^{-1}(x-m)
$$

2. Scaling

$$
			u_{ln} = \frac{u}{||u||}
$$

or (Kaldi中采用的方式)

$$
			u_{ln} = \frac{u(\frac{\Psi}{n} + I)}{||u||^2}
$$

更多可参考【Reference 5,9】

## Reference
*[1] [Front-End Factor Analysis for Speaker Verification](http://habla.dc.uba.ar/gravano/ith-2014/presentaciones/Dehak_et_al_2010.pdf)*    
*[2] [A Straightforward and Efficient Implementation of the Factor Analysis Model for Speaker Verification](http://mistral.univ-avignon.fr/doc/publis/07_Interspeech_Matrouf.pdf)*     
*[3] [A Small Footprint i-Vector Extractor](https://www.isca-speech.org/archive/odyssey_2012/papers/od12_001.pdf)*     
*[4] [Probabilistic Linear Discriminant Analysis for Inferences About Identity](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester2201112/prince-iccv07-plda.pdf)*    
*[5] [Analysis of I-vector Length Normalization in Speaker Recognition Systems](https://isca-speech.org/archive/archive_papers/interspeech_2011/i11_0249.pdf)*     
*[6] [Probabilistic Linear Discriminant Analysis](http://people.irisa.fr/Guillaume.Gravier/ADM/articles2018/Probabilistic_Linear_Discriminant_Analysis.pdf)*      
*[7] [Unifying Probabilistic Linear Discriminant Analysis Variants in Biometric Authentication](http://cs.uef.fi/~sizov/pdf/unifying_PLDA_ssspr2014.pdf)*      
*[8] [From single to multiple enrollment i-vectors: Practical PLDA scoring variants for speaker verification](https://www.sciencedirect.com/science/article/pii/S1051200414001377)*     
*[9] [ROBUST SPEAKER RECOGNITION BASED ON LATENT VARIABLE MODELS](https://drum.lib.umd.edu/bitstream/handle/1903/13092/GarciaRomero_umd_0117E_13566.pdf?sequence=1&isAllowed=y)*      

