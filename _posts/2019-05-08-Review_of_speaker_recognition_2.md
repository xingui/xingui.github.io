---
layout: post
title: 声纹识别的技术演变(二)
date: 2019-05-08
categories: 声纹识别
tags: [声纹识别, JFA]
grammar_cjkRuby: true
---

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

# 1 JFA算法详解
对于给定的某个speaker的超矢量 \\(s\\)，将其按如下分解
<p align="center">
<img src="/images/review_of_sre/5.png" width="75%" height="75%" />
</p>
其中：
* \\(m\\)：说话人无关超矢量（UBM超矢量）
* \\(V\\): 本征音矩阵，低秩
* \\(U\\): 本征信道矩阵，低秩
* \\(D\\): 残差矩阵（且为对角阵）
* \\(y\\): 说话人因子, 服从\\(N(0,1)\\)先验分布
* \\(x\\): 信道因子，服从\\(N(0,1)\\)先验分布
* \\(z\\): 说话人相关的残差因子，服从\\(N(0,1)\\)先验分布

## 1.1 Training
JFA的训练过程主要分为以下三步：
1. 假设 \\(U\\) 和 \\(D\\) 都为 \\(0\\), 训练本征音矩阵 \\(V\\)
2. 根据训练好的本征音矩阵 \\(V\\)，假设 \\(D\\) 为 \\(0\\)，训练本征信道矩阵 \\(D\\)
3. 根据训练好的 \\(V\\) 和 \\(U\\)，训练残差矩阵 \\(D\\)

### 1.1.1 训练本征音矩阵\\(V\\)
对于本征音矩阵 \\(V\\) 的训练，分为下面几步：
1. 针对每个speaker \\(s\\) 和 GMM component \\(c\\)，对所有的utterance \\(Y\\) 计算其充分统计量（Baum-Welch统计量）
    * 0阶统计量

    $$
    N_c(s)=\sum_{t\in s}\gamma_{t}(c)
    $$

    其中 \\(\gamma_{t}(c)\\) 为 \\(Y_t\\)帧在speaker \\(s\\)的GMM模型的第 \\(c\\) 个component上的后验概率
    * 1阶统计量

    $$
    F_c(s)=\sum_{t\in s}\gamma_{t}(c)Y_t
    $$

    * 2阶统计量

    $$
    S_c(s)=diag (\sum_{t\in s}\gamma_{t}(c)Y_tY_t^T)
    $$

    只保留对角线上的元素，使得\\(S_c(s)\\)为对角阵。
    
2. 计算1阶和2阶中心统计量

    $$
    \begin{aligned}
        \tilde{F_c}(s)&=F_c(s) - N_c(s)m_c  \\
        \tilde{S_c}(s)&=S_c(s) - diag (F_c(s)m_c^T+m_cF_c(s)^T-N_c(s)m_cm_c^T)
    \end{aligned}
    $$

    其中\\(m_c\\)是UBM模型中第\\(c\\)个component的均值。

3. 将所有component的上述统计量合并到一个大的矩阵中

    $$
    \begin{aligned}
        NN(s) &=\begin{bmatrix}
                    N_1(s)*I & & \\
                     & \ddots &  \\
                    & & N_C(s)*I
                \end{bmatrix}  \\
        FF(s) &=\begin{bmatrix}
                    \tilde{F_1}(s) \\
                    \vdots  \\
                    \tilde{F_C}(s)
                \end{bmatrix}  \\
        SS(s) &=\begin{bmatrix}
                    \tilde{S_1}(s) & & \\
                     & \ddots &  \\
                    & & \tilde{S_C}(s)
                \end{bmatrix}
    \end{aligned}
    $$

    其中\\(C\\)为GMM的component数量，\\(I\\)为单位矩阵。
4. E-step, 初始化说话人因子\\(y\\)
    令

    $$
    l_V(s)=I+V^T*\Sigma^{-1}*NN(s)*V
    $$

    可以导出\\(y(s)\\)服从如下分布

    $$
    y(s) \sim N(l_v^{-1}(s)*V^T*\Sigma^{-1}*FF(s), l_V^{-1}(s))
    $$

    也就是说\\(y(s)\\)的均值为

    $$
    \bar y(s) = l_v^{-1}(s)*V^T*\Sigma^{-1}*FF(s)
    $$

    其中\\(\Sigma\\)为UBM模型的协方差矩阵，上述公式的详细推导见 【Reference 2】.

5. M-step. 最大似然重估

    $$
    \begin{aligned}
    N_c &= \sum_s N_c(s) \\
    A_c &= \sum_s N_c(s)l_V^{-1}(s) \\
    \Bbb{C}&=\sum_s FF(s)*(l_V^{-1}(s)*V^T*\Sigma^{-1}*FF(s))^T  \\
    NN&=\sum_s NN(s)
    \end{aligned}
    $$

6. 估计\\(V\\)矩阵

    $$
    V=\begin{bmatrix}
    V_1 \\
    \vdots \\
    V_C
    \end{bmatrix} = \begin{bmatrix}
    A_1^{-1}*\Bbb{C}_1^T \\
    \vdots \\
    A_C^{-1}*\Bbb{C}_C^T
    \end{bmatrix}
    $$


7. 更新协方差矩阵\\(\Sigma\\)（可选）

    $$
    \Sigma=NN^{-1}\bigg(\big(\sum_s SS(s)\big)- diag (\Bbb{C}*V^T)\bigg)
    $$

8. 迭代STEPS 4~6（或4~7），即可得到本征音矩阵\\(V\\)，将其代入step 4，即可得到说话人因子。

### 1.1.2 训练本征信道矩阵\\(U\\)
本征信道矩阵\\(U\\)的训练与本征音矩阵的训练过程基本一致，唯一的不同在于计算中心统计量时需要减除说话人因子的影响，即

$$
\begin{aligned}
    \tilde{F_c}(s)&=F_c(s) - N_c(s)m_c - N_c(s)*V*y_c(s)\\
\end{aligned}
$$


### 1.1.3 训练残差矩阵\\(D\\)
残差矩阵\\(D\\)的训练与前述训练过程也基本一致，只有两个不同的地方：
1. 计算中心统计量时需要减除说话人因子和信道因子的影响，即

    $$
    \begin{aligned}
        \tilde{F_c}(s)&=F_c(s) - N_c(s)m_c - N_c(s)*V*y_c(s) - N_c(s)*U*x_c(s)\\
    \end{aligned}
    $$

2. 由于\\(D\\)时对角阵，因此在M-Step更新矩阵\\(A_c\\)和\\(\Bbb{C}\\)时，需要取其对角元素构成对角阵，即

    $$
    \begin{aligned}
    A_c &= \sum_s diag(N_c(s)l_V^{-1}(s)) \\
    \Bbb{C}&=\sum_s diag(FF(s)*(l_V^{-1}(s)*V^T*\Sigma^{-1}*FF(s))^T)  \\
    \end{aligned}
    $$

## 1.2 Evaluate
在估计的时候，得分的计算采用马氏距离，如下：

$$
Score = (V*y(target) + D*z(target))^T * \Sigma^{-1} *(FF(test)-NN(test)*m-NN(test)*U*x(test))
$$

## Reference
1. [JFA tutorial](http://www1.icsi.berkeley.edu/Speech/presentations/AFRL_ICSI_visit2_JFA_tutorial_icsitalk.pdf)
2. [Eigenvoice Modeling With Sparse Training Data](https://www.crim.ca/perso/patrick.kenny/eigenvoices.PDF?)
3. [A Study of Inter-Speaker Variability in Speaker Verification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.494.6825&rep=rep1&type=pdf)
4. [Joint Factor Analysis of Speaker and Session Variability: Theory and Algorithms](https://www.crim.ca/perso/patrick.kenny/FAtheory.pdf)
5. [Joint Factor Analysis versus Eigenchannels in Speaker Recognition](https://www.crim.ca/perso/patrick.kenny/FASysJ.pdf)
   
