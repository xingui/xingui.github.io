---
layout: post
title: 声纹识别的技术演变（四）
date: 2019-05-10
categories: 声纹识别
tags: [声纹识别, DNN, Bottleneck Feature]
grammar_cjkRuby: true
---

当DNN用于声纹识别的时候，它的发展路线在很多方面与语音识别很相似，最先演化出来的是bottleneck feature，然后是用DNN替换GMM，再演变到用DNN embedding。

# 1 Bottleneck Feature
![94b64652d689a08c798405e6738b1a61.png](evernotecid://B4A2B645-AE29-4E6B-99EF-D269D3FF3BF3/wwwevernotecom/17185513/ENResource/p2388)@w=500

首先需要训练一个Bottleneck 网络，网络输入为语音信号特征（e.g MFCC, FBank），输出为speaker 类别，训练完成后将Bottleneck层的输出作为语音信号的特征，然后再用传统的方案（如GMM-UBM，i-vector）建模识别。

<p align="center">
<img src="/images/review_of_sre/6.png" width="75%" height="75%" />
</p>
    
## 1.1 实现
典型的实现如Reference [1]:
1. **论文中采用交叉熵训练准则的变种（避免对nontarget speaker的依赖）**：
    ```math
    J_{LLR}=\alpha \sum_T log(1+e^{-u_T-c})+\beta \sum_N log(1+e^{u_N+c})
    ```
    其中 $u_T$ 和 $u_N$ 分别表示target 和 nontarget speaker的score，即最后一层的输出。另外
    ```math
    \begin{aligned}
    c &= \frac{\pi}{1-\pi}   \\
    \pi &= \frac{P_{target}\times C_{miss}}{P_{target}\times C_{miss} + (1-P_{target})\times C_{FA}}  \\
    \alpha &=\frac{\pi}{N_{target}} \\
    \beta &=\frac{1-\pi}{N_{nontarget}}
    \end{aligned}
    ```
    其中 $N_{target}$ 和 $N_{nontarget}$ 分别表示target speaker 和 nontarget speaker 的数量。$C_{miss}$ 表示miss error，论文取值为10， $C_{FA}$为false alarm error，取值为1， $P_{target}=0.01$.

    ***Note: 对新的损失函数的理解***
    损失函数同时考虑了正确类别和错误类别上的输出，目的是**提升在正确类别上的概率，同时抑制在单个错误类别上产生较大概率（避免对nontarget speaker的依赖）**。


2. **Recording-level training**
    采用recording-level的损失函数如下：
    ```math
    u = E_C[u^L]
    ```
    其中，$E_C$ 表示Utterence C的期望算子， $u^L$是最后一层网络的输出。

3. **类Teacher-student training**
    ![53f4f70d0a6bd576c37ba0e98f1d86e1.png](evernotecid://B4A2B645-AE29-4E6B-99EF-D269D3FF3BF3/wwwevernotecom/17185513/ENResource/p2389)@w=450

    假设bottleneck 网络的输出为$u_n(\Theta)$, baseline GMM-UBM的speaker score 为$u_n^M$，将上述两个score按如下公式融合作为网络的输出，参与loss的计算：
    ```math
    u_n'(\Theta)=\omega_Bu_n(\Theta)+\omega_Mu_n^M+\beta
    ```
    其中$\Theta$为网络参数。
    **增加$u_n^M$的目的是为了让网络可以区分「easy」和「difficult」样本，从而提升「difficult」样本的分类能力。**

    训练时，采用类似EM的训练方法
    * 首先固定$\Theat$，估计$\omega_B, \omega_M, \beta$
    ```math
    \omega_B^{*},\omega_M^{*},\beta^*=\arg min_{\omega_B, \omega_M, \beta} J_{LLR}(\omega_B, \omega_M, \beta|\Theta_{fixed})
    ```
    * 更新$\Theta$
    ```math
    \Theta^* =\arg min_{\Theta} J_{LLR}(\Theta| \omega_B, \omega_M, \beta)
    ```
    
采用上述方案后，**在同mic/不同mic的场景下，EER可以分别获得14%和18%的下降。**

## 1.2 利弊分析
* 性能相比于传统方案有所提升
* 未脱离传统架构，系统结构复杂，尤其训练过程较为繁琐

# 2 DNN-Ivector
## 2.1 实现
DNN代替GMM在语音识别中取得了比GMM-HMM更好的效果，suppose 在说话人识别中代替GMM也能取得更好的效果。
1. 在GMM-ivector方案中，ivector的训练和提取主要依赖于充分统计量的计算，即
$$
\begin{aligned}
N_k^{(i)} &= \sum_t \gamma_{kt}^{(i)}   \\
F_k^{(i)} &= \sum_t \gamma_{kt}^{(i)}x_t^{(i)}   \\
S_k^{(i)} &= \sum_t \gamma_{kt}^{(i)} x_t^{(i)}x_t^{{(i)}^T}
\end{aligned}
$$

2. 其中，GMM主要用于计算输入特征在每个component上的后验概率，如果DNN可以提供这个功能，就可以顺理成章的替换掉GMM。
![31cebbb8d7197969647db87d7608fd59.png](evernotecid://B4A2B645-AE29-4E6B-99EF-D269D3FF3BF3/wwwevernotecom/17185513/ENResource/p2390)@w=500
    

因此，可以用一个 supervised DNN 用于替换GMM，DNN 以senone作为建模单元，每个senone等价于GMM中的一个component，按照语音识别DNN的训练方法训练该DNN。
![3fc6be80a6b506d4a7aa09ff50c8c90b.png](evernotecid://B4A2B645-AE29-4E6B-99EF-D269D3FF3BF3/wwwevernotecom/17185513/ENResource/p2391)@w=500
1. 对senone进行建模有什么好处？
    > 在GMM-ivector中，GMM对全局的语音信号进行建模。假设某个speaker在说 /aa/ 时，由于口音问题，它可能会对齐到 /ao/ 的高斯中，此时经过MAP，它影响的是 /ao/所对应高斯的均值，而 /aa/ 对应的高斯均值影响很小，那么**在提取ivector的时候，ivector中将不再包含speaker所说的 /aa/的信息，而对senone进行建模即可规避这个问题。**
2. 是否可以采用supervised UBM？
    > 可以。先使用ASR做force alignment，用seg之后的数据训练GMM，即可得到supervised UBM。**论文中实验发现，性能相比GMM-ivector有所下降，分析觉得是GMM对senone建模的精度太差。**
        
在 Reference[3]中，**作者采用TDNN来代替GMM计算后验概率，得到充分统计量，取得了50%的EER下降。在保持同样计算量的条件下，获得了20%的EER下降。** 这得益于TDNN的两大优点：
1. TDNN可以有更长的context，可以看到更多的前后信息
2. TDNN在解码过程中，历史结果可以复用，从而大幅降低计算量。

## 2.2 分析
* **相比于GMM，DNN有更好的性能的原因在于DNN提供了更加精准的后验概率。**（本质上，GMM和DNN都是对整个特征空间进行建模，只是两者对全局的特征空间进行了不同的划分）


### Reference
*[1] [Bottleneck Features for Speaker Recognition](https://pdfs.semanticscholar.org/3469/fe6e53e65bced5736480afe34b6c16728408.pdf)*
*[2] [NOVEL SCHEME FOR SPEAKER RECOGNITION USING A PHONETICALLY-AWARE DEEP NEURAL NETWORK](http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202014/papers/p1714-lei.pdf)*


