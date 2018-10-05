---
layout: post
title: "Paper Note Improved Signal-to-Noise Ratio Estimation for Speech Enhancement"
date: 2018-10-05
categories: 论文笔记
tags: [TSNR, DD, 语音信号处理]
grammar_cjkRuby: true
---

# 1. 背景
对于加性噪声模型，带噪信号可以表⽰为 

$$
    x(t) = s(t)+n(t)
$$

其中\\(s(t)\\)表示语音信号，\\(n(t)\\)表示噪声信号。用\\(S(p,k)\\),\\(N(p,k)\\)和\\(X(p,k)\\)分别表示\\(s(t)\\),\\(n(t)\\)和\\(x(t)\\)的第\\(p\\)个短时帧的第\\(k\\)个频率分量。

**去噪的目的：**find an estimator \\(\hat{S}(p,k)\\) which minimizes the expected value of a given distortion measure conditionally to a set of spectral noisy features.     
**去噪的方法：**An estimate of \\(S(p,k)\\) is subsequently obtained by applying a spectral gain \\(G(p,k)\\) to each short-time spectral component \\(X(p,k)\\), The choice of the distortion measure determines the gain behavior, i.e. the trade-off between noise reduction and speech distortion.

对于大部分的语音增强技术都需要估计两个参数：后验SNR(posteriori SNR)和先验SNR(priori SNR)，他们的定义如下：

$$\begin{align}
    SNR_{post}(p,k) = \dfrac{|X(p,k)|^2}{E[|N(p,k)|^2]}
\end{align}$$

$$
\begin{align}                                                                                                                                                                                               
    SNR_{prio}(p,k) = \dfrac{E[|S(p,k)|^2]}{E[|N(p,k)|^2]}
\end{align}
$$

其中，\\(E[.]\\)是期望算子。

定义瞬时SNR(instantaneous SNR)(它可以认为是局部先验SNR的直接估计) 如下：

$$
\begin{align}
\begin{split}
SNR_{inst}(p,k) &= \dfrac{|X(p,k)|^2-E[|S(p,k)|^2]}{E[|N(p,k)|^2]}   \\
                &=SNR_{post}(p,k) -1
\end{split}
\end{align}
$$

在实际应用中，语音信号的能量谱密度(power spectrum density, PSD) \\(E[\|S(p,k)\|^2]\\) 和噪声的能量谱密度\\(E[\|N(p,k)\|^2]\\)都是未知的，只有带噪语音信号\\(X(p,k)\\)是已知的，因此，先验SNR和后验SNR都要估计。

对于噪声的能量谱密度(记为：\\({\hat{\gamma}}_{n}(p,k) \\) )，可以在背景噪声的信号段内进行估计(It can be practically estimated during speech pauses using a classic recursive relation or continuously using the Minimum Statistics or the Minima Controlled Recursive Averaging approach to get a more accurate estimate in case of noise level fluctuations)。

频谱增益可由以下函数得到：

$$
\begin{align}
  G(p,k)=g(\hat{SNR}_{prio}(p,k), \hat{SNR}_{post}(p,k))
\end{align}
$$

\\(g\\)可以选择不同的函数（e.g. amplitude or power spectral subtraction, Wiener filtering, MMSE STSA, MMSE LSA, OM LSA, etc.）

由此得到语音信号的估计：

$$
\begin{align}
  \hat{S}(p,k)=G(p,k)X(p,k)
\end{align}
$$

直观理解，通过 SNR 来计算增益，SNR 越大的地方，增益越大，反之增益越小.

# 2. Decision-Directed Approach
根据估计的噪声能量谱密度，得到先验 SNR 和后验 SNR如下：

$$
 \begin{align}
   \hat{SNR}_{post}(p,k) = \dfrac{|X(p,k)|^2}{\hat{\gamma}_n(p,k)}
 \end{align}
 $$
 
 $$
 \begin{align}
   \hat{SNR}_{pori}^{DD}(p,k) = \beta\dfrac{|\hat{S}(p-1,k)|^2}{\hat{\gamma}_n(p,k)} + (1-\beta)P[\hat{SNR}_{post}(p,k)-1]
 \end{align}
 $$
 
 其中，\\(P[.]\\) denotes the half-wave rectification, \\(\beta\\)一般取\\(0.98\\).

 选择维纳滤波器作为增益函数，那么有
 
 $$
 \begin{align}
   G_{DD}(p,k) = \dfrac{\hat{SNR}_{prio}^{DD}(p,k)}{1+\hat{SNR}_{prio}^{DD}(p,k)}
 \end{align}
$$

 分析 DD 算法，有
* 当瞬时 SNR 远大于0时，\\(\hat{SNR}_{pori}^{DD}(p,k)\\)相对于瞬时 SNR 总会有一帧的延迟。(因为\\(\beta\\)接近1)
* 当瞬时 SNR 接近于或小于0时，\\(\hat{SNR}_{pori}^{DD}(p,k)\\)是瞬时 SNR 的高度平滑，它的方差要比瞬时 SNR 更小，从而得到较好的降噪效果。
 * 在语音信号刚出现或恰好消失的边缘帧，会出现先验 SNR 过估计和欠估计的情况。

# 3. Two step noisy reduction(TSNR)
当\\(\beta\\)接近于1时，DD 算法引入了一帧的延迟，也就是说，当前帧计算的是上一帧的频谱增益。因此，我们可以用下一帧的频谱增益来估计当前帧的语音信号。因此 TSNR 的算法如下：
* 使用 DD 算法计算第\\(p\\)帧的频谱增益\\(G_{DD}(p,k)\\)
* 使用上一步计算的频谱增益和\\(p+1\\)帧来估计当前帧的先验 SNR

$$
    \begin{align}
      \hat{SNR}_{prio}^{TSNR}(p,k) &= \hat{SNR}_{prio}^{DD}(p+1,k)  \\
      &= {\beta}^{'} \dfrac{|G_{DD}(p,k)X(p,k)|^2}{\hat{\gamma}_n(p,k)}+(1-{\beta}^{'})P[\hat{SNR}_{post}(p+1,k)-1]
    \end{align}
$$
	
考虑到\\(\hat{SNR}_{post}(p+1,k)\\)需要\\(p+1\\)帧进行估计，这会引入额外的处理延迟，因此取\\({\beta}^{'}=1\\)，有

$$
\begin{align}
  \hat{SNR}_{prio}^{TSNR}(p,k) = {\beta}^{'} \dfrac{|G_{DD}(p,k)X(p,k)|^2}{\hat{\gamma}_n(p,k)}
\end{align}
$$

The choice of \\({\beta}^{'}=1\\) is valid only for the second step in order to refine the first step estimation: actually \\({\beta}\\) is set to a typical value of \\(0.98\\) for the first step.

依然采用维纳滤波器，那么频谱增益为

$$
 \begin{align}
   G_{TSNR}(p,k) = \dfrac{\hat{SNR}_{prio}^{TSNR}(p,k)}{1+\hat{SNR}_{prio}^{TSNR}(p,k)}
 \end{align}
 $$
 
因此，最终的语音信号估计为

$$
\begin{align}
  \hat{S}(p,k)=G_{TSNR}(p,k)X(p,k)
\end{align}
$$