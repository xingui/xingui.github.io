---
layout: post
title: Awesome Speaker Recognition
date: 2019-12-31
categories: 声纹识别
tags: [声纹识别]
grammar_cjkRuby: true
---
## Awesome Speaker Recognition

### Table of contents
* [Overview](#Overview)
* [Publications](#Publications)
* [Software](#Software)
* [Datasets](#Datasets)
* [Leaderboards](#Leaderboards)
* [Other learning materials](#Other-learning-materials)
* [Products](#Products)

### Overview
本文用于记录一些经典的说话人识别相关的论文、数据集、开源软件工具等资源，便于后续的回顾及使用。

本文会长期持续更新。。。

### Publications

| Time | Paper | Paper Note | Abstract |
|---|---|---|---|
|**2019**|[X-vector DNN Refinement with Full-length Recordings for Speaker Recognition](http://www.danielpovey.com/files/2019_interspeech_xvector_refinement.pdf)|【[论文笔记](https://www.evernote.com/l/AJRM8V8b8eJMD7XMFx0KfGXn2Ceep_0iZ9E)】|ETDNN + AMSoftmax + full-length recording refinement，采用cosine metrics，SWIT数据集上实现~28%的EER相对下降。|
||[Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification](https://arxiv.org/pdf/1903.12058.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/19156c1b-1da4-4bbe-9d84-847c81a89bef/d2789e3d64762067d4b84a3cc44ee871)】|多任务学习将输入信号的高阶统计量encoding到speaker embedding中，实验证明该方法能给轻微提高文本无关的说话人识别性能。|
||[An End-to-End Text-independent Speaker Verification Framework with a Keyword Adversarial Network](https://arxiv.org/pdf/1908.02612.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/dffe24c6-cc66-40d6-9087-3b0d9f4e397e/ede6b6715829e924c776bfc5a420db01)】|针对说话人确认这个任务，为了提升文本无关场景下的性能，作者引入ASR adversarial network，使得SE(speaker embedding)网络生成更好的文本无关embedding，实验证明该方法能给大幅提高文本无关的说话人识别性能。|
|**2018**|[Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification](http://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/51f5b7dd-3e75-4ee2-812b-1aa74a7b6bf0/3737a6dca83e772284ba65e60971dc15)】|作者在相关笔记xvector 的基础上，将average pooling 层替换为self-attention层，实验结果显示self-attention在不同duration的场景下都取得了一定性能的提升。|
||[X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/79908c1f-999c-4c9f-96d4-33f9d619244c/c91ae505146fc474db7639679de2a063)】|采用TDNN提取embedding，然后使用PLDA作为打分后端，相比ivector+PLDA，在性能上可以得到大幅提升。另外，作者还提出使用数据增广的方式，进一步提升性能。|
|**2017**|[TRISTOUNET: TRIPLET LOSS FOR SPEAKER TURN EMBEDDING](http://cn.arxiv.org/pdf/1609.04301v3)|||
||[Deep Speaker: an End-to-End Neural Speaker Embedding System](http://cn.arxiv.org/pdf/1705.02304)|【[论文笔记](https://www.evernote.com/shard/s148/sh/bfc57087-dfad-44c7-a361-aa8f70b5cab0/7fd61c98274053394d600a159dfbb257)】||
||[End-to-End Text-Independent Speaker Verification with Triplet Loss on Short Utterances](https://isca-speech.org/archive/Interspeech_2017/pdfs/1608.PDF)|||
|**2016**|[DEEP NEURAL NETWORK-BASED SPEAKER EMBEDDINGS FOR END-TO-END SPEAKER VERIFICATION](http://www.danielpovey.com/files/2016_slt_xvector.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/2a261749-0d70-4e82-941c-0ea762a58a7a/828aea39c2c7cb8f5d474d30bebbe253)】|传统的说话人识别方案中，一般采用Ivector作为前端，PLDA作为后端打分，本文黄总作者提出了一种end-to-end的说话人验证的方案，采用DNN网络提取前端特征，同时训练后端的打分参数，相比于Ivector，获得了较大的性能提升。另外由于两种方法具有一定的正交性，通过score fusion，性能可以取得进一步的提升。|
|**2015**|[Time delay deep neural network-based universal background models for speaker recognition](https://ieeexplore.ieee.org/document/7404779) |【[论文笔记](https://www.evernote.com/shard/s148/sh/a6f918a9-43be-4e78-81b8-4de481d35c3f/af5881b92b31fcc2747ac58f78c8e33c)】|鉴于TDNN在ASR领域中的优秀性能，针对DNN-Ivector的方案，作者采用TDNN来代替GMM计算后验概率，得到充分统计量，取得了50%的EER下降。在保持同样计算量的条件下，获得了20%的EER下降。|
||[Locally-Connected and Convolutional Neural Networks for Small Footprint Speaker Recognition](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43970.pdf) |【[论文笔记](https://www.evernote.com/shard/s148/sh/a35bbb85-31f6-48f6-9434-1a7343929506/e712357e5b6e3792b2b2c46972878cc4)】|（d-vector）中的进一步优化（主要针对模型大小），作者在输入层和第一个隐藏层采用了**局部连接（locally-connected）**或者CNN的连接方式，而非全连接（fully-connected）的方式，在模型大小减小70%的情况下，保持模型性能基本不变，在保持相同的模型大小情况下，可以得到8%的EER下降。|
||[End-to-End Text-Dependent Speaker Verification](http://cn.arxiv.org/pdf/1509.08062)|【[论文笔记](https://www.evernote.com/shard/s148/sh/fe0ae3f1-5934-40b4-81b6-80caa4ec32ce/46220d97c19dec01d8b68eb34dc6c95b)】|d-vector进一步优化，将打分的过程集成到网络里面，并对网络结构进行一定的优化，实现了一个有效、高精度、易于维护、small footprint的speaker verification system。|
|**2014**|[DEEP NEURAL NETWORKS FOR SMALL FOOTPRINT TEXT-DEPENDENT SPEAKER VERIFICATION](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41939.pdf) |【[论文笔记](https://www.evernote.com/shard/s148/sh/96d49377-23ad-44d5-83cf-de5275ab79ca/9d15039701fba11446038aa6f328c13f)】|DNN + Cosine distance实现speaker verification。以speaker作为分类目标，将所有隐藏层当做特征提取器，最后一个隐藏层的输出作为特征，采用average pooling，生成d-vector，采用cosine distance得到score/confidence.实验表明可以取得和ivector+PLDA相媲美的结果，但在噪声环境下有更好的鲁棒性，对这两种方法做score fusion，在安静和噪声环境下，分别可以得到14%和25%的EER下降。|
||[NOVEL SCHEME FOR SPEAKER RECOGNITION USING A PHONETICALLY-AWARE DEEP NEURAL NETWORK](http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202014/papers/p1714-lei.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/7fbe5ade-9169-42b5-a26f-8ead574bdf6e/212af252b39ea250cc428ee316abaa73)】|作者提出了一个DNN-ivector的方案用于speaker recognition，并取得的比ivector更好的性能。|
||[Unifying Probabilistic Linear Discriminant Analysis Variants in Biometric Authentication](http://cs.uef.fi/~sizov/pdf/unifying_PLDA_ssspr2014.pdf)|||
||[From single to multiple enrollment i-vectors: Practical PLDA scoring variants for speaker verification](https://www.sciencedirect.com/science/article/pii/S1051200414001377)|||
|**2012**|[A Small Footprint i-Vector Extractor](https://www.isca-speech.org/archive/odyssey_2012/papers/od12_001.pdf)|||
||[ROBUST SPEAKER RECOGNITION BASED ON LATENT VARIABLE MODELS](https://drum.lib.umd.edu/bitstream/handle/1903/13092/GarciaRomero_umd_0117E_13566.pdf?sequence=1&isAllowed=y)|||
|**2011**|[Analysis of I-vector Length Normalization in Speaker Recognition Systems](https://isca-speech.org/archive/archive_papers/interspeech_2011/i11_0249.pdf)|||
|**2010**|[Front-End Factor Analysis for Speaker Verification](http://habla.dc.uba.ar/gravano/ith-2014/presentaciones/Dehak_et_al_2010.pdf)|||
|**Before 2010**|[Bottleneck Features for Speaker Recognition](https://pdfs.semanticscholar.org/3469/fe6e53e65bced5736480afe34b6c16728408.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/a8b162a5-bdba-47f6-9c69-d017796421f1/3c9425e76836eb1aecc70bbe267a7688)】|借鉴语音识别的bottlenec feature，用于GMM-UBM的训练。实验结果显示可以取得接近GMM-UBM的性能。使用类似Teacher-Student的方案，将Baseline的GMM-UBM方案（未详细说明是GMM-UBM 还是 ivector）作为Teacher训练bottleneck网络，在同mic/不同mic的场景下，EER可以分别获得14%和18%的下降。|
||[Bottleneck Features for Speaker Recognition](https://pdfs.semanticscholar.org/3469/fe6e53e65bced5736480afe34b6c16728408.pdf)|||
||[A Straightforward and Efficient Implementation of the Factor Analysis Model for Speaker Verification](http://mistral.univ-avignon.fr/doc/publis/07_Interspeech_Matrouf.pdf)|||
|| [Probabilistic Linear Discriminant Analysis for Inferences About Identity](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester2201112/prince-iccv07-plda.pdf)|||
||[Probabilistic Linear Discriminant Analysis](http://people.irisa.fr/Guillaume.Gravier/ADM/articles2018/Probabilistic_Linear_Discriminant_Analysis.pdf)|||
||[Joint Factor Analysis versus Eigenchannels in Speaker Recognition](https://www.crim.ca/perso/patrick.kenny/FASysJ.pdf)|||
||[Joint Factor Analysis of Speaker and Session Variability: Theory and Algorithms](https://www.crim.ca/perso/patrick.kenny/FAtheory.pdf)|||
||[A Study of Inter-Speaker Variability in Speaker Verification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.494.6825&rep=rep1&type=pdf)
||[Eigenvoice Modeling With Sparse Training Data](https://www.crim.ca/perso/patrick.kenny/eigenvoices.PDF?)|||
||[Support Vector Machines using GMM Supervectors for Speaker Verification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.604&rep=rep1&type=pdf)|||
||[Speaker Verification Using Adapted Gaussian Mixture Models](http://speech.csie.ntu.edu.tw/previous_version/Speaker%20Verification%20Using%20Adapted%20Gaussain%20Mixture%20Models.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/06387ebc-b733-42e3-b74f-9ac14f2abdb3/a3114ea38286be455dfe42ff4fee507b)】|作者提出了一个基于GMM-UBM结果的speaker verification system，并得到了不错的结果。|


### Software


### Datasets


### Other learning materials


### Products
