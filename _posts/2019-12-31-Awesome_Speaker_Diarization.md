---
layout: post
title: Awesome Speaker Diarization
date: 2019-12-31
categories: 话者分离
tags: [话者分离]
grammar_cjkRuby: true
---
## Awesome Speaker Diarization

### Table of contents
* [Overview](#Overview)
* [Publications](#Publications)
* [Software](#Software)
* [Datasets](#Datasets)
* [Leaderboards](#Leaderboards)
* [Other learning materials](#Other-learning-materials)
* [Products](#Products)

### Overview
本文用于记录一些经典的话者分离相关的论文、数据集、开源软件工具等资源，便于后续的回顾及使用。

本文会长期持续更新。。。

### Publications

| Time | Paper | Paper Note | Abstract |
|-|-|-|-|
|**2017**|[Speaker Diarization using Deep Recurrent Convolutional Neural Networks for Speaker Embeddings](https://arxiv.org/pdf/1708.02840.pdf)|【[论文笔记](https://www.evernote.com/l/AJR3TKUyW4BMLbdV3DPtVSuh9eF8w0v0Pg0)】|采用RCNN提取speaker embedding，相比于Baseline，DER有相对30%的下降。|
||[Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks](https://pdfs.semanticscholar.org/edff/b62b32ffcc2b5cc846e26375cb300fac9ecc.pdf)|【[论文笔记](https://www.evernote.com/l/AJT50EQK-opBtLR-gKqOfZqDTYoNNbLMxbs)】|将SCD(Speaker change detection)任务视为序列标注任务，采用Bi-LSTM网络，相比于传统的BIC等方法，具有较大的提升。|
||[pyannote. metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0411.PDF)||A toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems|
||[Speaker Diarization Using Convolutional Neural Network for Statistics Accumulation Refinement](https://pdfs.semanticscholar.org/35c4/0fde977932d8a3cd24f5a1724c9dbca8b38d.pdf)|【[论文笔记](https://www.evernote.com/l/AJRIS4dTBMtJmb_LWab8KqJT59rO8FquwC8)】|CNN做SCD(Speaker change detection)，并将**输出的speaker change probability作用到ivector提取的统计量计算上**，从而产生更加准确的ivector描述，实验证明DER有约16%的下降。 |
||[SPEAKER DIARIZATION USING DEEP NEURAL NETWORK EMBEDDINGS](http://danielpovey.com/files/2017_icassp_diarization_embeddings.pdf)|【[论文笔记](https://www.evernote.com/l/AJR4sEICuc9A26emZQG2UgVD96hLkpsp3Sg)】|JHU的后续方案（[[1](https://www.evernote.com/l/AJQg_D-5eNhGQ4wpYgd6Pgv1ribO0c1Johk)][[2](https://www.evernote.com/l/AJQUsYgQzkJGgoUk7K0EzmaYttBCYcCLxTo)]，kaldi recipe），采用[DNN Based embedding](https://www.evernote.com/l/AJQqJhdJDXBOgpQcDqdipYp6gorqOcLHy48)替换ivector|
||[SPEAKER DIARIZATION WITH LSTM](https://arxiv.org/pdf/1710.10468.pdf)|【[论文笔记](https://www.evernote.com/l/AJQf2ml_61JGFJ7dHGwXD3rUt7168gHwTY8)】|Google 的Speaker Diarization方案: dvector + spectral clustering|
|**2016**|[A Speaker Diarization System for Studying Peer-Led Team Learning Groups](https://arxiv.org/pdf/1606.07136.pdf)|【[论文笔记](https://www.evernote.com/l/AJSrmbld7w1FBYYqQYAf_GrX455S1nxAR8E)】|独特的场景设计（PLTL，学生小组学习的会议场景，每个学生随身携带录音设备），针对每一路信号，SAD去除non-speech，采用GMM + 类BIC准则做分割，Hausdorff 距离做聚成2类，基于能量区分主说话人和其他说话人，结合多路信号的结果综合finetune得到最终结果。|
|**2015**|[DIARIZATION RESEGMENTATION IN THE FACTOR ANALYSIS SUBSPACE](https://engineering.jhu.edu/hltcoe/wp-content/uploads/sites/92/2016/10/Sell_Garcia-Romero_2015A.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/20fc3fb9-78d8-4643-8c29-62077a3e0bf5/ae26ced1cd49a2190b5c7e6375889b57)】|在[SPEAKER DIARIZATION WITH PLDA I-VECTOR SCORING AND UNSUPERVISED CALIBRATION](https://ieeexplore.ieee.org/document/7078610)的基础上，采用VB（[Variational Bayes](https://pdfs.semanticscholar.org/36db/5cc928d01b13246582d71bde84fabbd24a19.pdf)） resegmentation，进一步提升性能。|
|**2014**|[A Study of the Cosine Distance-Based Mean Shift for Telephone Speech Diarization](https://www.researchgate.net/publication/260661427_A_Study_of_the_Cosine_Distance-Based_Mean_Shift_for_Telephone_Speech_Diarization)|【[论文笔记](https://www.evernote.com/shard/s148/sh/22030678-b9f7-4c16-8c45-19081deaa920/672fe43afdf2561cc355d0cbc0985927)】|Recipe: ivector ==> BCCN(between class covariance normalization) ==> LN(length normalization) ==> PCA ==> Mean shift based clustering ==> Resegmentation|
| |[SPEAKER DIARIZATION WITH PLDA I-VECTOR SCORING AND UNSUPERVISED CALIBRATION](https://ieeexplore.ieee.org/document/7078610)|【[论文笔记](https://www.evernote.com/shard/s148/sh/14b18810-ce42-4682-8524-ecad04ce6698/b6d04261c08bc53ade213f31dffe370c)】|kaldi recipe: ivector + Plda|
| |[Artificial neural network features for speaker diarization](https://ieeexplore.ieee.org/abstract/document/7078608)|【[论文笔记](https://www.evernote.com/shard/s148/sh/061ab04e-fa8f-4a22-aea7-d02710e8a4e4/36e744d571e8d6f1b315c3af3d275cb6)】|训练二分类ANN（输入：two segments，输出：same/different speaker），然后采用ANN的bottleneck feature 和 MFCC 分别作为segment的feature，构建两组GMM，最终发射概率加权求和用于HMM做segmentation，Modified BIC准则做clustering|
|**2013**|[Unsupervised methods for speaker diarization: An integrated and iterative approach](http://groups.csail.mit.edu/sls/publications/2013/Shum_IEEE_Oct-2013.pdf)|||
| **2011**|[PLDA-based Clustering for Speaker Diarization of Broadcast Streams](https://pdfs.semanticscholar.org/0175/a752c5c72cadc7c0b899fd15f2f6b93c3335.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/ba92d8fc-7142-4be6-bb05-1ba653f4e5ca/c9b3377046c8c826969b252dd6c32fbd)】|GMM做SAD， BIC 准则做分割，引入ivector + plda做聚类|
| |[SPEAKER DIARIZATION OF MEETINGS BASED ON SPEAKER ROLE N-GRAM MODELS](https://publications.idiap.ch/downloads/papers/2011/Valente_ICASSP2011_2011.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/e21f69ac-ccaf-465a-b7c7-ac1e41f187b4/3f555aa1aa512bc461b0461d2dadc9c4)】|引入n-gram语言模型的思想，对会议中的参会任何的发言顺序按n-gram的思想建模，并引入到聚类过程中，有效提升聚类结果。|
| |[Unsupervised methods for speaker diarization: An integrated and iterative approach](http://groups.csail.mit.edu/sls/publications/2013/Shum_IEEE_Oct-2013.pdf)|||
||[Artificial neural network features for speaker diarization](https://ieeexplore.ieee.org/abstract/document/7078608)|||
|**before 2010**|[Speaker Diarization for Meeting Room Audio](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester2_2009_10/sun_IS2009_SpDia_meeting.PDF)|【[论文笔记](https://www.evernote.com/shard/s148/sh/b0ce89dc-9fe5-490f-8624-4a6180a56eab/fe9f559ad81b223f731e30eace9a141d)】|1. 会议场景 2. TODA(Time Difference of Arrival) feature 3. BIC for clustering|
||[Stream-based speaker segmentation using speaker factors and eigenvoices](https://www.researchgate.net/profile/Pietro_Laface/publication/224313019_Stream-based_speaker_segmentation_using_speaker_factors_and_eigenvoices/links/5770fe8608ae10de639dc121.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/b5d21b08-529b-4085-ba5e-b28ba4953a65/78ed283ff74897b589c815563bb8ebfd)】|stream 方式做话者分离，基于本征音向量 + GMM/HMM 做segmentation和 clustering。|
||[The LIA-EURECOM RT‘09 Speaker Diarization System](http://www.eurecom.fr/~evans/papers/pdfs/2763.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/e24a52ae-7686-42d0-831e-15b0271f55b0/88f8b00077f5fb29062873e7b889c0d5)】|基于GMM-HMM的自顶向下聚类做话者分离的典型方案。|
||[E-HMM approach for learning and adapting sound models for speaker indexing](https://www.researchgate.net/publication/245997669_E-HMM_approach_for_learning_and_adapting_sound_models_for_speaker_indexing)|||
||[The ICSI RT07s Speaker Diarization System](http://www.icsi.berkeley.edu/pubs/speech/ICSI_RT07_diarization.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/783140c8-f042-4602-9396-7ca7ecc4b81f/1aaa38a667c642f14c589eeef5d9a6ed)】|基于GMM-HMM的自底向上聚类做话者分离的典型方案。|
||[An Overview of Automatic Speaker Diarization Systems](https://alize.univ-avignon.fr/doc/publis/06_IEEE-TASP_Tranter.pdf) |【[论文笔记](https://www.evernote.com/shard/s148/sh/bd3e2b40-9367-4af0-ba3e-cd4c0a39938e/5d792a17e53f6fbed11a82df4b6d39c4)】|2006年的关于话者分离的综述，较为详细的分析了那个时候的话者分离的各种方案，并在RT evolution中比较了各个方案的性能。|
||[Robust Speaker Diarization for meetings](http://www1.icsi.berkeley.edu/~xanguera/PhD_Thesis.pdf)|||
||[A Spectral Clustering Approach to Speaker Diarization](http://www.ifp.illinois.edu/~hning2/papers/Ning_spectral.pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/e7a7b266-60b4-42a7-b92f-fd95d40dabfd/ffed8b7d326fccc31ace01bdacb8b47a)】|BIC做segmentation，GMM建模+KL距离做初步聚类，spectral clustering进行二次聚类，然后Cross EM进行refine|
||[Improved speaker segmentation and segments clustering using the bayesing information criterion](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.905&rep=rep1&type=pdf)|【[论文笔记](https://www.evernote.com/shard/s148/sh/e1589775-a55d-42cd-b291-49004ceabed5/a27bb60b7b714cf69212d8b760f2e056)】|BIC(Basysian Information Criterion)用于做分割和聚类的经典论文|

### Software
#### Speaker Diarization

| Tools | Language | Description | 
|-|-|-|
|[VB diarization](https://speech.fit.vutbr.cz/software/vb-diarization-eigenvoice-and-hmm-priors)| Python | Based on Bayesian Hidden Markov Model|
|[pyannote-metrics](https://github.com/pyannote/pyannote-metrics)| Python | A toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems|

### Datasets


### Other learning materials


### Products
