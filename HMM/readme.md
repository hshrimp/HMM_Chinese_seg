# HMM隐马尔可夫推理与中文分词实现
---
## 一、公式推导
### 1.1 几个概念

观测状态：中文分词中，观测状态就是我们要分词的文本，可以观测到的。

隐藏状态：分词时，以B表示词的开始的字，M表示词中间的字，E表示词最后的字，S表示单独的字（单独的字构成词）。

初始状态概率：统计文本开头的字分别是BMES的概率。理论上来说，M和E是不可能出现在句首的，概率基本为0.

转移概率矩阵：BEMS之间互相转换的概率矩阵。

发射概率矩阵（观测概率矩阵）：每个隐藏状态下，每个字出现的概率。

### 1.2 推导过程

给一段文本，即观测序列X(x<sub>1</sub>,x<sub>2</sub>...x<sub>n</sub>),那我们的目标就是给出概率最大的隐藏序列Y(y<sub>1</sub>,y<sub>2</sub>...y<sub>n</sub>).

即 argmax(p(Y|X))

其中 p(Y|X) = p(X,Y)/p(X) = p(X|Y)*p(Y)/p(X) , p(X)是常数，最大化过程中可忽略，因此最大化p(Y|X)相当于最大化p(X|Y)*p(Y)。

即 argmax(p(Y|X)) ~ argmax(p(X|Y)*p(Y))

（1）根据HMM的观测独立假设，是任一时刻的观测只依赖于该时刻的马尔科夫链的状态，与其他观测及状态无关。

得到 p(X|Y) = p(x<sub>1</sub>|y<sub>1</sub>)*p(x<sub>2</sub>|y<sub>2</sub>)...*p(x<sub>n</sub>|y<sub>n</sub>)

（2）根据HMM齐次马尔科夫假设，通俗地说就是 HMM 的任一时刻 t 的某一状态只依赖于其前一时刻的状态，与其它时刻的状态及观测无关，也与时刻 t 无关。

得到 p(Y) = p(y<sub>1</sub>)*p(y<sub>2</sub>|y<sub>1</sub>)p(y<sub>3</sub>|y<sub>2</sub>)...*p(y<sub>n</sub>|y<sub>n-1</sub>)  

综合（1）（2），有

p(X|Y)*p(Y) ~  p(y<sub>1</sub>)*p(x<sub>1</sub>|y<sub>1</sub>)*p(y<sub>2</sub>|y<sub>1</sub>)*p(x<sub>2</sub>|y<sub>2</sub>)*p(y<sub>3</sub>|y<sub>2</sub>)...*p(y<sub>n</sub>|y<sub>n-1</sub>)*p(x<sub>n</sub>|y<sub>n</sub>) 
 
其中p(y<sub>1</sub>)可由初始状态概率得到，p(x<sub>i</sub>|y<sub>i</sub>)由观测概率矩阵得到，p(y<sub>t</sub>|y<sub>t-1</sub>)由转移概率矩阵得到。

## 二、最大化隐藏序列概率--viterbi算法

viterbi是一种动态规划算法，之前在做文本生成时最后的beam search也用到这个算法。这里就不展开了。

## 三、其他

代码中使用的语料引用自GitHub链接：https://github.com/CQUPT-Wan/HMMwordseg.git，感谢大佬。

代码系自己实现，如有不足之处欢迎指正。

备注：最后的公式其实可以取对数变成加法形式，本文实现没有取对数，而是直接乘法，对于训练语料中没有出现的字，取平均概率。

