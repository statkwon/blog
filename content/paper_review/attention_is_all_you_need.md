---
title: "Attention Is All You Need"
date: 2022-04-29
categories:
  - "Paper Review"
tags:
  - "NLP"
  - "Transformer"
sidebar: false
---

## 1. Introduction
---

RNN 계열의 모형들은 Sequence Modeling에 큰 기여를 하였지만, Sequential한 구조로 인해 병렬 연산이 불가하여 Sequence의 길이가 길어질 수록 계산 효율성이 떨어진다는 단점을 가지고 있다. Attention Mechanism 역시 대부분의 경우 RNN 구조에 기반하기 때문에 이러한 문제로부터 자유로울 수 없다. 따라서 이 논문에서는 RNN 구조를 사용하지 않고, 오로지 Attention Mechanism으로만 구성된 Transformer라는 새로운 모형을 제안한다. Transformer는 RNN 구조를 사용하지 않기 때문에 병렬 연산이 가능하며, 번역 작업에 있어 기존의 방법론보다 짧은 시간 안에 SOTA를 달성할 수 있다.

## 3. Model Architecture
---

대부분의 우수한 Neural Sequence 모형이 그러하듯이, Transformer 역시 Encoder-Decoder 구조를 따른다. 하지만 Transformer는 Encoder와 Decoder에 Self-Attention과 Position-wise Fully Connected Layer를 사용한다는 점에서 기존 모형들과 차이를 갖는다. 3장에서는 이러한 모형 구조에 대해 자세히 설명하고 있다.

{{<figure src="/paper_review/transformer1.jpg" width="300">}}

### 3.1. Encoder and Decoder Stacks

**Encoder**

한 개의 Encoder에는 한 개의 Multi-Head Attention과 한 개의 Position-wise Feed-Forward Network가 포함된다. 그리고 이 두 Sub Layer는 항상 Reisudal Connection과 Layer Normalization을 동반한다. Residual Connection이란 ResNet의 Reisdual Block과 동일한 개념으로, Sub Layer의 Output에 Input을 더해주는 역할을 수행한다. 따라서 하나의 Sub Layer에 대한 연산 결과를 수식으로 표현하면 $\text{LayerNorm}(\mathbf{x}+\text{Sublayer}(\mathbf{x}))$가 된다. Transformer는 이와 같은 구조를 갖는 Encoder $N$개를 중첩하여 사용한다. (이 논문에서는 $N=6$을 사용하고 있다.)

**Decoder**

한 개의 Decoder에는 Encoder에 포함되어 있는 Multi-Head Attention과 Position-wise Feed-Forward Network 이외에도, Encoder의 Output을 Input으로 받아 작동하는 Multi-Head Attention이 추가로 포함되어 있다. 즉, Decoder에는 두 개의 Multi-Head Attention이 존재한다. Encoder에서와 마찬가지로 Decoder의 세 Sub Layer는 Residual Connection과 Layer Normalization을 동반하며, 이러한 구조를 $N$개 중첩하여 사용한다.

Encoder와 Decoder에서 Residual Connection을 수행하기 위해서는 Sub Layer의 Input과 Output의 차원이 항상 동일하게 유지되어야 한다. 따라서 Embedding Vector를 비롯한 Transformer의 모든 Sub Layer의 Output은 $d_\text{model}$차원을 갖는다. (이 논문에서는 $d_\text{model}=512$를 사용하고 있다.)

### 3.2. Attention

Attention Function은 Query와 Key 사이의 Compatibility Function(ex. Dot Product + Softmax)의 값을 그에 상응하는 Value에 대한 가중치로 사용하여 Value의 가중합을 출력하는 함수이다. Transformer의 Attention은 Query, Key, Value가 모두 Input Sequence의 Token Vector인 Self-Attention의 형태라는 점에서 seq2seq + Attention Mechanism과 같은 기존 방법론과 차이를 갖는다.

**Scaled Dot-Product Attention**

{{<figure src="/paper_review/transformer2.jpg" width="100">}}

$\text{Attention}(Q, K, V)=\text{softmax}\left(\dfrac{QK^T}{\sqrt{d_k}}\right)V$

Transformer는 Attention Function으로 위 식과 같은 형태의 Scaled Dot-Product Attention을 사용한다. $Q$, $K$, $V$는 Token Vector를 Row Vector로 갖는 행렬로, $\text{Attention}(Q, K, V)$는 각 Token에 대한 Attention Value를 Row Vector로 갖는 행렬이 된다. 여기서 주의해야할 점은 Token Vector의 차원인 $\sqrt{d_\text{model}}$이 아닌 $\sqrt{d_k}$로 Scaling 한다는 것이다. $\sqrt{d_k}$가 어떠한 값을 의미하는지 등은 Multi-Head Attention과 관련된 것으로, 다음 파트에서 확인할 수 있다. Transformer가 보편적으로 사용되는 Dot-Product Attention이 아닌 Scaled Dot-Product Attention을 사용하는 이유는 Query(Key)의 차원이 낮은 경우 Additive Attention보다 계산 효율성이 높은 Dot-Product Attention의 장점을 유지하면서도, (Query와 Key의 Dot Product를 $\sqrt{d}_k$로 나누어 줌으로써) 차원이 높아짐에 따라 그 성능이 저하되는 문제를 해결하기 위함이다. 

**Multi-Head Attention**

{{<figure src="/paper_review/transformer3.jpg" width="200">}}

$\begin{aligned}
\text{MultiHead}(Q, K, V)&=\text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\\\
\text{where} \\; \text{head}_i&=\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}$

where $W_i^Q\in\mathbb{R}^{d_\text{model}\times d_k}$, $W_i^K\in\mathbb{R}^{d_\text{model}\times d_k}$, $W_i^V\in\mathbb{R}^{d_\text{model}\times d_v}$, and $W^O\in\mathbb{R}^{hd_v\times d_\text{model}}$

Transformer는 $d_\text{model}$차원의 Token Vector에 대해 Attention을 수행하지 않고, $d_\text{model}$차원의 Query, Key, Value를 $d_k$, $d_k$, $d_v$차원으로 Projection한 벡터에 대해 Attention을 수행한다. 위의 $\text{head}_i$에 대한 식에서 $W_i^Q$, $W_i^K$, $W_i^V$는 Projection Matrix를, $QW_i^Q$, $KW_i^K$, $VW_i^V$는 Projection된 벡터를 Row Vector로 갖는 행렬을 의미한다. Transformer는 이러한 방식으로 서로 다른 $h$쌍의 $(QW_i^Q, KW_i^K, VW_i^V)$를 만들고, 각각에 대해 Attention을 병렬로 수행하여 $h$개의 Attention Value Matrix를 구한다. 이후 $h$개의 Attention Value Matrix를 가로로 연결하여 만든 행렬에 $W^O$를 곱함으로써 Multi-Head Attention Matrix를 구할 수 있다. (이 논문에서는 $h=8$, $d_k=d_v=d_\text{model}/h=64$를 사용하고 있다.) 이와 같이 여러 개의 Attention Head를 사용함으로써 Transformer는 서로 다른 관점의 정보를 수집할 수 있게 된다.

**Applications of Attention in Transformer**

Transformer의 Multi-Head Attention은 그 위치에 따라 수행하는 역할에 조금씩 차이가 있다.

- Encoder: 이전 Encoder의 Output으로부터 Query, Key, Value를 정의하고 Self-Attention을 수행한다.
- Decoder: 이전 Decoder의 Output으로부터 Query, Key, Value를 정의하고 Self-Attention을 수행한다. 이때 Auto-Regressive Property를 보존하기 위해 Decoder가 현재 위치보다 뒤에 있는 Token들을 참고하지 못하도록 $QK^T/\sqrt{d_k}$를 Masking 처리한다.
- Encoder-Decoder: 이전 Decoder의 Output으로부터 Query를, 마지막 Encoder의 Output으로부터 Key와 Value를 정의하고 Attention을 수행한다.

### 3.3. Position-wise Feed-Forward Networks

$\text{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2$

모든 Multi-Head Attention 이후에는 위 식과 같은 형태의 Fully Connected Feed-Forward Network가 뒤따른다. Input($x$)은 Multi-Head Attention Matrix가 된다. $W_1$은 $d_\text{model}\times d_{ff}$, $W_2$는 $d_{ff}\times d_\text{model}$의 크기를 가지며, 하나의 Encoder 내에서는 모든 위치에서 동일한 값을 갖지만, Encoder별로는 상이한 값을 갖는다. (이 논문에서는 $d_{ff}=2048$을 사용하고 있다.)

### 3.4. Embeddings and Softmax

기존 Sequence Transduction 모형들과 마찬가지로, Input Token과 Output Token을 $d_\text{model}$차원의 Embedding Vector로 변환한다. 또한 마지막 Decoder의 Output에 선형 변환 및 Softmax 함수를 취하여 Next-Token Probability를 예측한다. 이때 Embedding과 선형 변환을 위한 Weight Matrix는 항상 동일한 값을 가지며, Embedding Layer에서는 Weight에 $\sqrt{d_\text{model}}$을 곱하여 사용한다.

### 3.5. Positional Encoding

Transformer는 RNN이나 CNN 구조를 사용하지 않으므로 Sequence의 순서 정보를 활용하기 위한 별도의 장치를 필요로 하며, Positional Encoding이 이러한 역할을 수행한다.

$\text{PE}\_{(\text{pos}, 2i)}=\sin{(\text{pos}/10000^{2i/d_\text{model}})}$

$\text{PE}\_{(\text{pos}, 2i+1)}=\cos{(\text{pos}/10000^{2i/d_\text{model}})}$

where $\text{pos}$ is the position and $i$ is the dimension

어떠한 Sequence에 대한 Positional Encoding은 위와 같은 식을 따른다. Positional Encoding은 $d_\text{model}$차원의 Embedding Vector와 동일한 차원을 가지므로, 두 벡터를 더하여 Encoder와 Decoder의 Input으로 사용한다.

## 4. Why Self-Attention
---

Self-Attention은 다음과 같은 세 가지 항목에서 RNN 또는 CNN보다 더 높은 효율성을 갖는다.

- Total computation complexity per layer
- Amount of computation that can be parallelized
- Path length between long-range dependencies in the network

{{<figure src="/paper_review/transformer4.png" width="600">}}

## 5. Training
---

5장에서는 저자들이 Transformer를 학습시킬 때 사용한 여러 가지 조건에 대해 소개하고 있다. (차후에 디폴트 옵션으로 사용할 목적으로 기록)

Optimizer: Adam Optimizer with $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^-9$

Learning Rate: $\text{lrate}=d_\text{model}^{-0.5}\cdot\min(\text{step_num}^{-0.5}, \text{step_num}\cdot\text{warmup_steps}^{-1.5})$, where $\text{warmup_steps}=4000$

Regularization:
- Residual Dropout ($P_\text{drop}=0.1$)
  1. Dropout to the output of each sub-layer before residual connection & layer normalization
  2. Dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
- Label Smoothing: $\epsilon_\text{ls}=0.1$

## 7. Conclusion
---

- Transformer는 RNN, CNN 구조를 사용하지 않는 최초의 Sequence Transduction 모형이다.
- WMT 2014 English-to-German과 WMT 2014 English-to-French에서 SOTA를 달성
- Transformer를 Image, Audio, Video 데이터에도 적용할 수 있도록 확장하는 것을 추후 목표로 한다.

## Reference
---

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
2. [https://wikidocs.net/31379](https://wikidocs.net/31379)
