---
title: "AdaBoost"
date: 2022-01-12
categories:
  - "ML"
tags:
  - "Tree"
  - "Boosting"
  - "AdaBoost"
sidebar: false
---

## AdaBoost.M1

AdaBoost의 초기 버전인 AdaBoost.M1은 $y\in\\{-1, 1\\}$의 binary classification 문제를 해결하기 위한 모델로 제안되었다. AdaBoost의 핵심 컨셉은 "weak classifier를 모아 powerful committee를 만들자"이다. 여기서 weak은 random guessing 보다 조금 나은 정도를 의미한다. (ex. stump, a tree with only 2 nodes)

{{<figure src="/ml/ada1.png" width="400">}}

**Algorithm**
1. Initialize the observation weights $w\_i=1/N$, $i=1, 2, \ldots, N$. ➡ 첫 iteration에서는 각 데이터에 동등한 가중치를 적용한다.
2. For $m=1$ to $M$:
    1. Fit a classifier $G\_m(x)$ to the training data using weights $w_i$.
    2. Compute $\text{err}\_m=\dfrac{\sum\_{i=1}^Mw\_iI(y\_i\neq G\_m(x\_i))}{\sum\_{i=1}^Nw\_i}$.
    3. Compute $\alpha\_m=\log\left(\dfrac{1-\text{err}\_m}{\text{err}\_m}\right)$. ➡ $m$ 번째 weak classifier의 accuracy가 random guessing($1/2$)보다 좋으면 양수, 그렇지 않으면 음수 가중치를 적용한다.
    4. Set $w\_i\leftarrow w\_i\cdot\exp[\alpha\_m\cdot I(y\_i\neq G\_m(x\_i))]$, $i=1, 2, \ldots, N$. ➡ 오분류한 데이터의 가중치를 늘리고 정분류한 데이터의 가중치를 줄인다.
3. Output $G(x)=\text{sign}\left[\sum\_{m=1}^M\alpha\_mG\_m(x)\right]=\underset{k}{\text{argmax}}\sum\_{m=1}^M\alpha\_m\cdot I(G\_m(x)=k)$. ➡ $M$개 모델의 가중합의 부호로 최종 예측값을 계산한다.

AdaBoost는 basis function $b(x;\gamma\_m)$이 $G\_m(x)$이고 loss function $L(y, f(x))$가 $e^{-yf(x)}$인 Forward Stagewise Additive Modeling의 형태이다. (각각의 weak classifier를 basis로 하는 basis expansion)

$f(x)=\sum\_{m=1}^M\beta\_mb(x;\gamma\_m)$과 같은 basis expansion 형태의 모델은 $\displaystyle \min\_{\{\beta\_m, \gamma\_m\}\_1^M}\sum\_{i=1}^NL(y\_i, f(x\_i))$의 최적해를 구하는 것이 쉽지 않다. 그런데 subproblem인 $\displaystyle \min\_{\beta, \gamma}\sum\_{i=1}^NL(y\_i, \beta b(x\_i;\gamma))$만 풀 수 있어도 해결책을 구할 수 있다!?\
&rarr; 한 개의 basis function씩 최적화하여 더해나간다! 이때 이미 최적화된 파라미터는 더 이상 건들지 않는다.

**Algorithm**
1. Initialize $f\_0(x)=0$.
2. For $m=1$ to $M$:
    1. Compute $(\beta\_m, \gamma\_m)=\underset{\beta, \gamma}{\text{argmin}}\sum\_{i=1}^NL(y\_i, f\_{m-1}(x\_i)+\beta b(x\_i;\gamma))$.
    2. Set $f\_m(x)=f\_{m-1}(x)+\beta\_mb(x;\gamma\_m)$.

## AdaBoost-SAMME

기존 loss 대신 multi-class exponential loss를 사용하면 AdaBoost를 multi-class 문제에도 적용할 수 있다. 결과적으로 기존 알고리즘에서 각 weak classifier에 대한 가중치만 다음과 같이 변경된다.

Setting: $y\in\\{1, 2, \ldots, K\\}$

$\alpha\_m=\log\left(\dfrac{1-\text{err}\_m}{\text{err}\_m}\right)+\log(K-1)$

위 식이 의미하는 것은 어떤 weak classifier의 가중치가 양수가 되려면 $1-\text{err}\_m>1/K$이어야 한다는 것이다. \
&rarr; 어떤 weak classifier의 accuracy가 random guessing($1/K$)보다 좋아야 한다.

## AdaBoost.R2

AdaBoost를 regression 문제에 적용하고 싶으면 다음과 같은 loss를 사용하면 된다.

$\displaystyle D=\sup\_i\vert\hat{y}\_i^{(m)}-y\_i\vert$

$L\_i=\begin{cases} \dfrac{\vert\hat{y}\_i^{(m)}-y\_i\vert}{D} & \text{(linear)} \\\\ \dfrac{\vert\hat{y}\_i^{(m)}-y\_i\vert^2}{D} & \text{(square)} \\\\ 1-\exp\left[\dfrac{\vert\hat{y}\_i^{(m)}-y\_i\vert}{D}\right] & \text{(exponential)} \end{cases}$

$\displaystyle \bar{L}\_i=\sum\_{i=1}^Nw\_iL\_i$

$\beta\_m=\dfrac{1-\bar{L}}{\bar{L}}$

$w\_i\leftarrow w\_i\cdot\beta\_m^{(1-L\_i)}$

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class adaboost. Statistics and its Interface, 2(3), 349-360.
3. Drucker, H. (1997, July). Improving regressors using boosting techniques. In Icml (Vol. 97, pp. 107-115).
