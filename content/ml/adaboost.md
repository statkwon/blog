---
title: "AdaBoost Classifier"
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

"weak classifier를 모아 powerful committee를 만들자"

여기서 weak은 random guessing 보다 조금 나은 정도를 의미한다. (ex. stump, a tree with only 2 nodes)

{{<figure src="/ml/ada1.png" width="400">}}

Setting: $y\in\\{-1, 1\\}$

AdaBoost는 학습한 (weak) 모델의 분류 결과에 따라 데이터의 가중치를 조절하면서 다음 (weak) 모델을 학습시키고, 그렇게 얻은 $M$개의 모델의 가중합의 부호로 최종 예측값을 계산한다. 첫 번째 step에서는 각 데이터에 동등한 가중치($1/N$)를 적용하고, 이후 $m$ 번째 step마다 오분류한 데이터의 가중치를 늘리고 정분류한 데이터의 가중치를 줄인다.

{{<rawhtml>}}
<details><br>
<summary>AdaBoost</summary>
<ol>
<li>Initialize the observation weights $w_i=1/N$, $i=1, 2, \ldots, N$.</li>
<li>For $m=1$ to $M$:
  <ol>
  <li>Fit a classifier $G_m(x)$ to the training data using weights $w_i$.</li>
  <li>Compute $\text{err}_m=\dfrac{\sum_{i=1}^Mw_iI(y_i\neq G_m(x_i))}{\sum_{i=1}^Nw_i}$.</li>
  <li>Compute $\alpha_m=\log((1-\text{err}_m)/\text{err}_m)$.</li>
  <li>Set $w_i\leftarrow w_i\cdot\exp[\alpha_m\cdot I(y_i\neq G_m(x_i))]$, $i=1, 2, \ldots, N$.</li>
  </ol>
</li>
<li>Output $G(x)=\text{sign}\left[\sum_{m=1}^M\alpha_mG_m(x)\right]=\underset{k}{\text{argmax}}\sum_{m=1}^M\alpha_m\cdot I(G_m(x)=k)$.</li>
</ol>
</details><br>
{{</rawhtml>}}

그런데 이러한 AdaBoost는 basis function $b(x;\gamma\_m)$이 $G\_m(x)$이고 loss function $L(y, f(x))$가 $e^{-yf(x)}$인 Forward Stagewise Additive Modeling이다. (각각의 weak classifier를 basis로 하는 basis expansion)

{{<rawhtml>}}
<details><br>
<summary>Forward Stagewise Additive Modeling</summary>
$f(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)$과 같은 basis expansion 형태의 모델은 $\displaystyle \min_{\{\beta_m, \gamma_m\}_1^M}\sum_{i=1}^NL(y_i, f(x_i))$의 최적해를 구하는 것이 쉽지 않다.<br>
그런데 subproblem인 $\displaystyle \min_{\beta, \gamma}\sum_{i=1}^NL(y_i, \beta b(x_i;\gamma))$만 풀 수 있어도 해결책을 구할 수 있다!?<br>
&rarr; 한 개의 basis function씩 최적화하여 더해나간다! 이때 이미 최적화된 파라미터는 더 이상 건들지 않는다.<br><br>
<ol>
<li>Initialize $f_0(x)=0$.</li>
<li>For $m=1$ to $M$:
<ol>
  <li>Compute $(\beta_m, \gamma_m)=\underset{\beta, \gamma}{\text{argmin}}\sum_{i=1}^NL(y_i, f_{m-1}(x_i)+\beta b(x_i;\gamma))$.</li>
  <li>Set $f_m(x)=f_{m-1}(x)+\beta_mb(x;\gamma_m)$.</li>
</ol>
</li>
</ol>
</details><br>
{{</rawhtml>}}

## Multi-class AdaBoost
기존 loss 대신 multi-class exponential loss를 사용하면 AdaBoost를 multi-class 문제에도 적용할 수 있다. 결과적으로 기존 알고리즘에서 각 weak classifier에 대한 가중치만 다음과 같이 변경된다.

Setting: $y\in\\{1, 2, \ldots, K\\}$

$\alpha\_m=\log\left(\dfrac{1-\text{err}\_m}{\text{err}\_m}\right)+\log(K-1)$

위 식이 의미하는 것은 어떤 weak classifier의 가중치가 양수가 되려면 $1-\text{err}\_m>1/K$이어야 한다는 것이다. \
&rarr; 어떤 weak classifier의 accuracy가 random guessing($1/K$)보다 좋아야 한다.


---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class adaboost. Statistics and its Interface, 2(3), 349-360.
