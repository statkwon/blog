---
title: "LDA-Dimension Reduction"
date: 2022-02-02
categories:
  - "ML"
tags:
  - "LDA"
  - "Dimension"
sidebar: false
plotly: true
---

LDA는 MLE를 사용하여 $\boldsymbol{\mu}_k$와 $\Sigma$를 추정하는데, 이러한 추정 방식은 High-Dimension에서 불안정하다는 문제를 갖는다. ($p/n\rightarrow\infty$인 경우 MLE의 Aymptotic Property가 보장되지 않는다.) 이러한 문제는 $p$차원의 데이터 $\mathbf{x}$를 보다 낮은 차원의 데이터 $\mathbf{z}$로 변환한 후 LDA를 적용함으로써 해결할 수 있다. 이러한 변환을 수행하는 가장 간단한 방법은 $l$($<\\!\\!<p$)차원의 Subspace에 데이터를 Projection 시키는 것이다. 이때 단순히 차원을 낮추는 것뿐만 아니라, Projection 이후 데이터를 가장 잘 분류할 수 있는 Subspace를 찾는 것이 합리적이다. 따라서 우리는 Projection 이후 범주 간 분산은 최대화하고, 범주 내 분산은 최소화하는 $\mathbb{R}^p$의 Subspace를 찾는 것을 목표로 한다.

## Two-Class Case(Fisher's LDA)

Binary Classification의 경우, Fisher가 제안한 방식을 통해 최적의 Subspace를 찾을 수 있다.

{{<figure src="/ml/lda-dr1.jpeg" width="700">}}

데이터를 어떤 벡터 $\mathbf{w}$에 Projection 시켰을 때 Projection된 범주별 평균을 $m_1=\mathbf{w}^T\boldsymbol{\mu}_1$, $m_2=\mathbf{w}^T\boldsymbol{\mu}_2$, Projection된 각 데이터를 $z=\mathbf{w}^T\mathbf{x}$라고 하자. 우리의 목표는 Projection 이후 범주 간 분산은 최대화되면서 동시에 범주 내 분산은 최소화되는 벡터 $\mathbf{w}$를 찾는 것이다. 이를 식으로 나타내면 다음과 같다.

$\displaystyle \max_\mathbf{w}\dfrac{(m_2-m_1)^2}{s_1^2+s_2^2}$, where $s_k^2=\sum_i(z_i-m_k)^2$

그룹 간 공분산 행렬 $S_B$와 그룹 내 공분산 행렬 $S_W$를 사용하면 위 식을 다음과 같이 표현할 수 있다.

$\displaystyle \max_\mathbf{w}\dfrac{\mathbf{w}^TS_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}$, where $S_B=(\boldsymbol{\mu}\_2-\boldsymbol{\mu}\_1)^T(\boldsymbol{\mu}\_2-\boldsymbol{\mu}\_1)$ and $\displaystyle S_W=\sum_{k=1}^2\sum_{i:y_i=1}(\mathbf{x}_i-\boldsymbol{\mu}_k)^T(\mathbf{x}_i-\boldsymbol{\mu}_k)$

{{<rawhtml>}}
<details>
<summary>Proof</summary>
$\begin{aligned}
\mathbf{w}^TS_B\mathbf{w}&=\mathbf{w}^T(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^T\mathbf{w} \\
&=(m_2-m_1)(m_2-m_1)
\end{aligned}$
<br><br>
$\begin{aligned}
\mathbf{w}^TS_W\mathbf{w}&=\sum_{i:y_i=1}\mathbf{w}^T(\mathbf{x}_i-\boldsymbol{\mu}_1)(\mathbf{x}_i-\boldsymbol{\mu}_1)^T\mathbf{w}+\sum_{i:y_i=2}\mathbf{w}^T(\mathbf{x}_i-\boldsymbol{\mu}_2)(\mathbf{x}_i-\boldsymbol{\mu}_2)^T\mathbf{w} \\
&=\sum_{i:y_i=1}(z_n-m_1)^2+\sum_{i:y_i=2}(z_n-m_2)^2
\end{aligned}$
</details>
<br>
{{</rawhtml>}}

이때 Objective Function이 상수이므로, $\mathbf{x}$에 대해 미분하여 최적해를 구할 수 있다.

$\begin{aligned}
\dfrac{\partial}{\partial\mathbf{w}}\dfrac{\mathbf{w}^TS_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}&=\dfrac{1}{(\mathbf{w}^TS_W\mathbf{w})^2}\left\\{\left(\dfrac{\partial}{\partial\mathbf{w}}\mathbf{w}^TS_B\mathbf{w}\right)\cdot\mathbf{w}^TS_W\mathbf{w}-\mathbf{w}^TS_B\mathbf{w}\cdot\left(\dfrac{\partial}{\partial\mathbf{w}}\mathbf{w}^TS_W\mathbf{w}\right)\right\\} \\\\
&=\dfrac{1}{(\mathbf{w}^TS_W\mathbf{w})^2}\left\\{(S_B+S_B^T)\mathbf{w}\cdot\mathbf{w}^TS_W\mathbf{w}-\mathbf{w}^TS_B\mathbf{w}\cdot(S_W+S_W^T)\mathbf{w}\right\\} \\\\
&=\dfrac{2S_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}-\dfrac{\mathbf{w}^TS_B\mathbf{w}\cdot2S_W\mathbf{w}}{(\mathbf{w}^TS_W\mathbf{w})^2}\overset{\text{let}}{=}0
\end{aligned}$

$\begin{aligned}
&\Leftrightarrow \mathbf{w}^TS_W\mathbf{w}\cdot2S_B\mathbf{w}=\mathbf{w}^TS_B\mathbf{w}\cdot2S_W\mathbf{w} \\\\
&\Leftrightarrow S_B\mathbf{w}=\lambda S_W\mathbf{w}, \\;\text{where}\\;\lambda=\dfrac{\mathbf{w}^TS_B\mathbf{w}}{\mathbf{w}^TS_W\mathbf{w}}
\end{aligned}$

이러한 형태의 문제를 Generalized Eigenvalue Problem이라고 하며, 여기서 $S_W$는 Invertible하므로 $S_W^{-1}S_B\mathbf{w}=\lambda\mathbf{w}$의 Regular Eigenvalue Problem으로 바꾸어줄 수 있다. 따라서 최적해 $\mathbf{w}$는 $S_W^{-1}S_B$의 가장 큰 Eigenvalue에 상응하는 Eigenvector가 된다.

## Multi-Class Case

지금부터 Multiclass의 경우에 대해 최적의 Subspace를 찾는 일반화된 과정에 대해 살펴볼 것이다.

{{<figure src="/ml/lda-dr2.jpeg" width="200">}}

범주가 $K$개인 경우, Feature Space의 차원을 $K-1$차원까지 줄이는 것은 어렵지 않다. 우리는 위 그림으로부터 데이터 $\mathbf{x}$와 Centroid $c_1$, $c_2$와의 거리를 각각 $d_1$, $d_2$라고 했을 때, $d_1>d_2$의 관계는 데이터를 Hyperplane에 Projection 시킨 후에도 $d_1'>d_2'$으로 그대로 유지됨을 확인할 수 있다. 이는 데이터와 Hyperplane 사이의 수직 거리($h$)가 모든 Centroid에 대해 동일한 값이므로 데이터와 Centroid 사이의 거리를 측정함에 있어 영향을 미치지 않는 정보이기 때문이다. 따라서 $K$개의 Centroid를 포함하는 $K-1$차원의 Hyperplane에 데이터를 Projection 시킴으로써 데이터를 분류하는데 필요한 정보의 손실 없이 Feature Space의 차원을 낮출 수 있다.

{{<plotly json="/ml/lda-dr3.json" height="600px">}}

하지만 이와 같은 단순한 방식은 범주의 개수가 많을 경우($K>\\!\\!>3$), $K-1$차원 역시 굉장히 높은 차원이 된다는 한계점을 갖는다. 이러한 경우 Centroid에 대한 PCA를 사용하여 차원을 더 낮추는 것이 가능하다. 보다 자세한 과정은 다음과 같다.
1. Class Centroid의 행렬 $M_{K\times p}$과 그룹 내 공분산 행렬 $S_W$를 구한다.
2. $S_W$에 대한 Eigen-Decomposition을 사용하여 Centroid에 Whitening Transformation을 적용한다. ($M^*=MW^{-1/2}$)
3. 변형된 Centroid의 Principal Component Direction을 찾아 그것들이 Span하는 Subspace에 데이터를 Projection 시킨다.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. Murphy, K. P. (2022). Probabilistic machine learning: an introduction. MIT press.
