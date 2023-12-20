---
title: "Ridge Regression"
date: 2023-12-18
lastmod: 2023-12-20
categories:
  - "ML"
tags:
  - "Ridge"
  - "Regression"
  - "Regularization"
sidebar: false
---

## Ridge Regression

Ridge regression은 선형 회귀에 $L\_2$ penalty를 추가한 모형이다. input scale에 따라 해가 달라지기 때문에 일반적으로 standardized된 input을 사용한다.

$$\begin{align}
\hat{\boldsymbol{\beta}}&=\underset{\boldsymbol{\beta}}{\text{argmin}}\left\\{(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})+\lambda\Vert\boldsymbol{\beta}\Vert\_2^2\right\\} \\\\
&=\underset{\beta}{\text{argmin}}\left\\{\sum\_{i=1}^N\left(y\_i-\sum\_{j=1}^Px\_{ij}\beta\_j\right)^2+\lambda\sum\_{j=1}^P\beta\_j^2\right\\} \\\\
&=\underset{\beta}{\text{argmin}}\sum\_{i=1}^N\left(y\_i-\sum\_{j=1}^Px\_{ij}\beta\_j\right)^2 \\; \text{subject to} \\; \sum\_{j=1}^P\beta\_j^2\leq t
\end{align}$$

위 식의 해를 구하면 $\hat{\boldsymbol{\beta}}=(X^TX+\lambda I)^{-1}X^T\mathbf{y}$이 된다. 이때 $X^TX$가 positive semi-definite이므로, $(X^TX+\lambda I)$는 positive definite이 되어 항상 invertible하게 된다.

## Geometric Interpretation

{{<figure src="/ml/ridge1.gif" width="400">}}

$X$의 SVD를 활용하여 $\hat{\mathbf{y}}$을 다음과 같이 나타낼 수 있다. ($X=UDV^T$)

$$\begin{align}
\hat{\mathbf{y}}&=X(X^TX+\lambda I)^{-1}X^T\mathbf{y} \\\\
&=UD(D^2+\lambda I)^{-1}DU^T\mathbf{y} \\\\
&=\sum\_{j=1}^P\dfrac{d\_j^2}{d\_j^2+\lambda}\mathbf{u}\_j\mathbf{u}\_j^T\mathbf{y}
\end{align}$$

$\mathbf{u}\_j\mathbf{u}\_j^T\mathbf{y}=\text{proj}\_{\mathbf{u}\_j}\mathbf{y}$이므로, $\hat{\mathbf{y}}$은 $\text{col}(X)$의 orthonormal basis인 $\mathbf{u}\_j$에 $\mathbf{y}$를 projection한 것을 $\frac{d\_j^2}{d\_j^2+\lambda}$만큼 축소한 벡터들의 합과 같다. 이때 $\mathbf{u}\_j$가 $X$의 $j$ 번째 normalized principal component($\frac{X\mathbf{v}\_j}{d\_j}$)라는 점에서, 분산이 작은 방향일수록 높은 shrinkage를 받음을 알 수 있다.

변수 간 [multicollinearity]()가 존재할 경우 regression coefficient가 불안정해지는 문제가 발생하는데, Ridge regression이 이를 완화하는데 도움이 된다.

"If we consider fitting a linear surface over this domain, the configuration of the data allow us to determine its gradient more accurately in the long direction than the short. Ridge regression protects against the potentially high variance of gradients estimated in the short directions. The implicit assumption is that the response will tend to vary most in the directions of high variance of the inputs. This is often a reasonable assumption, since predictors are often chosen for study because they vary with the response variable, but need not hold in general."

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. https://online.stat.psu.edu/stat508/lesson/5/5.1
