---
title: "Support Vector Classifier"
date: 2022-01-23
categories:
  - "ML"
tags:
  - "SVM"
sidebar: false
---

Support Vector Classifier는 Optimal Seperating Hyperplane을 Nonseperable Case에 대해 일반화한 모형이다. Support Vector Classifier 역시 Margin을 최대화하는 방향으로 작동하지만, 일정 수준의 오분류를 허용함으로써 Nonseperable Case에서도 수렴할 수 있다는 것이 차이점이다.

{{<figure src="/ml/svc1.png" width="400">}}

일정 수준의 오분류를 허용한다는 것을 'Slack Variable'라고 불리는 $\xi_i$를 사용하여 다음과 같이 표현할 수 있다. (위 그림에서 $\xi\_i^*=M\xi\_i$이다.)

$\displaystyle \max\_{\boldsymbol{\beta}, \beta\_0, \Vert\boldsymbol{\beta}\Vert=1}M \quad\text{subject to}\quad y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)\geq M-\xi\_i,\\; \xi\_i\geq 0,\\; \sum\_{i=1}^N\xi\_i\leq c,\\; ^\forall i$

하지만 이러한 형태의 제약 조건을 사용할 경우, 더 이상 주어진 문제가 Convex Optimization에 속하지 않는다. 따라서 직관성은 다소 떨어지더라도 Convex Optimization의 조건을 충족시키기 위하여 다음과 같은 형태의 제약식을 사용한다.

$\displaystyle \max\_{\boldsymbol{\beta}, \beta\_0, \Vert\boldsymbol{\beta}\Vert=1}M \quad\text{subject to}\quad y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)\geq M(1-\xi\_i),\\; \xi\_i\geq 0,\\; \sum\_{i=1}^N\xi\_i\leq c,\\; ^\forall i$

이때 $\xi_i$는 $\mathbf{x}_i$가 잘못 위치한 정도에 비례하는 값이라고 할 수 있다. $\mathbf{x}\_i$가 오분류된 경우 $\xi\_i$는 $1$보다 큰 값을 갖는다. 따라서 $\sum\xi\_i\leq c$의 상한을 두는 것은 오분류되는 데이터의 최대 개수를 $c$개로 제한하는 것으로 해석할 수 있다.

[Optimal Seperating Hyperplanes](/ml/optimal_seperating_hyperplanes)에서 그랬던 것처럼, $M=1/\Vert\boldsymbol{\beta}\Vert$로 둠으로써 다음과 같은 형태로 문제를 변형하고, Dual Problem을 풀어 최적해를 구할 수 있다.

$\displaystyle \Leftrightarrow \min\_{\boldsymbol{\beta}, \beta\_0}\dfrac{1}{2}\Vert\boldsymbol{\beta}\Vert^2 \quad\text{subject to}\quad y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)\geq 1-\xi\_i,\\; \xi\_i\geq 0,\\; \sum\_{i=1}^N\xi\_i\leq c,\\; ^\forall i$ \
$\displaystyle \Leftrightarrow \min_{\boldsymbol{\beta}, \beta_0}\dfrac{1}{2}\Vert\boldsymbol{\beta}\Vert^2+c\sum_{i=1}^N\xi_i \quad\text{subject to}\quad y_i(\mathbf{x}_i^T\boldsymbol{\beta}+\beta_0)≥1-\xi_i,\\; \xi_i≥0,\\; ^\forall i$

Optimal Seperating Hyperplane과 마찬가지로 $\hat{G}(\mathbf{x})=\text{sign}(\hat{\beta}_0+\hat{\boldsymbol{\beta}}^T\mathbf{x})$로 새로운 데이터를 분류한다. 최적의 $c$는 Cross-Validation으로 결정한다.

## Process for Solving Dual Problem

Lagrangian Primal Function: $\displaystyle l(\boldsymbol{\beta}, \beta\_0, \boldsymbol{\xi}, \boldsymbol{\lambda}, \boldsymbol{\mu})=\dfrac{1}{2}\Vert\boldsymbol{\beta}\Vert^2+c\sum\_{i=1}^N\xi\_i-\sum\_{i=1}^N\lambda\_i\\{y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)-(1-\xi\_i)\\}-\sum\_{i=1}^N\mu\_i\xi\_i$

$\displaystyle \dfrac{\partial l}{\partial\boldsymbol{\beta}}=\boldsymbol{\beta}^T-\sum\_{i=1}^N\lambda\_iy\_i\mathbf{x}\_i^T=0 \quad\Rightarrow\quad \boldsymbol{\beta}^*=\sum\_{i=1}^N\lambda\_iy\_i\mathbf{x}\_i$ \
$\displaystyle \dfrac{\partial l}{\partial\beta\_0}=-\sum\_{i=1}^N\lambda\_iy\_i=0 \quad\Rightarrow\quad \sum_{i=1}^N\lambda\_iy\_i=0$ \
$\displaystyle \dfrac{\partial l}{\partial\boldsymbol{\xi}}=c\mathbf{1}^T-\boldsymbol{\lambda}^T-\boldsymbol{\mu}^T=0 \quad\Rightarrow\quad \boldsymbol{\lambda}=c\mathbf{1}-\boldsymbol{\mu}$

Lagrangian Dual Function: \
$\begin{aligned}
l(\boldsymbol{\beta}^\*, \beta\_0^\*, \boldsymbol{\xi}^\*, \boldsymbol{\lambda}, \boldsymbol{\mu})&=\dfrac{1}{2}\sum\_{i=1}^N\sum\_{j=1}^N\lambda_i\lambda\_jy\_iy\_j\mathbf{x}\_i^T\mathbf{x}\_j+c\sum\_{i=1}^N\xi\_i-\sum\_{i=1}^N\sum\_{j=1}^N\lambda\_i\lambda\_jy\_iy\_j\mathbf{x}\_i^T\mathbf{x}\_j-\sum\_{i=1}^N\lambda\_iy\_i\beta\_0+\sum\_{i=1}^N\lambda\_i-\sum\_{i=1}^N\lambda\_i\xi\_i-\sum\_{i=1}^N\mu\_i\xi\_i \\\\
&=\sum\_{i=1}^N\lambda\_i-\dfrac{1}{2}\sum\_{i=1}^N\sum\_{j=1}^N\lambda\_i\lambda\_jy\_iy\_j\mathbf{x}\_i^T\mathbf{x}\_j
\end{aligned}$

Dual Problem: $\displaystyle \max\_{\boldsymbol{\lambda}, \boldsymbol{\mu}}l(\boldsymbol{\beta}^\*, \beta\_0^\*, \boldsymbol{\xi}^\*, \boldsymbol{\lambda}, \boldsymbol{\mu}) \quad\text{subject to}\quad \mathbf{0}\leq\boldsymbol{\lambda}\leq c\mathbf{1},\\; \mathbf{0}\leq\boldsymbol{\mu}\leq c\mathbf{1},\\; \sum\_{i=1}^N\lambda\_iy\_i=0$

최적해 $\boldsymbol{\beta}^\*$와 $\beta\_0^\*$는 KKT Condition인 $\lambda\_i\\{y\_i(\mathbf{x}\_i^T\boldsymbol{\beta})-(1-\xi\_i)\\}=0$과 $\mu_i\xi_i=0$을 만족시켜야 한다. 따라서 $\lambda_i>0$인 경우 $y_i(\mathbf{x}_i^T\boldsymbol{\beta}+\beta_0)=1-\xi_i$이고, $\lambda_i=0$인 경우 $y_i(\mathbf{x}_i^T\boldsymbol{\beta}+\beta_0)>1-\xi_i$이다. 이를 이용하여 앞서 구한 $\displaystyle \boldsymbol{\beta}^\*=\sum\_{i=1}^N\lambda_iy_i\mathbf{x}_i$를 $\displaystyle \boldsymbol{\beta}^\*=\sum\_{i:\lambda_i\neq0}\lambda_iy_i\mathbf{x}_i$로 나타낼 수 있다. 즉, Solution Vector $\boldsymbol{\beta}^\*$는 $y_i(\mathbf{x}_i^T\boldsymbol{\beta}+\beta_0)=1-\xi_i$를 만족시키는 데이터들의 선형 결합으로 정의됨을 알 수 있다. 이러한 데이터를 Support Point(또는 Support Vector)라고 한다. $\beta_0^\*$는 Support Point 중 Margin의 경계 위에 있는 데이터에 대해 $\lambda_i\\{y_i(\mathbf{x}_i^T\boldsymbol{\beta})-(1-\xi_i)\\}=0$을 풂으로써 구할 수 있다. 일반적으로는 값의 안정성을 위해 여러 개의 $\beta_0^\*$를 구하여 평균을 내는 방식을 사용한다.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
