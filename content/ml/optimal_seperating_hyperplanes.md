---
title: "Optimal Seperating Hyperplanes"
date: 2022-01-22
categories:
  - "ML"
tags:
  - "SVM"
sidebar: false
---

Optimal Seperating Hyperplane은 Perceptron Algorithm의 해가 유일하지 않다는 단점을 보완하기 위해 고안된 방식이다. Perceptron Learning Algorithm과 마찬가지로 $y\in\\{-1, 1\\}$의 Binary Classification 문제에 적용되지만, 데이터와 분류 경계선 사이의 빈 공간을 뜻하는 'Margin'이라는 새로운 개념을 도입해다는 점에서 차이가 있다. Optimal Seperating Hyperplane의 목표는 이 공간을 최대화하는 Hyperplane을 찾는 것이다.

{{<figure src="/ml/osh1.png" width="400">}}

Thus, we have to find optimal $\boldsymbol{\beta}$ and $\beta_0$ which maximizes $M$ when the distance between each points and the boundary is greater than or equal to $M$.

$\begin{aligned}
&\max\_{\boldsymbol{\beta}, \beta\_0, \Vert\boldsymbol{\beta}\Vert=1}M \quad\text{subject to}\quad y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)\geq M,\\; ^\forall i \\\\
&\Leftrightarrow \max\_{\boldsymbol{\beta}, \beta\_0}M \quad\text{subject to}\quad \dfrac{1}{\Vert\boldsymbol{\beta}\Vert}y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)\geq M,\\; ^\forall i \\\\
&\Leftrightarrow \max\_{\boldsymbol{\beta}, \beta\_0}M \quad\text{subject to}\quad y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)\geq M\Vert\boldsymbol{\beta}\Vert,\\; ^\forall i
\end{aligned}$

If we set $\Vert\boldsymbol{\beta}\Vert=1/M$, we can convert this optimization problem as below.

$\displaystyle \Leftrightarrow \min\_{\boldsymbol{\beta}, \beta\_0}\dfrac{1}{2}\Vert\boldsymbol{\beta}\Vert^2 \quad\text{subject to}\quad y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)\geq 1,\\; ^\forall i$

Because the objective function and the constraint set are both convex here, it is a convex optimization problem. Also the strong duality holds since the inequality constraint is an affine function of $\boldsymbol{\beta}$ and $\beta_0$. Therefore, we can solve a dual problem, which is much easier to solve, instead of the original problem to get the optimal solution. Then the Optimal Seperating Hyperplane classifies the new observations with a function $\hat{G}(\mathbf{x})=\text{sign}(\hat{\beta}_0+\hat{\boldsymbol{\beta}}^T\mathbf{x})$.

**Process for Solving Dual Problem**

Lagrangian Primal Function: $\displaystyle l(\boldsymbol{\beta}, \beta\_0, \boldsymbol{\lambda})=\dfrac{1}{2}\Vert\boldsymbol{\beta}\Vert^2-\sum\_{i=1}^N\lambda_i\\{y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)-1\\}$

$\displaystyle \dfrac{\partial l}{\partial\boldsymbol{\beta}}=\boldsymbol{\beta}^T-\sum\_{i=1}^N\lambda\_iy\_i\mathbf{x}\_i^T=0 \quad\Rightarrow\quad \boldsymbol{\beta}^*=\sum\_{i=1}^N\lambda_iy_i\mathbf{x}\_i$ \
$\displaystyle \dfrac{\partial l}{\partial\beta\_0}=-\sum\_{i=1}^N\lambda\_iy\_i=0 \quad\Rightarrow\quad \sum\_{i=1}^N\lambda\_iy\_i=0$

Lagrangian Dual Function: \
$\begin{aligned}
l(\boldsymbol{\beta}^\*, \beta\_0^*, \boldsymbol{\lambda})&=\dfrac{1}{2}\sum\_{i=1}^N\sum\_{j=1}^N\lambda\_i\lambda\_jy\_iy\_j\mathbf{x}\_i^T\mathbf{x}\_j-\sum\_{i=1}^N\sum\_{j=1}^N\lambda\_i\lambda\_jy\_iy\_j\mathbf{x}\_i^T\mathbf{x}\_j-\sum\_{i=1}^N\lambda\_iy\_i\beta\_0+\sum\_{i=1}^N\lambda\_i \\\\
&=\sum\_{i=1}^N\lambda\_i-\dfrac{1}{2}\sum\_{i=1}^N\sum\_{j=1}^N\lambda\_i\lambda\_jy\_iy\_j\mathbf{x}\_i^T\mathbf{x}\_j
\end{aligned}$

Dual Problem: $\displaystyle \max_{\boldsymbol{\lambda}}l(\boldsymbol{\beta}^\*, \beta_0^*, \boldsymbol{\lambda}) \quad\text{subject to}\quad \boldsymbol{\lambda}≥\mathbf{0},\\; \sum_{i=1}^N\lambda_iy_i=0$

The optimal solution $\boldsymbol{\beta}^\*$ and $\beta\_0^*$ should satisfy the KKT Condition, $\lambda\_i\\{y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)-1\\}=0$. Thus, when $\lambda\_i>0$, $y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)$ should be $1$ and this implies that $\mathbf{x}\_i$ is on the boundary of the margin. On the other hand, when $\lambda\_i=0$, $y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)$ is greater than $1$ and $\mathbf{x}\_i$ is not on the boundary of the margin this time. We can use this fact to change $\displaystyle \boldsymbol{\beta}^\*=\sum\_{i=1}^N\lambda\_iy\_i\mathbf{x}\_i$ to $\displaystyle \boldsymbol{\beta}^\*=\sum\_{i:\lambda\_i\neq0}\lambda\_iy\_i\mathbf{x}\_i$. That is, the solution vector $\boldsymbol{\beta}^\*$ is defined as a linear combination of the points on the boundary of the margin. We call these points as Support Point(or Support Vector). $\beta\_0^\*$ can be obtained by solving $\lambda\_i\\{y\_i(\mathbf{x}\_i^T\boldsymbol{\beta}+\beta\_0)-1\\}=0$ for any of the support points.

However, as we can easily check in the constraint, Optimal Seperating Hyperplane also does not have a feasible solution when the data is not linearly seperable.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
