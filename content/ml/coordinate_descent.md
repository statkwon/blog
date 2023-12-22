---
title: "Coordinate Descent"
date: 2023-12-21
lastmod: 2023-12-21
categories:
  - "ML"
tags:
  - "Optimization"
sidebar: false
---

## Coordinate Descent

convex & differentiable한 함수 $g$와 convex한 함수 $h$에 대하여 $f(\mathbf{x})=g(\mathbf{x})+\sum\_{i=1}^nh\_i(x\_i)$ 일 때, 각 좌표축에 대하여 $f$를 최소화시킨 점 $\mathbf{x}$는 항상 global minimizer이다. (증명은 첫 번째 reference 참고) 따라서 다음과 같은 과정을 반복하여 $f$의 global minimizer를 근사할 수 있다.

> For $k=1, 2, 3, \ldots$
>
> $x\_1^{(k)}=\underset{x\_1}{\text{argmin}}f(x\_1, x\_2^{(k-1)}, x\_3^{(k-1)}, \ldots, x\_n^{(k-1)})$ \
> $x\_2^{(k)}=\underset{x\_2}{\text{argmin}}f(x\_1^{(k)}, x\_2, x\_3^{(k-1)}, \ldots, x\_n^{(k-1)})$ \
> $x\_3^{(k)}=\underset{x\_3}{\text{argmin}}f(x\_1^{(k)}, x\_2^{(k)}, x\_3, \ldots, x\_n^{(k-1)})$ \
> $\cdots$ \
> $x\_n^{(k)}=\underset{x\_n}{\text{argmin}}f(x\_1^{(k)}, x\_2^{(k)}, x\_3^{(k)}, \ldots, x\_n)$

초깃값 $\mathbf{x}^{(0)}$로는 적당한 값을 사용한다.

## Python Code for Example

Coordinate Descent 방식을 사용하여 Linear regression의 회귀 계수를 구해보자. 우리의 목표는 $f(\boldsymbol{\beta})=(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})$를 최소화하는 $\boldsymbol{\beta}$를 찾는 것이므로,

$$\begin{align}
\nabla\_if(\boldsymbol{\beta})&=\mathbf{x}\_i^T(\mathbf{y}-X\boldsymbol{\beta}) \\\\
&=\mathbf{x}\_i^T(\mathbf{y}-\mathbf{x}\_i\beta\_i-X\_{-i}\boldsymbol{\beta}\_{-i}) \\\\
&=0
\end{align}$$

을 만족하는 $\beta\_i=\dfrac{\mathbf{x}\_i^T(\mathbf{y}-X\_{-i}\boldsymbol{\beta}\_{-i})}{X\_i^TX\_i}$로 업데이트를 진행하면 된다.

```py
import numpy as np
from sklearn.datasets import make_regression
```

```py
def coordinate_descent(X: np.ndarray, y: np.ndarray, n_iter: int):
    _, p = X.shape
    beta = np.zeros(p)
    
    for _ in range(n_iter):
        for i in range(p):
            X_i = np.column_stack((X[:, :i], X[:, (i + 1):]))
            beta_i = np.concatenate((beta[:i], beta[(i + 1):]))
            beta[i] = (X[:, i].T @ (y - X_i @ beta_i)) / (X[:, i].T @ X[:, i])

    return beta
```

```py
X, y, coef = make_regression(n_samples=100, n_features=5, noise=0.1, coef=True, random_state=0)
print(coef)  ## [45.70587613 85.71249175 97.99623263 11.73155642 42.37063535]
beta = coordinate_descent(X, y, 10)
print(beta)  ## [45.69972366 85.72175552 98.00526381 11.72151389 42.37038922]
```

---

**Reference**

1. https://convex-optimization-for-all.github.io/contents/chapter23/2021/03/28/23_01_Coordinate_descent/
