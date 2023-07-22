---
title: "Local Regression"
date: 2021-03-20
categories:
  - "ML"
tags:
sidebar: false
---

Boundary issues of Nadaraya-Watson kernel regression can be solved by fitting a straight line rather than constants locally. This is the concept of local regression which fits a seperate weighted least squares at each target point $\mathbf{X}_0$. For convenience, let's consider the one-dimensional input space from now on.

## Local Linear Regression

$\displaystyle\underset{\boldsymbol{\beta}(x_0)}{\text{argmin}}(\mathbf{y}-X\boldsymbol{\beta}(x_0))^TW(x_0)(\mathbf{y}-X\boldsymbol{\beta}(x_0))$

The coefficients of local linear regression can be obtained by finding a optimal solution of the problem above, where $W$ is a diagonal matrix whose $i$th diagonal elements are $K_\lambda(x_0, x_i)$. It can be done by the same way as in the linear regression.

$\begin{aligned}
&\dfrac{\partial}{\partial\boldsymbol{\beta}(x_0)}(\mathbf{y}-X\boldsymbol{\beta}(x_0))^TW(x_0)(\mathbf{y}-X\boldsymbol{\beta}(x_0)) \\\\
&=\dfrac{\partial}{\partial\boldsymbol{\beta}(x_0)}(\mathbf{y}^TW(x_0)\mathbf{y}-\boldsymbol{\beta}^TX^TW(x_0)\mathbf{y}-\mathbf{y}^TW(x_0)X\boldsymbol{\beta}+\boldsymbol{\beta}^TX^TW(x_0)X\boldsymbol{\beta}) \\\\
&=-2X^TW(x_0)\mathbf{y}+2X^TW(x_0)X\boldsymbol{\beta}
\end{aligned}$

Without a doubt, $\hat{\boldsymbol{\beta}}(x_0)=(X^TW(x_0)X)^{-1}X^TW(x_0)\mathbf{y}$ is the solution. Now we can get $\hat{f}(x_0)$ as below.

$\begin{aligned}
\hat{f}(x_0)&=\mathbf{x}\_0^T(X^TW(x_0)X)^{-1}X^TW(x_0)\mathbf{y} \\\\
&=\mathbf{l}(x_0)^T\mathbf{y} \\\\
&=\sum_{i=1}^nl_i(x_0)y_i
\end{aligned}$

where $\mathbf{x}_0=\begin{bmatrix} 1 \\\\ x_0 \end{bmatrix}$. Sometimes we call $l_i(x_0)$ as equivalent kernel.

Local linear regression can be the alternative of Nadaraya-Watson kernel regression because it automatically reduces the bias to first order. Below is the proof for this property.

1\) $\displaystyle\sum_{i=1}^nl_i(x_0)=1$, $\displaystyle\sum_{i=1}^nl_i(x_0)x_i=x_0$

We've already showed that $\displaystyle\mathbf{x}\_0^T(X^TW(x_0)X)^{-1}X^TW(x_0)\mathbf{y}=\sum_{i=1}^nl_i(x_0)y_i$. Now let $\mathbf{v}_j=\begin{bmatrix} x_1^j & x_2^j & \cdots & x_n^j \end{bmatrix}^T$.

Then we can show that $\displaystyle\mathbf{x}\_0^T(X^TW(x_0)X)^{-1}X^TW(x_0)\mathbf{v}_j=\sum\_{i=1}^nl_i(x_0)x_i^j$.

$\begin{aligned}
\mathbf{x}_0^T(X^TW(x_0)X)^{-1}X^TW(x_0)\begin{bmatrix} \mathbf{v}_0 & \mathbf{v}_1 \end{bmatrix}&=\mathbf{x}_0^T(X^TW(x_0)X)^{-1}X^TW(x_0)X \\\\
&=\mathbf{x}_0^T \\\\
&=\begin{bmatrix} 1 & x_0 \end{bmatrix}
\end{aligned}$

$\displaystyle\therefore \sum_{i=1}^nl_i(x_0)=1, \\; \sum_{i=1}^nl_i(x_0)x_i=x_0$

2\) $\text{Bias}(\hat{f}(x_0))=\dfrac{f''(x_0)}{2}\sum_{i=1}^n(x_i-x_0)^2l_i(x_0)+R$

By using taylor expansion, we can write down the expectation of $\hat{f}(x_0)$ as below.

$\begin{aligned}
\text{E}[\hat{f}(x_0)]&=\sum_{i=1}^nl_i(x_0)f(x_i) \\\\
&=f(x_0)\sum_{i=1}^nl_i(x_0)+f'(x_0)\sum_{i=1}^n(x_i-x_0)l_i(x_0)+\dfrac{f''(x_0)}{2}\sum_{i=1}^n(x_i-x_0)^2l_i(x_0)+R
\end{aligned}$

We showed that $\displaystyle\sum_{i=1}^nl_i(x_0)=1$ and $\displaystyle\sum_{i=1}^nl_i(x_0)x_i=x_0$.

$\displaystyle\sum_{i=1}^n(x_i-x_0)l_i(x_0)=\sum_{i=1}^nx_il_i(x_0)-x_0\sum_{i=1}^nl_i(x_0)=0$

$\displaystyle\therefore\text{E}[\hat{f}(x_0)]=f(x_0)+\dfrac{f''(x_0)}{2}\sum_{i=1}^n(x_i-x_0)^2l_i(x_0)+R$

This implies that the bias of $\hat{f}(x_0)$ only depends on the second derivative and the higher-order terms.

## Local Polynomial Regression

The only difference between local linear regression and local polynomial regression is the maximum degree of the model. We just substitute $X$ with $\begin{bmatrix} \mathbf{1} & \mathbf{x} & \mathbf{x}^2 & \cdots & \mathbf{x}^d \end{bmatrix}$. Then the bias of $d$th-order local polynomial only depends on the $(d+1)$th derivative and the higher-order terms. We can prove this by the similar way as in the local linear.

Usually local linear fits are useful to dramatically decrease the bias at the boundaries, while local quadratic fits tend to be most helpful in reducing bias due to curvature in the interior of the domain.

The benefit of automatic kernel carpentry comes out as the dimension gets highger. A tendency of data getting closer to the boundary makes the asymmetry problem more serious, but local regression can take care of it. However, if the dimension of the input space becomes larger than three, local regression becomes less useful as the range for a neighborhood gets larger.

## Python Code for Local Regression

Github Link: [MyLocalRegression.ipynb](https://github.com/statkwon/ML_Study/blob/master/MyLocalRegression.ipynb)

```py
import math
import numpy as np

class MyLocalRegression:
    def __init__(self, kernel='Tri-Cube', width=10):
        self.kernel = kernel
        self.width = width
    
    def tricube(self, x):
        return np.where(abs(x) <= 1, (1-abs(x)**3)**3, 0)
    
    def epanechnikov(self, x):
        return np.where(abs(x) <= 1, 0.75*(1-x**2), 0)
    
    def gaussian(self, x):
        return 1/np.sqrt(2*math.pi)*np.exp(-0.5*(x**2))
    
    def predict(self, X_train, y_train, X_test):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_pred = np.array([])
        for i in range(len(X_test)):
            t = abs(X_train-X_test[i])/self.width
            if self.kernel == 'Tri-Cube':
                d = self.tricube(t)
            elif self.kernel == 'Epanechnikov':
                d = self.epanechnikov(t)
            else:
                d = self.gaussian(t)
            X_train_nonzero = np.column_stack((np.power(X_train[d != 0], 0), X_train[d != 0]))
            y_train_nonzero = y_train[d != 0]
            W = np.diag(d[d != 0])
            y_pred = np.append(y_pred, np.transpose(np.array((1, X_test[i]))).dot(np.linalg.inv(np.transpose(X_train_nonzero).dot(W).dot(X_train_nonzero))).dot(np.transpose(X_train_nonzero)).dot(W).dot(y_train_nonzero))
        return y_pred
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
