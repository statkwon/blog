---
title: "Linear Regression"
date: 2021-02-20
categories:
  - "ML"
tags:
sidebar: false
---

Linear Regression is a bit classical model, but still has some advantages. It's prediction performance can outperform the latest methods in some specific situations such as small data, low SNR, or sparse data. Also, it can be expanded to nonlinear models by transforming the inputs.

Linear Regression assumes the linear form of regression function as $\text{E}[Y\vert X]=X\boldsymbol{\beta}$.

{{<figure src="/ml/lr1.png" width="300">}}

We have to estimate $\boldsymbol{\beta}$ to fit our linear model and the most common way is to use a LSE(Least Squares Estimate). LSE is an estimate which minimizes the residual sum of squares $(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})$.

$\begin{aligned}
\dfrac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})&=\dfrac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{y}^T-\boldsymbol{\beta}^TX^T)(\mathbf{y}-X\boldsymbol{\beta}) \\\\
&=\dfrac{\partial}{\partial\beta}(\mathbf{y}^T\mathbf{y}-\boldsymbol{\beta}^TX^T\mathbf{y}-\mathbf{y}^TX\boldsymbol{\beta}+\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}) \\\\
&=-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}
\end{aligned}$

$\dfrac{\partial^2}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta}^T}(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})=2X^TX$

Thus, if we solve the equation $-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}=0$, we can get the unique solution $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$ when $X^TX$ is nonsingular. This condition is same as that $X$ has a full colunm rank. If don't, a solution still exists, but is not unique. This is the reason why we have to consider the variables to be uncorrelated while fitting the regression model.

## Least Squares Estimate

It is useful to think about the geometric meaning of least squares estimate.

{{<figure src="/ml/lr2.png" width="400">}}

When the inverse exists, we can express $\hat{\mathbf{y}}=X\hat{\boldsymbol{\beta}}$ as $X(X^TX)^{-1}X^T\mathbf{y}$. If we let $H=X(X^TX)^{-1}X^T$, it is not that difficult to recognize that $H$ is a projection matrix due to the fact that $H$ is idempotent and symmetric. Thus, $\hat{\mathbf{y}}$ can be thought of as the result of projection of $\mathbf{y}$ onto the column space of $X$. Furthermore, if $X$ does not have full column rank, $\hat{\mathbf{y}}$ can still be regarded as the result of projection of $\mathbf{y}$, but not an orthogonal one.

## Statistical Inferences

With a bit more strict assumption, we can make some statistical inferences about model parameters. From now on, we will assume that $\boldsymbol{\epsilon}$ follows a multivariate normal distribution with the mean vector $\mathbf{0}$ and the covariance matrix $\sigma^2I_n$. Then it is easy to show that $\hat{\boldsymbol{\beta}}\sim N_p(\boldsymbol{\beta}, \sigma^2(X^TX)^{-1})$.

$\begin{aligned}
\text{E}[\hat{\boldsymbol{\beta}}]&=\text{E}[(X^TX)^{-1}X^T\mathbf{y}] \\\\
&=(X^TX)^{-1}X^T\text{E}[\mathbf{y}] \\\\
&=(X^TX)^{-1}X^TX\boldsymbol{\beta} \\\\
&=\boldsymbol{\beta}
\end{aligned}$

$\begin{aligned}
\text{Cov}(\hat{\boldsymbol{\beta}})&=\text{Cov}((X^TX)^{-1}X^T\mathbf{y}) \\\\
&=(X^TX)^{-1}X^T\text{Cov}(\mathbf{y})X(X^TX)^{-1} \\\\
&=\sigma^2(X^TX)^{-1}
\end{aligned}$

Now we will get the distribution of $\hat{\sigma}^2$. First, let $\mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}$, then it is easy to show $\mathbf{e}\sim N(\mathbf{0}, \sigma^2(I-H))$.

$\hat{\sigma}^2=\dfrac{1}{n-p-1}\mathbf{e}^T\mathbf{e}$

We will use a property of quadratic forms. Because $\mathbf{e}\sim N(\mathbf{0}, \sigma^2(I-H))$, $\vert\sigma^2(I-H)\vert>0$, $\dfrac{1}{\sigma^2}\sigma^2(I-H)$ is idempotent and $\text{tr}(I-H)=n-p-1$, we can say $\dfrac{1}{\sigma^2}\mathbf{e}^T\mathbf{e}\sim\chi_{n-p-1}^2$. Thus,

$(n-p-1)\hat{\sigma}^2\sim\sigma^2\chi^2_{n-p-1}$

With these results, we can get some confidence intervals for the parameters or can conduct some hypothesis tests.

## Gauss-Markov Theorem

We use the LSE because it has some good properties. Gauss-Markov theorem states that the LSE is BLUE, which means Best Linear Unbiased Estimate. According to this theorem, LSE has the smallest variance among all linear unbiased estimates. We've already showed that LSE is an unbaised estimate, so we will just proved for the smallest variance.

Let $\tilde{\boldsymbol{\beta}}=C\mathbf{y}$ is an unbiased estimate for $\boldsymbol{\beta}$, where $C=(X^TX)^{-1}X^T+D$ and $D$ is not a zero matrix.

$\begin{aligned}
\text{E}[\tilde{\boldsymbol{\beta}}]&=\text{E}\left[\left\((X^TX)^{-1}X^T+D\right\)\mathbf{y}\right] \\\\
&=\text{E}\left[\left\((X^TX)^{-1}X^T+D\right\)(X\boldsymbol{\beta}+\boldsymbol{\epsilon})\right] \\\\
&=\left\((X^TX)^{-1}X^T+D\right\)X\boldsymbol{\beta}+\left\((X^TX)^{-1}X^T+D\right\)\text{E}[\boldsymbol{\epsilon}] \\\\
&=(X^TX)^{-1}X^TX\boldsymbol{\beta}+DX\boldsymbol{\beta} \\\\
&=(I+DX)\boldsymbol{\beta}
\end{aligned}$

We assumed that $\tilde{\boldsymbol{\beta}}$ is an unbiased estimate of $\boldsymbol{\beta}$, so $DX$ should be $0$.

$\begin{aligned}
\text{Var}(\tilde{\boldsymbol{\beta}})&=\sigma^2\left\((X^TX)^{-1}X^T+D\right\)\left\(X(X^TX)^{-1}+D^T\right\) \\\\
&=\sigma^2\left\((X^TX)^{-1}X^TX(X^TX)^{-1}+(X^TX)^{-1}X^TD^T+DX(X^TX)^{-1}+DD^T\right\) \\\\
&=\sigma^2(X^TX)^{-1}+\sigma^2(X^TX)^{-1}(DX)^T+\sigma^2DX(X^TX)^{-1}+\sigma^2DD^T \\\\
&=\sigma^2(X^TX)^{-1}+\sigma^2DD^T \\\\
&=\text{Var}(\hat{\boldsymbol{\beta}})+\sigma^2DD^T
\end{aligned}$

This equatinos hold due to the fact that $DX=0$. Thus, it is evident that $\text{Var}(\hat{\boldsymbol{\beta}})≤\text{Var}(\tilde{\boldsymbol{\beta}})$, which means that $\hat{\boldsymbol{\beta}}$ has the smallest variance among all linear unbiased estimates.

Nonetheless, there might be another better estimate for $\boldsymbol{\beta}$. We proved that the LSE has the smallest variance among unbiased estimates, not among the all estimates. We still has a possibility to trade a little bias for a larger reduction in variance. Ridge regression is one of the example of this possibility.

## Regression by Successive Orthogonalization

{{<figure src="/ml/lr3.png" width="400">}}

If all column vectors of $X$ are orthogonal to each other, it will be much easier to get coefficients of multiple linear regression. We can just project $\mathbf{y}$ onto each vector $\mathbf{x}_i$ and get $\hat{\beta}_i=\dfrac{\mathbf{y}\cdot\mathbf{z}_i}{\mathbf{z}_i\cdot\mathbf{z}_i}$, due to the fact that orthogonal inputs do not effect on each other's parameter estimates. However, orthogonal inputs are actually an impossible situation. Though, we could use this idea to make an efficient way for parameter estimation in multiple regression.

$\begin{bmatrix} \mathbf{x}_0 & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix}\quad\Rightarrow\quad\begin{bmatrix} \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_p \end{bmatrix}$

By using Gram-Schmidt process, we can transform the column vectors of $X$ to the orthogonal vectors. Then, $\hat{\beta}_p$ can be calculated by projecting $\mathbf{y}$ onto $\mathbf{z}_p$, which can be written as $\hat{\beta}_p=\dfrac{\mathbf{y}\cdot\mathbf{z}_p}{\mathbf{z}_p\cdot\mathbf{z}_p}$. If we change the order of the column vectors of $X$, this can be applied to any coefficient $\hat{\beta}_i$. Now we can state that $\hat{\beta}_i$ represents the additional contribution of $\mathbf{x}_j$ on $\mathbf{y}$, after $\mathbf{x}_j$ has been adjusted for $\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}\_{j-1}, \mathbf{x}\_{j+1}, \ldots, \mathbf{x}_p$.

This can be proved with calculation via $QR$-decomposition of $X$.

$\begin{aligned}
X&=\begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix} \\\\
&=\begin{bmatrix} \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_p \end{bmatrix}\begin{bmatrix} 1 & \dfrac{\mathbf{x}_1\cdot\mathbf{z}_0}{\mathbf{z}_0\cdot\mathbf{z}_0} & \cdots & \dfrac{\mathbf{x}_p\cdot\mathbf{z}_0}{\mathbf{z}_0\cdot\mathbf{z}_0} \\\\ 0 & 1 & \cdots & \dfrac{\mathbf{x}_1\cdot\mathbf{z}_1}{\mathbf{z}_1\cdot\mathbf{z}_1} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & 1 \end{bmatrix} \\\\
&=\begin{bmatrix} \dfrac{\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert} & \dfrac{\mathbf{z}_1}{\Vert\mathbf{z}_1\Vert} & \cdots & \dfrac{\mathbf{z}_p}{\Vert\mathbf{z}_p\Vert} \end{bmatrix}\begin{bmatrix} \Vert\mathbf{z}_0\Vert & \dfrac{\mathbf{x}_1\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert} & \cdots & \dfrac{\mathbf{x}_p\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert} \\\\ 0 & \Vert\mathbf{z}_1\Vert & \cdots & \dfrac{\mathbf{x}_p\cdot\mathbf{z}_1}{\Vert\mathbf{z}_1\Vert} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & \Vert\mathbf{z}_p\Vert \end{bmatrix} \\\\
&=QR
\end{aligned}$

Now we solve the new equation $\mathbf{y}=QR\boldsymbol{\beta}$, rather than the original one. Because $Q$ is an orthogonal matrix, we can write as $R\boldsymbol{\beta}=Q^T\mathbf{y}$.

$\begin{bmatrix} \Vert\mathbf{z}_0\Vert & \dfrac{\mathbf{x}_1\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert} & \cdots & \dfrac{\mathbf{x}_p\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert} \\\\ 0 & \Vert\mathbf{z}_1\Vert & \cdots & \dfrac{\mathbf{x}_p\cdot\mathbf{z}_1}{\Vert\mathbf{z}_1\Vert} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & \Vert\mathbf{z}_p\Vert \end{bmatrix}\begin{bmatrix} \beta_0 \\\\ \beta_1 \\\\ \vdots \\\\ \beta_p \end{bmatrix}=\begin{bmatrix} \dfrac{\mathbf{y}\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert} \\\\ \dfrac{\mathbf{y}\cdot\mathbf{z}_1}{\Vert\mathbf{z}_1\Vert} \\\\ \vdots \\\\ \dfrac{\mathbf{y}\cdot\mathbf{z}_p}{\Vert\mathbf{z}_p\Vert} \end{bmatrix}$

$R$ is an upper-triangular matrix, so we just use the back-substitution to solve the equation above. For example, first we can get $\hat{\beta}_p=\dfrac{\mathbf{y}_p\cdot\mathbf{z}_p}{\Vert\mathbf{z}_p\Vert^2}$, then $\hat{\beta}\_{p-1}=\dfrac{\mathbf{y}\cdot\mathbf{z}\_{p-1}}{\Vert\mathbf{z}\_{p-1}\Vert^2}-\hat{\beta}_p\dfrac{\mathbf{x}_p\cdot\mathbf{z}\_{p-1}}{\Vert\mathbf{z}\_{p-1}\Vert^2}$. Repeating this process, we can obtain $\hat{\boldsymbol{\beta}}$ without calculating the inverse.

## Python Code for Linear Regression

Github Link: [MyLinearRegression.ipynb](https://github.com/statkwon/ML_Study/blob/master/MyLinearRegression.ipynb)

```py
import numpy as np

class MyLinearRegerssion:
    def fit(self, X_train, y_train):
        ones = np.ones(len(X_train))
        X_train = np.array(X_train)
        X_train = np.column_stack((np.ones(len(X_train)), X_train))
        y_train = np.array(y_train)
        self.beta = np.linalg.inv(np.transpose(X_train).dot(X_train)).dot(np.transpose(X_train)).dot(y_train)
        
    def predict(self, X_test):
        ones = np.ones(len(X_test))
        X_test = np.array(X_test)
        X_test = np.column_stack((np.ones(len(X_test)), X_test))
        return X_test.dot(self.beta)
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. Weisberg, S. (2005). Applied linear regression (Vol. 528). John Wiley & Sons.
3. [https://en.wikipedia.org/wiki/Gauss-Markov_theorem](https://en.wikipedia.org/wiki/Gauss-Markov_theorem)
