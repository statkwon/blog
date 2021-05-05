---
title: "Linear Regression"
date: 2021-03-10
draft: false
---

**Linear Regression**

Linear Regression assumes the linear relationship between the inputs and output.

$$\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon}\sim (\mathbf{0}, \sigma^2I_n)$$

Here $\mathbf{y}$ and $\boldsymbol{\epsilon}$ is a $n\times 1$ vector, $X$ is a $n\times p$ matrix, and $\boldsymbol{\beta}$ is a $p\times 1$ vector.

We have to estimate $\boldsymbol{\beta}$ to fit our linear model and the most common way is to use a LSE(Least Squares Estimate). LSE is an estimate which minimizes $(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})$.

$\begin{aligned}
\dfrac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})&=\dfrac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{y}^T-\boldsymbol{\beta}^TX^T)(\mathbf{y}-X\boldsymbol{\beta}) \\\\
&=\dfrac{\partial}{\partial\beta}(\mathbf{y}^T\mathbf{y}-\boldsymbol{\beta}^TX^T\mathbf{y}-\mathbf{y}^TX\boldsymbol{\beta}+\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}) \\\\
&=-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}
\end{aligned}$

Thus, if we solve the equation $-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}=0$, we can get the unique solution $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$ when $X^TX$ is nonsingular. This condition is same as that $X$ has a full colunm rank. If don't, a solution still exists, but is not unique. This is the reason why we have to consider the variables to be uncorrelated while fitting the regression model.

---

**Least Squares Estimate**

It is useful to think about the geometric meaning of least squares estimate.

{{<figure src="/esl_fig_3.2.png" width="400" height="200">}}

When the inverse exists, we can express $\hat{\mathbf{y}}=X\hat{\boldsymbol{\beta}}$ as $X(X^TX)^{-1}X^T\mathbf{y}$. If we let $H=X(X^TX)^{-1}X^T$, it is not that difficult to recognize that $H$ is a projection matrix due to the fact that $H$ is idempotent and symmetric. Thus, $\hat{\mathbf{y}}$ can be thought of as the result of projection of $\mathbf{y}$ onto the column space of $X$. Furthermore, if $X$ does not have full column rank, $\hat{\mathbf{y}}$ can still be regarded as the result of projection of $\mathbf{y}$, but not an orthogonal one.

---

**Statistical Inferences**

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

Distribution of $\hat{\sigma}^2$ can also be obtained, $(n-p-1)\hat{\sigma}^2\sim\sigma^2\chi^2_{n-p-1}$. With these results, we can get some confidence intervals for the parameters or can conduct some hypothesis tests.

---

**Gauss-Markov Theorem**

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

---

**Reference**

1. Elements of Statistical Learning
2. https://en.wikipedia.org/wiki/Gauss-Markov_theorem