---
title: "Linear Regression"
date: 2021-02-20
weight: 3
---

**Linear Regression**

Linear Regression assumes the linear form of regression function as $\text{E}[Y\vert X]=X\beta$.

{{<figure src="/esl_fig_3.1.png" width="300" height="200">}}

We have to estimate $\boldsymbol{\beta}$ to fit our linear model and the most common way is to use a LSE(Least Squares Estimate). LSE is an estimate which minimizes the residual sum of squares $(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})$.

$\begin{aligned}
\dfrac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})&=\dfrac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{y}^T-\boldsymbol{\beta}^TX^T)(\mathbf{y}-X\boldsymbol{\beta}) \\\\
&=\dfrac{\partial}{\partial\beta}(\mathbf{y}^T\mathbf{y}-\boldsymbol{\beta}^TX^T\mathbf{y}-\mathbf{y}^TX\boldsymbol{\beta}+\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}) \\\\
&=-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}
\end{aligned}$

$\dfrac{\partial^2}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta}^T}(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})=2X^TX$

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

Now we will get the distribution of $\hat{\sigma}^2$. First, let $\mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}$, then it is easy to show $\mathbf{e}\sim N(\mathbf{0}, \sigma^2(I-H))$.

$\hat{\sigma}^2=\dfrac{1}{n-p-1}\mathbf{e}^T\mathbf{e}$

We will use a property of quadratic forms. Because $\mathbf{e}\sim N(\mathbf{0}, \sigma^2(I-H))$, $\vert\sigma^2(I-H)\vert>0$, $\dfrac{1}{\sigma^2}\sigma^2(I-H)$ is idempotent and $\text{tr}(I-H)=n-p-1$, we can say $\dfrac{1}{\sigma^2}\mathbf{e}^T\mathbf{e}\sim\chi_{n-p-1}^2$. Thus,

$(n-p-1)\hat{\sigma}^2\sim\sigma^2\chi^2_{n-p-1}$

With these results, we can get some confidence intervals for the parameters or can conduct some hypothesis tests.

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

**Computation with Gram-Schmidt Process**

Coefficients of linear regression can be computed by using Gram-Schmidt process.

$$X=\begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix}$$

Let the first column vector $\mathbf{1}$(or $\mathbf{x}_0$) as $\mathbf{z}_0$, we can get $\mathbf{z}_1$, which is orthogonal to $\mathbf{z}_0$. 두 번째 벡터 $\mathbf{z}_1$을 $\mathbf{1}$이 Span하는 공간에 Projection한 것을 $\text{proj}\_{X_0}\mathbf{x}_1$이라고 하면 $\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1$은 $\mathbf{z}_0$에 Orthogonal한 벡터가 된다. 따라서 $\mathbf{z}_1=\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1$이라고 할 수 있다.

$$\mathbf{z}_1=\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1=\mathbf{x}_1-\dfrac{\mathbf{x}_1\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert^2}\mathbf{z}_0$$

다음으로 $\mathbf{z}_0$와 $\mathbf{z}_1$에 대해 Orthogonal한 벡터 $\mathbf{z}_2$를 구할 수 있다. 마찬가지로 $\mathbf{z}_0$와 $\mathbf{z}_1$이 Span하는 공간에 $\mathbf{x}_2$를 Projection한 것을 $\text{proj}\_{X_1}\mathbf{x}_2$라고 하면 $\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2$는 $\mathbf{z}_0$와 $\mathbf{z}_1$에 모두 Orthogonal한 벡터가 된다. 따라서 $\mathbf{z}_2=\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2$이다.

$$\mathbf{z}_2=\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2=\mathbf{x}_2-\dfrac{\mathbf{x}_2\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert^2}\mathbf{z}_0-\dfrac{\mathbf{x}_2\cdot\mathbf{z}_1}{\Vert\mathbf{z}_1\Vert^2}\mathbf{z}_1$$

이런 과정을 반복하면 $\mathbf{z}_p$까지 구할 수 있다. $\begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix}$를 $\begin{bmatrix} \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_p \end{bmatrix}$로 굳이 바꾸어 준 이유는 $\mathbf{y}$를 각각의 $\mathbf{z}_i$에 Projection하여 $i$ 번째 $X$ 변수의 회귀 계수를 $\hat{\beta}_i=\dfrac{\mathbf{y}\cdot\mathbf{z}_i}{\Vert\mathbf{z}_i\Vert^2}$로 쉽게 계산할 수 있기 때문이다. 이유는 아래의 그림을 통해 확인할 수 있다.

우리는 $\hat{\mathbf{y}}$을 $a\mathbf{1}+b\mathbf{x}_1$라고 나타낼 수 있다. 이때 $b\mathbf{x}_1$은 $(c\mathbf{1}-a\mathbf{1})+d\mathbf{z}_1$과 같다. 이를 앞의 식에 대입해보면 $\hat{\mathbf{y}}$은 곧 $c\mathbf{1}+d\mathbf{z}_1$과 같다는 것을 확인할 수 있다. 즉, $\mathbf{y}$를 $\mathbf{x}_1$에 Projection 했을 때의 회귀 계수와 $\mathbf{z}_1$에 Projection 했을 때의 회귀 계수는 같은 값을 갖게 된다. 이에 더해, $\mathbf{z}_1, \ldots, \mathbf{z}_p$가 서로 Orthogonal 하기 때문에 $\mathbf{y}$를 $\mathbf{z}_1, \ldots, \mathbf{z}_p$에 Projection 했을 때의 회귀 계수는 $\mathbf{y}$를 각각의 $\mathbf{z}_i$에 Projection 했을 때의 회귀 계수와 같다.

**Reference**

1. Elements of Statistical Learning
2. https://en.wikipedia.org/wiki/Gauss-Markov_theorem