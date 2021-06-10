---
title: "Logistic Regression"
date: 2021-03-01
weight: 7
---

**Linear Regression for Qualitative Output**

Our goal is to model the posterior probability $P(Y_k\vert\mathbf{X})$ or the discriminant function $\delta_k(\mathbf{X})$, because we want to set a decision boundary as $\\{\mathbf{X}:P(Y_k\vert\mathbf{X})=P(Y_l\vert\mathbf{X})\\}$ or $\\{\mathbf{X}:\delta_k(\mathbf{X})=\delta_l(\mathbf{X})\\}$. We will classify $\mathbf{X}$ to the class with the largest value of $P(Y_k\vert\mathbf{X})$ or $\delta_k(\mathbf{X})$.

First, we can consider the case fitting a linear model for the posterior probability as $P(Y_k\vert\mathbf{X})=\mathbf{X}\boldsymbol{\beta}_k$ and it is quite reasonable in that

$\text{E}[Y_k\vert\mathbf{X}]=P(Y_1\vert\mathbf{X})\cdot0+\cdots+P(Y_k\vert\mathbf{X})\cdot1+\cdots+P(Y_K\vert\mathbf{X})\cdot0=P(Y_k\vert\mathbf{X})$.

The figure below depicts this situation.


{{<figure src="/fig9.jpeg" width="600">}}

However, this method has some disadvantages. We know that a probability should be between $0$ and $1$ and the sum of all probabilities should be $1$. If we assume the linear form of posterior probability, it only satisfies that $\sum_k\hat{P}(Y_k\vert\mathbf{X})=1$ for any $\mathbf{X}$. The probability can be negative or greater than $1$ as the figure above. Furthermore, this method cannot be used when the number of classes are larger than $3$. We can handle this problems by using another form of function.

---

**Logistic Regression**

Logistic regression assumes the posterior probability to have a form of a logistic function which ensures that they sum to one and remain in $[0, 1]$.

$P(Y_k\vert\mathbf{X})=\dfrac{\exp{(\mathbf{X}\boldsymbol{\beta}\_k)}}{1+\sum_{l=1}^{K-1}\exp{(\mathbf{X}\boldsymbol{\beta}_l)}} \qquad P(Y_K\vert\mathbf{X})=\dfrac{1}{1+\sum_{l=1}^{K-1}\exp{(\mathbf{X}\boldsymbol{\beta}_l)}}$

When $K=2$, this model is especially simple, since there is only a single linear function and we will concentrate only on this case from now on.

$P(Y_1\vert\mathbf{X})=\dfrac{\exp{(\mathbf{X}\boldsymbol{\beta})}}{1+\exp{(\mathbf{X}\boldsymbol{\beta})}} \qquad P(Y_0\vert\mathbf{X})=\dfrac{1}{1+\exp{(\mathbf{X}\boldsymbol{\beta})}}$

{{<figure src="/fig10.jpeg" width="600">}}

To estimate the coefficients, we will assume that our target variable $Y$ follows a binomial distribution with $p_1(\mathbf{x}_i;\boldsymbol{\beta})=P(Y_1\vert\mathbf{X})$. Then the log-likelihood can be written as

$\begin{aligned}
l(\boldsymbol{\beta}\vert\mathbf{y})&=\sum_{i=1}^n\left\\{y_i\log{p_1(\mathbf{x}_i;\boldsymbol{\beta})}+(1-y_i)\log{(1-p_1(\mathbf{x}_i;\boldsymbol{\beta}))}\right\\} \\\\
&=\sum_{i=1}^n\left\\{y_i\log{\dfrac{\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}}{1+\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}}} +(1-y_i)\log{\dfrac{1}{1+\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}}}\right\\} \\\\
&=\sum_{i=1}^n\left\\{y_i\boldsymbol{\beta}^T\mathbf{x}_i-\log{(1+\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)})}\right\\}
\end{aligned}$

To maximize the log-likelihood, we set its derivatives to zero.

$\begin{aligned}
\dfrac{\partial}{\partial\boldsymbol{\beta}}l(\boldsymbol{\beta}\vert\mathbf{y})&=\sum_{i=1}^n\left\\{y_i\mathbf{x}_i-\dfrac{\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}}{1+\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}}\mathbf{x}_i\right\\} \\\\
&=\sum_{i=1}^n\mathbf{x}_i(y_i-p_1(\mathbf{x}_i;\boldsymbol{\beta}))=0
\end{aligned}$

Now we will solve this equation by using Newton-Raphson algorithm.

---

*Brief Summary of Newton-Raphson Algorithm*

{{<figure src="/fig11.jpeg" width="400">}}

$L_1=y-f(x_1)=f'(x_1)(x-x_1)$

$x_2=x_1-\dfrac{f(x_1)}{f'(x_1)} \quad\cdots\quad \underset{n\rightarrow\infty}{\lim}x_n=r$

---

We need the second dervatives of log-likelihood.

$\begin{aligned}
\dfrac{\partial^2}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta}^T}l(\boldsymbol{\beta}\vert\mathbf{y})&=-\sum_{i=1}^n\mathbf{x}_i\dfrac{\mathbf{x}_i^T\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}\left\\{1+\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}\right\\}-\mathbf{x}_i^T\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}}{\left\\{1+\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}\right\\}^2} \\\\
&=-\sum_{i=1}^n\mathbf{x}_i\mathbf{x}_i^T\dfrac{\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}}{\left\\{1+\exp{(\boldsymbol{\beta}^T\mathbf{x}_i)}\right\\}^2} \\\\
&=-\sum_{i=1}^n\mathbf{x}_i\mathbf{x}_i^Tp_1(\mathbf{x}_i;\boldsymbol{\beta})(1-p_1(\mathbf{x}_i;\boldsymbol{\beta}))
\end{aligned}$

Starting with some inital point $\boldsymbol{\beta}^\text{old}$, we will continuously update $\boldsymbol{\beta}$ until it converges.

$\boldsymbol{\beta}^\text{new}=\boldsymbol{\beta}^\text{old}-\left(\dfrac{\partial^2}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta}^T}l(\boldsymbol{\beta}\vert\mathbf{y})\right)^{-1}\dfrac{\partial}{\partial\boldsymbol{\beta}}l(\boldsymbol{\beta}\vert\mathbf{y})$

To make the calcuation easier, we will use some matrix notations.

Let $\dfrac{\partial}{\partial\boldsymbol{\beta}}l(\boldsymbol{\beta}\vert\mathbf{y})=\mathbf{X}^T(\mathbf{y}-\mathbf{p})$ and $\dfrac{\partial^2}{\partial\boldsymbol{\beta}\partial\boldsymbol{\beta}^T}l(\boldsymbol{\beta}\vert\mathbf{y})=-\mathbf{X}^T\mathbf{W}\mathbf{X}$.

Then we can write down the updating algorithm as

$\begin{aligned}
\boldsymbol{\beta}^\text{new}&=\boldsymbol{\beta}^\text{old}+(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y}-\mathbf{p}) \\\\
&=(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}(\mathbf{X}\boldsymbol{\beta}^\text{old}+\mathbf{W}^{-1}(\mathbf{y}-\mathbf{p})) \\\\
&=(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{z}
\end{aligned}$

Usually $\boldsymbol{\beta}=\mathbf{0}$ is used as an inital point for the iterative procedure, though the convergence is never guaranteed. Typically the algorithm does converge, since the log-likelihood is concave, but overshooting can occur.

---

**Reference**

1. Elements of Statistical Learning