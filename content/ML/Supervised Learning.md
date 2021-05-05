---
title: "Supervised Learning"
draft: true
---

**Regression**

Let $Y$ as a random variable in $\mathbb{R}$ and $X$ as a random vector in $\mathbb{R}^p$. We want to know the relationship between $Y$ and $X$. Actually, we seek a best function $f(X)$ for predicting $Y$. We will first consider the case that $Y$ is a quantitative output. For this case, we use a squared error loss $L(Y, f(X))=(Y-f(X))^2$. Our goal is to find $f$ which minimizes the expected loss $\text{E}[L(Y, f(X))]$.

$\begin{aligned}
\text{E}[(Y-f(X))^2]&=\int\int(y-f(x))^2f_{X, Y}(x, y)dxdy \\\\
&=\int\left\\{\int(y-f(x))^2f_{Y|X}(y|x)dy\right\\}f_X(x)dx
\end{aligned}$

It is obvious that minimizing $\text{E}[(Y-f(X))^2]$ is same as to minimize the inside integral. Thus,

$\begin{aligned}
\dfrac{\partial}{\partial f}\int(y-f(x))^2f_{Y|X}(y|x)dy&=\int\dfrac{\partial}{\partial f}\left\\{(y-f(x))^2f_{Y|X}(y|x)\right\\}dy \\\\
&=-2\int(y-f(x))f_{Y|X}(y|x)dy=0
\end{aligned}$

$\displaystyle f(x)\int f_{Y|X}(y|x)dy=\int yf_{Y|X}(y|x)dy$

$\therefore f(x)=\text{E}[Y|X=x]$

Now we know that the best prediction for $Y$ is the conditional expectation of $Y$ given $X$. However, we don't know the true distribution of $Y$ and $X$, so as the exact value of $\text{E}[Y|X]$. We have to approximate this. Though, it is reaosnable to assume our true model as $Y=f(X)+\epsilon$ for this regression problem, where $\epsilon$ is a random variable with mean $0$ and variance $\sigma^2$. This error term can explain the uncontrollable influence such as the effect of variables we missed.

---

Let's talk about the way how we estimate $f(X)$ now. We need some criteria to choose the best approximation. Two things can be introduced, accuracy and interpretability. If we want our model to be accurate about some predictions or classifications for unseen data, test accuracy should be our criterion. On the other hand, if our goal is to explain the relationship between $Y$ and $X$, model interpretability might be the suitable criterion. In this post, we will concentrate more on the model accuracy, because interpretability is a quite subjective concept.

$\hat{f}(X)$ can be affected by several factors: which model to use, variable to use, and which hyper-parameter to use. This factors can be determined by comparing each cases with test error. We can express this error calculated by using $\hat{f}(X)$ which is fitted by using training set $\mathcal{T}$ as $\text{E}[L(Y, \hat{f}(X))\vert\mathcal{T}]$. However, here the training set is fixed and we want our $\hat{f}(X)$ to be the best for every possible training sets. Thus, we should minimize the expected test error $\text{E}[\text{E}[(Y, \hat{f}(X))|\mathcal{T}]]=\text{E}[L(Y, \hat{f}(X))]$. We cannot know the exact value of it, but we can estimate it.

---

**Classification**

For a quantitative output $Y$, we usually use a $0$-$1$ loss $L(G, G(X))=I_{G\neq G(X)}$.

---

**Bias-Variance Trade-off**

$\begin{aligned}
\text{E}[(y_0-\hat{f}(\mathbf{x}_0))^2]&=\text{E}[(f(\mathbf{x}_0)+\epsilon_0-\hat{f}(\mathbf{x}_0))^2] \\\\
&=\text{E}[(f(\mathbf{x}_0)-\hat{f}(\mathbf{x}_0))^2]+2\text{E}[(f(\mathbf{x}_0)-\hat{f}(\mathbf{x}_0))\epsilon_0]+\text{E}[\epsilon_0^2] \\\\
&=\text{E}[(f(\mathbf{x}_0)-\hat{f}(\mathbf{x}_0))^2]+\text{Var}(\epsilon_0) \\\\
&=\text{E}[(f(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)]-(\hat{f}(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)]))^2]+\text{Var}(\epsilon_0) \\\\
&=\text{E}[(f(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)])^2]+\text{E}[(\hat{f}(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)])^2]+\text{Var}(\epsilon_0) \\\\
&=[\text{Bias}(\hat{f}(\mathbf{x}_0))]^2+\text{Var}(\hat{f}(\mathbf{x}_0))+\text{Var}(\epsilon_0)
\end{aligned}$
