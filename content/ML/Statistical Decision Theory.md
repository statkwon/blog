---
title: "Statistical Decision Theory"
date: 2021-03-30
draft: true
TableOfContents: true
---

In our data, there exist both pattern and noise. Thus, to find this pattern from our data, we assume a statistical model as below.

$$Y=f(X)+\epsilon, \quad \epsilon_i\overset{\text{iid}}{\sim}(0, \sigma^2)$$

We want to know the probability of $Y$ when $X$ is given. It does not mean to tell the exact value of $Y$ for corresponding $X$. We should not as there exists the error term. However, knowing the true distribution of $Y$ given $X$ is impossible unless we have the total data. So rather we will estimate some useful informations about $Y|X$ and $f(X)$ is the mean of $Y$.

What if we have to tell only one value for $Y$ given $X$?

Seek a function which minimizes the expected loss.

$\begin{aligned}
\text{E}[L(Y, f(X))]&=\int\int L(y, f(x))f_{X, Y}(x, y)dxdy \\\\
&=\int\int L(y, f(x))f_{Y|X}(y|x)f_X(x)dxdy \\\\
&=\int\left\\{\int L(y, f(x))f_{Y|X}(y|x)dy\right\\}f_{X}(x)dx
\end{aligned}$

Minimizing the expected loss is same as minimizing $\int L(y, f(x))f_{Y|X}(y|x)dy$.

## 1. Regression Case

Squared Error Loss: $L(Y, f(X))=(Y-f(X))^2$

$\displaystyle\underset{f}{\text{argmin}}\int(y-f(x))^2f_{Y|X}(y|x)dy$

$\begin{aligned}
\dfrac{\partial}{\partial f}\int(y-f(x))^2f_{Y|X}(y|x)dy&=\int\dfrac{\partial}{\partial f}\left\\{(y-f(x))^2f_{Y|X}(y|x)\right\\}dy \\\\
&=-2\int(y-f(x))f_{Y|X}(y|x)dy=0
\end{aligned}$

$\begin{aligned}
\therefore f(x)&=\int yf_{Y|X}(y|x)dy \\\\
&=\text{E}[Y|X=x]
\end{aligned}$

## 2. Classification Case

$0$-$1$ Loss: $I_{y_k\neq f_k(x)}$

$\displaystyle\underset{f}{\text{argmin}}\int(y-f(x))^2f_{Y|X}(y|x)dy$