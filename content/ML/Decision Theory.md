---
title: "Decision Theory"
weight: 1
comments: true
---

Let $\mathbf{X}$ be a random vector and $Y$ be a random variable. We want to make a prediction of $Y$ at any point $\mathbf{X}=\mathbf{x}$. Let's assume that we know $f_{\mathbf{X}, Y}(\mathbf{x}, y)$ the joint distribution of $\mathbf{X}$ and $Y$. How can we make the 'best' prediction of $Y$?

**Regression**

We will first consider the case that $Y$ is a quantitative output which is called as a regression problem. We can think about the bivariate distribution of $f_{X, Y}(x, y)$.

{{<figure src="/fig1.jpeg" width="600" height="400">}}

$Y$ which satisfies the largest conditional probability $P(Y\vert X=x)$ or the median value of $Y$ might be the answer. We need some criteria to decide which one is the best.

For this case, a measurement for 'best' is usually a squared error loss, $L(Y, f(X))=(Y-f(X))^2$. Our goal is to find $f$ which minimizes the expected loss $\text{E}[L(Y, f(X))]$.

$\begin{aligned}
\text{E}[(Y-f(X))^2]&=\int\int(y-f(x))^2f_{X, Y}(x, y)dxdy \\\\
&=\int\left\\{\int(y-f(x))^2f_{Y\vert X}(y\vert x)dy\right\\}f_X(x)dx
\end{aligned}$

It is obvious that minimizing $\text{E}[(Y-f(X))^2]$ is same as to minimize the inside integral. Thus,

$\begin{aligned}
\dfrac{\partial}{\partial f}\int(y-f(x))^2f_{Y\vert X}(y\vert x)dy&=\int\dfrac{\partial}{\partial f}\left\\{(y-f(x))^2f_{Y\vert X}(y\vert x)\right\\}dy \\\\
&=2\int(y-f(x))f_{Y\vert X}(y\vert x)dy=0
\end{aligned}$

$\displaystyle f(x)\int f_{Y\vert X}(y\vert x)dy=\int yf_{Y\vert X}(y\vert x)dy$

$\therefore f(x)=\text{E}[Y\vert X=x]$

Now we know that the best prediction for $Y$ is the conditional expectation of $Y$ given $X$.

---

**Classification**

Next case is about a quantitative output $Y$, related to a classification problem. Again, we will consider the bivariate distribution $f_{X, Y}(x, y)$, where $\displaystyle\int_x\sum_yf_{X, Y}(x, y)dx=1$. If $Y$ is a binary variable, we can represent a joint pdf of $X$ and $Y$ as below.

{{<figure src="/fig2.jpeg" width="600" height="400">}}

The most intuitive way for choosing an appropriate value for $f(X)$ is to use $c$ which satisfies $\underset{k}{\text{argmax}}P(X, Y=k)$. By using a bayes' theorem, it is evident that $\underset{k}{\text{argmax}}P(X, Y=k)$ is same as $\underset{k}{\text{argmax}}P(Y=k\vert X)$. This is called as a bayes classifier.

This intutition can be supported by using $0$-$1$ loss, $L(Y, f(X))=I_{Y\neq f(X)}$.

$\begin{aligned}
\text{E}[I_{Y\neq f(X)}]&=\int_x\sum_yI_{Y\neq f(X)}f_{X, Y}(x, y)dx \\\\
&=\int_x\sum_yI_{Y\neq f(X)}f_{Y\vert X}(y\vert x)f(x)dx
\end{aligned}$

As for the regression case, we will minize the inside summation.

$\underset{f}{\text{argmin}}\\,I_{Y_1\neq f(X)}f_{Y_1\vert X}(y_1\vert x)+\cdots+I_{Y_K\neq f(X)}f_{Y_K\vert X}(y_K\vert x) \\\\
=\underset{f}{\text{argmin}}\\,f_{Y_1\vert X}(y_1\vert x)+\cdots+f_{Y_{k-1}\vert X}(y_{k-1}\vert x)+f_{Y_{k+1}\vert X}(y_{k+1}\vert x)+\cdots+f_{Y_K\vert X}(y_K\vert x) \\\\
=\underset{f}{\text{argmin}}\\,\left\\{1-f_{Y_k\vert X}(y_k\vert x)\right\\} \\\\
=\underset{k}{\text{argmax}}\\,f_{Y_k\vert X}(y_k\vert x)$

---

**$k$-NN Method**

If we know the true distribution of $f_{X, Y}(x, y)$, we can easily predict qualitative $Y$ as $\text{E}[Y|X=x]$ and quantitative $Y$ as $\underset{k}{\text{argmax}}\\,f_{Y_k\vert X}(y_k\vert x)$. However, actually we don't know the exact form of this joint distribution. One might think that $k$-NN regressor and classifier could be the best approximations for these.

---

**Reference**

1. Elements of Statistical Learning