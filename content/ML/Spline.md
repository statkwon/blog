---
title: "Splines"
date: 2021-04-10
lastmod: 2021-04-10
draft: false
TableOfContents: true
---

To move beyond linearity, we will transform $X$ and use linear models in this new space of derived input features.

$$f(X)=\sum_{m=1}^M\beta_mh_m(X)$$

## 1. Piecewise Polynomials

Divide the domain of $X$ into contiguous intervals and fit polynomial regression models in each interval.

**Piecewise Constant**

![fig 3](/ml/fig/fig3.png)

**Piecewise Linear**

**Piecewise Cubic**

The number of paramters to fit piecewise polynomial is $(M+1)\times(K+1)$, where $M$ is the order and $K$ is the number of knots.

Piecewise polynomials are erratic at each knots.

## 2. Regression Splines

We can obtain regression splines by adding a continuity constraint to piecewise polynomials.

Regression splines are often called the $M$th order spline, a piecewise polynomial of degree $M$, that is continuous and has continuous derivatives of orders $1, \ldots, M-1$ at its know points.

**$M$th order spline with $K$ knots**

$h_j(X)=X^{j-1} \quad (j=1, 2, \ldots, M+1) \\\\
h_{M+1+l}(X)=(X-\xi_l)_+^M \quad (l=1, 2, \ldots, K) \\\\
f(X)=\beta_1+\beta_2X\cdots+\beta_{M+1}X^M+\beta_{M+2}(X-\xi_1)_+^M+\cdots+\beta_{M+K+1}(X-\xi_K)_+^M$

{{<rawhtml>}}
<details>
<summary>Proof</summary>
We will make a proof for the case when $M=3$ and $K=2$.
<br>
$f(X)=\beta_1+\beta_2X+\beta_3X^2+\beta_4X^3+\beta_5(X-\xi_1)_+^3+\beta_6(X-\xi_2)_+^3$
<br>
We will show the continuity of $f(X)$, $f'(X)$, $f''(X)$ at the knots $\xi_1$, $\xi_2$ ($\xi_1<\xi_2$).
<br><br>
1) Continuity of $f(X)$
<br>
$\begin{aligned}
f(\xi_1-h)&=\beta_1+\beta_2(\xi_1-h)+\beta_3(\xi_1-h)^2+\beta_4(\xi_1-h)^3+\beta_5(\xi_1-h-\xi_1)_+^3+\beta_6(\xi_1-h-\xi_2)_+^3 \\
&=\beta_1+\beta_2(\xi_1-h)+\beta_3(\xi_1-h)^2+\beta_4(\xi_1-h)^3
\end{aligned}$
<br>
$f(\xi_1+h)=\beta_1+\beta_2(\xi_1+h)+\beta_3(\xi_1+h)^2+\beta_4(\xi_1+h)^3+\beta_5(\xi_1+h-\xi_1)_+^3+\beta_6(\xi_1+h-\xi_2)_+^3$
<br>
$\displaystyle\lim_{h\rightarrow0}f(\xi_1-h)=\lim_{h\rightarrow0}f(\xi_1+h)=\beta_1+\beta_2\xi_1+\beta_3\xi_1^2+\beta_4\xi_1^3$
<br>
$\displaystyle\therefore\lim_{x\rightarrow\xi_1}f(x)=\beta_1+\beta_2\xi_1+\beta_3\xi_1^2+\beta_4\xi_1^3=f(\xi_1)$
<br>
$f(X)$ is continuous at $\xi_1$ and we can show that $f(X)$ is continuous at $\xi_2$ by a similar way.
<br><br>
2) Continuity of $f'(X)$ at $\xi_1$, $\xi_2$
<br>
$\begin{aligned}
f'(\xi_1^-)&=\lim_{h\rightarrow0}\dfrac{f(\xi_1)-f(\xi_1-h)}{h} \\
&=\lim_{h\rightarrow0}\dfrac{\beta_2h+2\beta_3\xi_1h+3\beta_4\xi_1^2h+O(h^2)}{h} \\
&=\beta_2+2\beta_3\xi_1+3\beta_4\xi_1^2
\end{aligned}$
<br>
$\begin{aligned}
f'(\xi_1^+)&=\lim_{h\rightarrow0}\dfrac{f(\xi_1+h)-f(\xi_1)}{h} \\
&=\lim_{h\rightarrow0}\dfrac{\beta_2h+2\beta_3\xi_1h+3\beta_4\xi_1^2h+\beta_5(\xi_1+h-\xi_1)_+^3+\beta_6(\xi_1+h-\xi_2)_+^3+O(h^2)}{h} \\
&=\beta_2+2\beta_3\xi_1+3\beta_4\xi_1^2
\end{aligned}$
<br>
$\therefore\lim_{x\rightarrow\xi_1}f'(x)=\beta_2+2\beta_3\xi_1+3\beta_4\xi_1^2=f'(\xi_1)$
<br>
$f'(X)$ is continuous at $\xi_1$ and we can show that $f'(X)$ is continuous at $\xi_2$ by a similar way.
<br><br>
3) Continuity of $f''(X)$ at $\xi_1$, $\xi_2$
<br>
Similarly, $\lim_{x\rightarrow\xi_1}f''(x)=6\beta_4\xi_1^2=f''(\xi_1)$ and $\lim_{x\rightarrow\xi_2}f''(x)=6\beta_4\xi_2^2=f''(\xi_2)$.
<br><br>
We showed that $f(X)$, $f'(X)$ and $f''(X)$ are all continuous at $\xi_1$, $\xi_2$. Thus, $f(X)$ represents a cubic spline with two knots.
</details>
{{</rawhtml>}}

The number of parameters to fit regression spline is $(M+1)\times(K+1)-M\times K=M+K+1$.

However, adding a continuity constraint still cannot fix the irregularity beyond the boundaries.

## 3. Natural Splines

Adding another constraint to regression spline, we can fit natural splines. We will reduce the degree beyond the boundary to $\dfrac{M-1}{2}$.

The number of parameters to fit natural splines is $(M+1)\times(K-1)-\left(\dfrac{M-1}{2}+1\right)\times2-M\times K=K$ and we can see that it is independent of $M$.

Natural Cubic Spline is the most common one.

**Natural Cubic Spline**

$N_1(X)=1, \quad N_2(X)=X, \quad N_{k+2}(X)=d_k(X)-d_{K-1}(X) \\\\
d_k(X)=\dfrac{(X-\xi_k)_+^3-(X-\xi_K)_+^3}{\xi_k-\xi_K}$

{{<rawhtml>}}
<details>
<summary>Proof</summary>
$\displaystyle f(X)=\sum_{j=1}^4\beta_jX^{j-1}+\sum_{k=1}^K\theta_k(X-\xi_k)_+^3$
<br><br>
1) $(-\infty, \xi_1) \quad\Rightarrow\quad f(X)=\beta_1+\beta_2X+\beta_3X^2+\beta_4X^3$
<br>
Because of the degree constraint beyond the boundary, $\beta_3$ and $\beta_4$ should be $0$.
<br><br>
2) $(\xi_K, \infty) \quad\Rightarrow\quad \displaystyle f(X)=\beta_1+\beta_2X+\sum_{k=1}^K\theta_k(X^3-3X^2\xi_k-3X\xi_k^2-\xi_k^3)$
<br>
Because of the constraint beyond the boundary, $\displaystyle\sum_{k=1}^K\theta_k=0$ and $\displaystyle\sum_{k=1}^K\theta_k\xi_k=0$.
<br><br>
3) $\displaystyle\theta_K=-\sum_{k=1}^{K-1}\theta_k$
<br><br>
4) $\displaystyle\theta_{K-1}=\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}$
<br>
$\begin{aligned}
\sum_{k=1}^K\theta_k\xi_k&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}+\theta_K\xi_K \\
&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}-\sum_{k=1}^{K-1}\theta_k\xi_K \\
&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}-\sum_{k=1}^{K-2}\theta_k\xi_K-\theta_{K-1}\xi_K=0
\end{aligned}$
<br>
$\displaystyle\Leftrightarrow\theta_{K-1}(\xi_K-\xi_{K-1})=\sum_{k=1}^{K-2}\theta_k(\xi_k-\xi_K)$
<br>
$\therefore\displaystyle\theta_{K-1}=\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}$
<br><br>
5) $\displaystyle f(X)=\beta_1+\beta_2X+\sum_{k=1}^{K-2}\phi_k\left\{d_k(X)-d_{K-1}(X)\right\}$
<br>
Let $\displaystyle f(X)=\beta1+\beta_2X+\sum_{k=1}^K\theta_k(X-\xi_k)_+^3=\beta_1+\beta_2X+g(X)$.
<br>
$\begin{aligned}
g(X)&=\sum_{k=1}^{K-1}\theta_k(X-\xi_k)_+^3+\theta_K(X-\xi_K)_+^3 \\
&=\sum_{k=1}^{K-1}\theta_k(X-\xi_k)_+^3-\sum_{k=1}^{K-1}\theta_k(X-\xi_K)_+^3 \\
&=\sum_{k=1}^{K-1}\theta_k\left\{(X-\xi_k)_+^3-(X-\xi_K)_+^3\right\} \\
&=\sum_{k=1}^{K-2}\theta_k\left\{(X-\xi_k)_+^3-(X-\xi_K)_+^3\right\}+\theta_{K-1}\left\{(X-\xi_{K-1})_+^3-(X-\xi_K)_+^3\right\} \\
&=\sum_{k=1}^{K-2}\theta_k\left\{(X-\xi_k)_+^3-(X-\xi_K)_+^3\right\}+\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}\left\{(X-\xi_{K-1})_+^3-(X-\xi_K)_+^3\right\} \\
&=\sum_{k=1}^{K-2}\theta_k(\xi_k-\xi_K)\left\{\dfrac{(X-\xi_k)_+^3-(X-\xi_K)_+^3}{\xi_K-\xi_k}-\dfrac{(X-\xi_{K-1})_+^3-(X-\xi_K)_+^3}{\xi_K-\xi_{K-1}}\right\} \\
&=\sum_{k=1}^{K-2}\phi_k\left\{d_k(X)-d_{K-1}(X)\right\}
\end{aligned}$
<br>
$\begin{aligned}
f(X)&=\beta_1+\beta_2X+g(X) \\
&=\beta_1+\beta_2X+\sum_{k=1}^{K-2}\phi_k\left\{d_k(X)-d_{K-1}(X)\right\}
\end{aligned}$
</details>
{{</rawhtml>}}

## 4. Smoothing Splines

Without any constraint on the form of $f(X)$, we can make the $\text{RSS}$ to be $0$ by choosing any function that interpolates all data points. However, this will be connected to an overfitting problem. To prevent this, we will use the regularization term.

$$\underset{f}{\text{argmin}}\sum_{i=1}^n\left\\{y_i-f(x_i)\right\\}^2+\lambda\int\left\\{f^{\left\(\frac{M+1}{2}\right\)}(x)\right\\}^2dx$$

Here $\lambda$ is called a smoothing parameter. If $\lambda=0$, $f$ can be any function that interpolates the data. If $\lambda=\infty$, $f$ will be a simple line. Now we will show that the unique minimizer of this criterion is a natural cubic spline with knots at each of the $x_i$. We will consider the case with $M=3$. Other cases can be proved similarly.

Suppose that $n≥2$ and $g$ is the natural cubic spline interpolant to the pairs $(x_i, y_i)$, with $a<x_1<\cdots<x_n<b$. Let $\tilde{g}$ be any other differentiable function on $[a, b]$ that interpolates the $n$ pairs and be the unique solution of $\displaystyle\underset{f}{\text{argmin}}\sum_{i=1}^n\left\\{y_i-f(x_i)\right\\}^2+\lambda\int\left\\{f''(x)\right\\}^2dx$. Also let $h(x)=\tilde{g}(x)-g(x)$.

1\) $\displaystyle\int_a^bg''(x)h''(x)dx=0$

$\begin{aligned}
\int_a^bg''(x)h''(x)dx&=g''(x)h'(x)\Big\vert_a^b-\int_a^bg'''(x)h'(x)dx \\\\
&=-\int_a^bg'''(x)h'(x)dx \quad (\because g''(a)=g''(b)=0) \\\\
&=-g'''(x)h(x)\Big\vert_a^b+\int_a^bg''''(x)h'(x)dx \\\\
&=-g'''(x)h(x)\Big\vert_{x_1}^{x_n} \quad (\because g''''(x)=0) \\\\
&=0 \quad (\because h(x_1)=h(x_n)=0)
\end{aligned}$

$\displaystyle\therefore\int_a^bg''(x)h''(x)dx=0$

2\) $\displaystyle\int_a^b\left\\{\tilde{g}''(x)\right\\}^2dx≥\int_a^b\left\\{g''(x)\right\\}^2dx$

$\begin{aligned}
\int_a^b\left\\{\tilde{g}(x)\right\\}^2dx&=\int_a^b\left\\{g''(x)+h''(x)\right\\}^2dx \\\\
&=\int_a^b\left\\{g''(x)\right\\}^2dx+\int_a^b\left\\{h''(x)\right\\}^2dx+2\int_a^bg''(x)h''(x)dx \\\\
&=\int_a^b\left\\{g''(x)\right\\}^2dx+\int_a^b\left\\{h''(x)\right\\}^2dx \quad (\because \int_a^bg''(x)h''(x)dx=0) \\\\
&≥\int_a^b\left\\{g''(x)\right\\}^2dx
\end{aligned}$

$\displaystyle\therefore\int_a^b\left\\{\tilde{g}''(x)\right\\}^2dx≥\int_a^b\left\\{g''(x)\right\\}^2dx$

3\) Natural Cubic Spline with $n$ knots is the unique minizer.

$\displaystyle\int_a^b\left\\{\tilde{g}''(x)\right\\}^2dx≥\int_a^b\left\\{g''(x)\right\\}^2dx \\\\
\displaystyle\Leftrightarrow\lambda\int_a^b\left\\{\tilde{g}''(x)\right\\}^2dx≥\lambda\int_a^b\left\\{g''(x)\right\\}^2dx$

We knot that $g(x)$ is the natural cubic spline. We assumed that $\tilde{g}$ is the unique minizer, so $\tilde{g}$ should be the natural cubic spline.

Since the solution is a natural spline, we can write it as $\displaystyle f(x)=\sum_{j=1}^n\theta_jN_j(x)$. The problem about $f(x)$ has been reduced to the problem of estimating $\theta_j$.

$$\underset{\theta}{\text{argmin}}\\;(y-N\theta)^T(y-N\theta)+\lambda\theta^T\Omega_N\theta$$

$\\{N\\}_{ij}$ is $N_j(x_i)$ and $\\{\Omega_N\\}\_{jk}$ is $\int N''_j(t)N''_k(t)dt$. The solution will be $\hat{\theta}=(N^TN+\lambda\Omega_N)^{-1}N^Ty$.