---
title: "Smoothing Spline"
---

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

---

**Reference**

1. Elements of Statistical Learning