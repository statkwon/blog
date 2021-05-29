---
title: "Natural Spline"
---

Adding another constraint to regression spline, we can fit natural splines. We will reduce the degree beyond the boundary to $\dfrac{M-1}{2}$.

The number of parameters to fit natural splines is $(M+1)\times(K-1)-\left(\dfrac{M-1}{2}+1\right)\times2-M\times K=K$ and we can see that it is independent of $M$.

Natural Cubic Spline is the most common one.

**Natural Cubic Spline**

$N_1(X)=1, \quad N_2(X)=X, \quad N_{k+2}(X)=d_k(X)-d_{K-1}(X)$

$d_k(X)=\dfrac{(X-\xi_k)\_+^3-(X-\xi_K)_+^3}{\xi_k-\xi_K}$

---

We will make a proof for the formula above.

$\displaystyle f(X)=\sum_{j=1}^4\beta_jX^{j-1}+\sum_{k=1}^K\theta_k(X-\xi_k)_+^3$

1\) $(-\infty, \xi_1) \quad\Rightarrow\quad f(X)=\beta_1+\beta_2X+\beta_3X^2+\beta_4X^3$

Because of the degree constraint beyond the boundary, $\beta_3$ and $\beta_4$ should be $0$.

2\) $(\xi_K, \infty) \quad\Rightarrow\quad \displaystyle f(X)=\beta_1+\beta_2X+\sum_{k=1}^K\theta_k(X^3-3X^2\xi_k-3X\xi_k^2-\xi_k^3)$

Because of the constraint beyond the boundary, $\displaystyle\sum_{k=1}^K\theta_k=0$ and $\displaystyle\sum_{k=1}^K\theta_k\xi_k=0$.

3\) $\displaystyle\theta_K=-\sum_{k=1}^{K-1}\theta_k$

4\) $\displaystyle\theta_{K-1}=\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}$

$\begin{aligned}
\sum_{k=1}^K\theta_k\xi_k&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}+\theta_K\xi_K \\\\
&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}-\sum_{k=1}^{K-1}\theta_k\xi_K \\\\
&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}-\sum_{k=1}^{K-2}\theta_k\xi_K-\theta_{K-1}\xi_K=0
\end{aligned}$

$\displaystyle\Leftrightarrow\theta_{K-1}(\xi_K-\xi_{K-1})=\sum_{k=1}^{K-2}\theta_k(\xi_k-\xi_K)$

$\therefore\displaystyle\theta_{K-1}=\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}$

5\) $\displaystyle f(X)=\beta_1+\beta_2X+\sum_{k=1}^{K-2}\phi_k\left\\{d_k(X)-d_{K-1}(X)\right\\}$

Let $\displaystyle f(X)=\beta1+\beta_2X+\sum_{k=1}^K\theta_k(X-\xi_k)_+^3=\beta_1+\beta_2X+g(X)$.

$\begin{aligned}
g(X)&=\sum_{k=1}^{K-1}\theta_k(X-\xi_k)_+^3+\theta_K(X-\xi_K)_+^3 \\\\
&=\sum_{k=1}^{K-1}\theta_k(X-\xi_k)_+^3-\sum_{k=1}^{K-1}\theta_k(X-\xi_K)_+^3 \\\\
&=\sum_{k=1}^{K-1}\theta_k\left\\{(X-\xi_k)_+^3-(X-\xi_K)_+^3\right\\} \\\\
&=\sum_{k=1}^{K-2}\theta_k\left\\{(X-\xi_k)_+^3-(X-\xi_K)_+^3\right\\}+\theta_{K-1}\left\\{(X-\xi_{K-1})_+^3-(X-\xi_K)_+^3\right\\} \\\\
&=\sum_{k=1}^{K-2}\theta_k\left\\{(X-\xi_k)_+^3-(X-\xi_K)_+^3\right\\}+\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}\left\\{(X-\xi_{K-1})_+^3-(X-\xi_K)_+^3\right\\} \\\\
&=\sum_{k=1}^{K-2}\theta_k(\xi_k-\xi_K)\left\\{\dfrac{(X-\xi_k)_+^3-(X-\xi_K)_+^3}{\xi_K-\xi_k}-\dfrac{(X-\xi_{K-1})_+^3-(X-\xi_K)_+^3}{\xi_K-\xi_{K-1}}\right\\} \\\\
&=\sum_{k=1}^{K-2}\phi_k\left\\{d_k(X)-d_{K-1}(X)\right\\}
\end{aligned}$

$\begin{aligned}
\therefore f(X)&=\beta_1+\beta_2X+g(X) \\\\
&=\beta_1+\beta_2X+\sum_{k=1}^{K-2}\phi_k\left\\{d_k(X)-d_{K-1}(X)\right\\}
\end{aligned}$

---

**Reference**

1. Elements of Statistical Learning