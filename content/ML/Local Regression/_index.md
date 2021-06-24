---
title: "Local Regression"
date: 2021-03-20
weight: 13
---

Nadaraya-Watson kernel regression usually suffers from a boundary issue and local regression can be the alternative.

**Local Linear Regression**

$\hat{f}(x_0)=\hat{\alpha}(x_0)+\hat{\beta}(x_0)x_0$

We can get this local linear estimate by solving the problem below.

$\displaystyle\underset{\alpha(x_0), \beta(x_0)}{\text{argmin}}\sum_{i=1}^nK_\lambda(x_0, x_i)\left\\{y_i-\alpha(x_0)-\beta(x_0)x_i\right\\}^2$

This model uses the weighted observations in the neighborhood of $x_0$ and fit a regression line with them. This can be regraded as solving a weighted least square problem. Thus we can write $\hat{f}(x_0)$ as

$\begin{aligned}
\hat{f}(x_0)&=\mathbf{b}(x_0)^T(B^TW(x_0)B)^{-1}B^TW(x_0)y \\\\
&=\mathbf{l}(x_0)^Ty \\\\
&=\sum_{i=1}^nl_i(x_0)y_i
\end{aligned}$

where $\mathbf{b}(x)=\begin{bmatrix} 1 \\\\ x \end{bmatrix}$, $B=\begin{bmatrix} 1 & x_1 \\\\ 1 & x_2 \\\\ \vdots & \vdots \\\\ 1 & x_n \end{bmatrix}$ and $W=\begin{bmatrix} K_\lambda(x_0, x_1) & 0 & \cdots & 0 \\\\ 0 & K_\lambda(x_0, x_2) & \cdots & 0 \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & K_\lambda(x_0, x_n) \end{bmatrix}$.

Sometimes we call the $l_i(x_0)$ as equivalent kernel.

Local linear regression has its advantage on the boundary due to the fact that it can reduce the first-order bias automatically. We will prove this.

---

1\) $\displaystyle\sum_{i=1}^nl_i(x_0)=1$, $\displaystyle\sum_{i=1}^nl_i(x_0)x_i=x_0$

We knot that $\displaystyle\mathbf{b}(x_0)^T(B^TW(x_0)B)^{-1}B^TW(x_0)y=\sum_{i=1}^nl_i(x_0)y_i$, and let $\mathbf{v}_j=\begin{bmatrix} x_1^j & x_2^j & \cdots & x_n^j \end{bmatrix}^T$.

Then we can show that $\displaystyle\mathbf{b}(x_0)^T(B^TW(x_0)B)^{-1}B^TW(x_0)\mathbf{v}_j=\sum\_{i=1}^nl_i(x_0)x_i^j$.

$\begin{aligned}
\mathbf{b}(x_0)^T(B^TW(x_0)B)^{-1}B^TW(x_0)\begin{bmatrix} \mathbf{v}_0 & \mathbf{v}_1 \end{bmatrix}&=\mathbf{b}(x_0)^T(B^TW(x_0)B)^{-1}B^TW(x_0)B \\\\
&=\mathbf{b}(x_0)^T \\\\
&=\begin{bmatrix} 1 & x_0 \end{bmatrix}
\end{aligned}$

$\displaystyle\therefore \sum_{i=1}^nl_i(x_0)=1, \\; \sum_{i=1}^nl_i(x_0)x_i=x_0$

2\) $\text{Bias}(\hat{f}(x_0))=\dfrac{f''(x_0)}{2}\sum_{i=1}^n(x_i-x_0)^2l_i(x_0)+R$

By using taylor expansion, we can write down the expectation of $\hat{f}(x_0)$ as below.

$\begin{aligned}
\text{E}[\hat{f}(x_0)]&=\sum_{i=1}^nl_i(x_0)f(x_i) \\\\
&=f(x_0)\sum_{i=1}^nl_i(x_0)+f'(x_0)\sum_{i=1}^n(x_i-x_0)l_i(x_0)+\dfrac{f''(x_0)}{2}\sum_{i=1}^n(x_i-x_0)^2l_i(x_0)+R
\end{aligned}$

We already showed that $\displaystyle\sum_{i=1}^nl_i(x_0)=1$ and $\displaystyle\sum_{i=1}^nl_i(x_0)x_i=x_0$.

$\displaystyle\sum_{i=1}^n(x_i-x_0)l_i(x_0)=\sum_{i=1}^nx_il_i(x_0)-x_0\sum_{i=1}^nl_i(x_0)=0$

$\displaystyle\therefore\text{E}[\hat{f}(x_0)]=f(x_0)+\dfrac{f''(x_0)}{2}\sum_{i=1}^n(x_i-x_0)^2l_i(x_0)+R$

This implies that the bias of $\hat{f}(x_0)$ only depends on the second derivative and the higher-order terms.

---

**Local Polynomial Regression**

$\displaystyle\hat{f}(x_0)=\hat{\alpha}(x_0)+\sum_{j=1}^d\hat{\beta}_j(x_0)x_0^j$

Same as above, we should solve the minimization problem.

$\displaystyle\underset{\alpha(x_0), \beta_j(x_0)}{\text{argmin}}\sum_{i=1}^nK_\lambda(x_0, x_i)\left\\{y_i-\alpha(x_0)-\sum_{j=1}^d\beta_j(x_0)x_i^j\right\\}^2$

The bias of $d$th-order local polynomial only depends on the $(d+1)$th derivative and the higher-order terms. We can prove this by the similar way as in the local linear.

As a result, local linear regression cannot control the bias related to the interior curvature. Local polynomial can be the solution, but it also increases the model variance. Therefore, we use the local polynomial with 

---

**Reference**

1. Elements of Statistical Learning