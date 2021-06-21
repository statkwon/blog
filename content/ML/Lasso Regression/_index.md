---
title: "Lasso Regression"
date: 2021-02-28
weight: 6
draft: true
---

Lasso regression is also a shrinkage method, but differs from ridge regression in that it uses $l_1$ penalty term.

$\begin{aligned}
\hat{\beta}^{\text{lasso}}&=\underset{\beta}{\text{argmin}}\left\\{\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2+\lambda\sum_{j=1}^p\vert\beta_j\vert\right\\} \\\\
&=\underset{\beta}{\text{argmin}}\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2 \\; \text{subject to} \\; \sum_{j=1}^p\vert\beta_j\vert≤t
\end{aligned}$

This new $l_1$ penalty makes the coefficients nonlinear in the $y_i$, and there is no closed form expression.

---

**Reference**

1. Elements of Statistical Learning