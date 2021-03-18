---
title: "Ridge Regression"
date: 2021-03-10
draft: true
weight: 5
TableOfContents: true
---

## 1. Calculus Perspective

### 1.1. Lagrange Multiplier

### 1.2. Interpretation

$$\begin{aligned}
\hat{\beta}^{\text{ridge}}&=\underset{\beta}{\text{argmin}}\left\\{\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2+\lambda\sum_{j=1}^p\beta_j^2\right\\} \\\\
&=\underset{\beta}{\text{argmin}}\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2 \quad \text{subject to} \quad \sum_{j=1}^p\beta_j^2≤t
\end{aligned}$$

## 2. Linear Algebraic Perspective

### 2.1. Singular Value Decomposition

### 2.2. Interpretation

$\begin{aligned}
\dfrac{\partial\text{SSE}}{\partial\beta}&=\dfrac{\partial\left\\{(y-X\beta)^T(y-X\beta)+\lambda\beta^T\beta\right\\}}{\partial\beta} \\\\
&=\dfrac{\partial\left\\{(y^T-\beta^TX^T)(y-X\beta)+\lambda\beta^T\beta\right\\}}{\partial\beta} \\\\
&=\dfrac{\partial(y^Ty-\beta^TX^Ty-y^TX\beta+\beta^TX^TX\beta+\lambda\beta^T\beta)}{\partial\beta} \\\\
&=-2X^Ty+2X^TX\beta+2\lambda\beta
\end{aligned}$

이 식이 $0$이 되게 하는 $\beta$는 아래와 같다.

$$\hat{\beta}^{\text{ridge}}=(X^TX+\lambda I)^{-1}X^Ty$$