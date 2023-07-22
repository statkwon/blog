---
title: "QR-Decomposition"
date: 2021-02-20
categories:
  - "ML"
tags:
  - "QR-Decomposition"
sidebar: false
---

When column vectors of a matrix $A$ who has a full column rank are $\\{\mathbf{w}\_1, \mathbf{w}\_2, \ldots, \mathbf{w}\_k\\}$, an orthonormal basis made through the Gram-Schmidt process are $\\{\mathbf{q}\_1, \mathbf{q}\_2, \ldots, \mathbf{q}\_k\\}$. Here we can express $\mathbf{w}\_i$ as below.

$\begin{aligned}
\mathbf{w}\_1&=(\mathbf{w}\_1\cdot\mathbf{q}\_1)\mathbf{q}\_1 \\\\
\mathbf{w}\_2&=(\mathbf{w}\_2\cdot\mathbf{q}\_1)\mathbf{q}\_1+(\mathbf{w}\_2\cdot\mathbf{q}\_2)\mathbf{q}\_2 \\\\
\vdots \\\\
\mathbf{w}\_k&=(\mathbf{w}\_k\cdot\mathbf{q}\_1)\mathbf{q}\_1+(\mathbf{w}\_k\cdot\mathbf{q}\_2)\mathbf{q}\_2+\cdots+(\mathbf{w}\_k\cdot\mathbf{q}\_k)\mathbf{q}\_k
\end{aligned}$

Now we can write down this equations with the matrix multiplication.

$\begin{bmatrix} \mathbf{w}\_1 & \mathbf{w}\_2 & \cdots & \mathbf{w}\_k \end{bmatrix}=\begin{bmatrix} \mathbf{q}\_1 & \mathbf{q}\_2 & \cdots & \mathbf{q}\_k \end{bmatrix}\begin{bmatrix} (\mathbf{w}\_1\cdot\mathbf{q}\_1)  & (\mathbf{w}\_2\cdot\mathbf{q}\_1) & \cdots & (\mathbf{w}\_k\cdot\mathbf{q}\_1) \\\\ 0 & (\mathbf{w}\_2\cdot\mathbf{q}\_2) & \cdots & (\mathbf{w}\_k\cdot\mathbf{q}\_2) \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & (\mathbf{w}\_k\cdot\mathbf{q}\_k) \end{bmatrix}$

We can see that the matrix $A$ can be decomposed to the multiplication of these two matrces. This is called a $QR$-decomposition of $A$.

---

**Reference**

1. Contemporary Linear Algebra
