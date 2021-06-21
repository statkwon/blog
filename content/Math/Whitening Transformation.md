---
title: "Whitening Transformation"
date: 2021-03-05
---

**Whitening Transformation**

We want to find a linear transformation which transforms the covariance matrix of a random vector $\mathbf{X}$ to an identity matrix. Suppose $\mathbf{X}$ is a random vector with nonsingular covariance matrix $\Sigma$ and mean $\mathbf{0}$. Then the transformation $\mathbf{Y}=W\mathbf{X}$ with a whitening matrix $W$ satisfying the condition $W^TW=\Sigma^{-1}$ yields the whitened random vector $\mathbf{Y}$ with unit digital covariance.

$\text{Cov}(\mathbf{Y})=\text{Cov}(W\mathbf{X})=W\text{Cov}(\mathbf{X})W^T=W\Sigma W^T=I_p \quad\Rightarrow\quad W^TW=\Sigma^{-1}$

There are several ways to find $W$, but we will cover only for the ZCA whitening. It uses a eigenvalue decomposition of $\Sigma$.

$\Sigma=UDU^T=UD^{1/2}D^{1/2}U^T \quad\Rightarrow\quad \Sigma^{-1}=UD^{-1/2}D^{-1/2}U^T=W^TW$

If we let $W=D^{-1/2}U^T$, this satisfies the all conditions above. Thus, if we multiply $D^{-1/2}U^T$ to a random vector $\mathbf{X}$, we can get a random vector $\mathbf{Y}$, whose covariance matrix is an identity matrix. This is called a whitening transformation or sphering transformation.

$\mathbf{Y}=D^{-1/2}U^T\mathbf{X}$

---

This can also be applied to a data matrix. Suppose $X$ is a data matrix with zero sample mean vector. We will transform the sample covariance matrix of $X$ to an identity matrix as before.

---

**Reference**

1. [https://en.wikipedia.org/wiki/Whitening_transformation](https://en.wikipedia.org/wiki/Whitening_transformation)