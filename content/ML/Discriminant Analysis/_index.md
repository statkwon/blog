---
title: "Discriminant Analysis"
date: 2021-03-05
weight: 8
---

**Linear Discriminant Analysis**

LDA is similar to logistic regression in that it tries to estimate the poterior probability. However, compare to the logistic regression which directly assumes the logistic form, it makes use of Bayes' Rule.

$P(Y_k\vert\mathbf{X})\approx P(\mathbf{X}\vert Y_k)P(Y_k)$

From now on, we will use $\pi_k$ instead of $P(Y_k)$ for simplicity. LDA rather assumes that $P(\mathbf{X}\vert Y_k)$ follows a multivariate gaussian distribution with a common covariance matrix $\Sigma$ for all $k$, i.e.

$P(\mathbf{X}\vert Y_k)=\dfrac{1}{(2\pi)^{p/2}\vert\Sigma\vert^{1/2}}e^{-\frac{1}{2}(\mathbf{X}-\boldsymbol{\mu}_k)^T\Sigma^{-1}(\mathbf{X}-\boldsymbol{\mu}_k)}$.

Then we can set the decision boundary as $\\{\mathbf{X}:P(Y_k\vert\mathbf{X})=P(Y_l\vert\mathbf{X})\\}=\\{\mathbf{X}:P(\mathbf{X}\vert Y_k)\pi_k=P(\mathbf{X}\vert Y_l)\pi_l\\}$ and this can be modified as

$\left\\{\mathbf{X}:P(\mathbf{X}\vert Y_k)\pi_k=P(\mathbf{X}\vert Y_l)\pi_l\right\\} \\\\
\Leftrightarrow \left\\{\mathbf{X}:\log{P(\mathbf{X}\vert Y_k)}+\log{\pi_k}=\log{P(\mathbf{X}\vert Y_l)}+\log{\pi_l}\right\\} \\\\
\Leftrightarrow \left\\{\mathbf{X}:\mathbf{X}^T\Sigma^{-1}\boldsymbol{\mu}_k-\dfrac{1}{2}\boldsymbol{\mu}_k^T\Sigma^{-1}\boldsymbol{\mu}_k+\log{\pi_k}=\mathbf{X}^T\Sigma^{-1}\boldsymbol{\mu}_l-\dfrac{1}{2}\boldsymbol{\mu}_l^T\Sigma^{-1}\boldsymbol{\mu}_l+\log{\pi_l}\right\\} \\\\
\Leftrightarrow \left\\{\mathbf{X}:\log{\dfrac{P(\mathbf{X}\vert Y_k)}{P(\mathbf{X}\vert Y_l)}}=\log{\dfrac{\pi_l}{\pi_k}}\right\\} \\\\
\Leftrightarrow \left\\{\mathbf{X}:-\dfrac{1}{2}(\boldsymbol{\mu}_k+\boldsymbol{\mu}_l)^T\Sigma^{-1}(\boldsymbol{\mu}_k-\boldsymbol{\mu}_l)+\mathbf{X}^T\Sigma^{-1}(\boldsymbol{\mu}_k-\boldsymbol{\mu}_l)=\log{\dfrac{\pi_l}{\pi_k}}\right\\}$

We call $\delta_k(\mathbf{X})=\mathbf{X}^T\Sigma^{-1}\boldsymbol{\mu}_k-\dfrac{1}{2}\boldsymbol{\mu}_k^T\Sigma^{-1}\boldsymbol{\mu}_k+\log{\pi_k}$ as a linear discriminant function and we will classify $\mathbf{X}$ to the class with the largest value of $\delta_k(\mathbf{X})$. However, we don't know the exact value of $\boldsymbol{\mu}_k$, $\Sigma$ and $\pi_k$, so we need to estimate them. Usually we estimate $\boldsymbol{\mu}_k$ as a sample mean vector $\bar{\mathbf{X}}_k$, $\Sigma$ as a covariance matrix $S$, and $\pi_k$ as $\dfrac{n_k}{n}$, where $n_k$ is the number of $k$th class observations. The masking problem with more than $3$ classes in linear regression can be avoided by using LDA.

---

**Quadratic Discriminant Analysis**

A similar method without the equality assumption of $\Sigma_k$ is called a quadratic discriminant analysis. Due to this weak assumption, the convenient cancellation in LDA does not occur in QDA. Similar to LDA, we can get a quadratic discriminant function as

$\delta_k(\mathbf{X})=-\dfrac{1}{2}\log{\vert\Sigma_k\vert}-\dfrac{1}{2}(\mathbf{X}-\boldsymbol{\mu}_k)^T\Sigma_k^{-1}(\mathbf{X}-\boldsymbol{\mu}_k)+\log{\pi_k}$.

---

We have to calculate the inverse of the covariance matrix for both methods and this can be simplified by using the eigenvalue decomposition of $\hat{\Sigma}_k$. Substituting $\hat{\Sigma}_k$ with $U_kD_kU_k^T$, we can obtain

$\delta_k(\mathbf{X})=-\dfrac{1}{2}\sum_l\log{d_{kl}}-\dfrac{1}{2}[U_k^T(\mathbf{X}-\boldsymbol{\mu}_k)]^TD_k^{-1}[U_k^T(\mathbf{X}-\boldsymbol{\mu}_k)]+\log{\pi_k}$.

Both techniques are quite useful despite the strict assumptions about gaussian distribution and equal covariance because the decision boundaries can be estimated with much lower variance than more exotic alternatives.

---

**Regularized Discriminant Analysis**

Regularized discriminant analysis is a compromise between LDA and QDA, which allows one to shrink the separate covariances of QDA toward a common covariance as in LDA. The regularized covariance matrices have the form

$\hat{\Sigma}_k(\alpha)=\alpha\hat{\Sigma}_k+(1-\alpha)\hat{\Sigma}$,

where $\hat{\Sigma}$ is the pooled covariance matrix as used in LDA. $\alpha$ can be selected by cross-validation.

---

**Reference**

1. Elements of Statistical Learning