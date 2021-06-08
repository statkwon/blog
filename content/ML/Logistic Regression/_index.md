---
title: "Logistic Regression"
date: 2021-03-01
weight: 7
---

**Linear Regression for Qualitative Output**

Our goal is to model the posterior probability $P(Y_k\vert\mathbf{X})$ or the discriminant function $\delta_k(\mathbf{X})$, because we want to set a decision boundary as $\\{\mathbf{X}:P(Y_k\vert\mathbf{X})=P(Y_l\vert\mathbf{X})\\}$ or $\\{\mathbf{X}:\delta_k(\mathbf{X})=\delta_l(\mathbf{X})\\}$. We will classify $\mathbf{X}$ to the class with the largest value of $P(Y_k\vert\mathbf{X})$ or $\delta_k(\mathbf{X})$.

First, we can consider the case fitting a linear model for the posterior probability as $P(Y_k\vert\mathbf{X})=\mathbf{X}\boldsymbol{\beta}_k$ and it is quite reasonable in that

$\text{E}[Y_k\vert\mathbf{X}]=P(Y_1\vert\mathbf{X})\cdot0+\cdots+P(Y_k\vert\mathbf{X})\cdot1+\cdots+P(Y_K\vert\mathbf{X})\cdot0=P(Y_k\vert\mathbf{X})$.

The figure below depicts this situation.


{{<figure src="/fig9.jpeg" width="600">}}

However, this method has some disadvantages. We know that a probability should be between $0$ and $1$ and the sum of all probabilities should be $1$. If we assume the linear form of posterior probability, it only satisfies that $\sum_k\hat{P}(Y_k\vert\mathbf{X})=1$ for any $\mathbf{X}$. The probability can be negative or greater than $1$ as the figure above. We can handle this problem by using another form of function.

---

**Logistic Regression**

{{<figure src="/fig10.jpeg" width="600">}}

---

**Reference**

1. Elements of Statistical Learning