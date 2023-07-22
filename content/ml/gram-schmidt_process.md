---
title: "Gram-Schmidt Process"
date: 2021-02-20
categories:
  - "ML"
tags:
  - "Orthogonalization"
  - "Gram-Schmidt"
sidebar: false
---

Every nonzero subspaces in $R^n$ have its own prthonormal basis. Gram-Schmidt process is the process to change the basis of the nonzero subspace in $R^n$ to  orthonormal basis. Now we will check the process to find the orthonormal basis of subspace $W$ in $R^n$ whose basis is $\\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k\\}$.

{{<figure src="/ml/gram-schmidt1.png" width="250">}}

If we let the first vector $\mathbf{w}_1$ as $\mathbf{v}_1$, then we can find $\mathbf{v}_2$, which is an orthogonal vector to $\mathbf{v}_1$. Now let $\text{proj}\_{W_1}\mathbf{w}_2$ as the projection of the second vector $\mathbf{w}_2$ on to the subspace spanned by $\mathbf{v}_1$, then $\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2$ is an orthogonal vector to $\mathbf{v}_1$. Thus, we can say that $\mathbf{v}_2=\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2$.

$$\mathbf{v}_2=\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2=\mathbf{w}_2-\dfrac{\mathbf{w}_2\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1$$

{{<figure src="/ml/gram-schmidt2.png" width="250">}}

Similarly, we can find a vector $\mathbf{v}_3$ which is orthogonal to $\mathbf{v}_1$ and $\mathbf{v}_2$. Let $\text{proj}\_{W_2}\mathbf{w}_3$ as the projection of $\mathbf{w}_3$ onto the subspace spanned by $\mathbf{v}_1$ and $\mathbf{v}_2$, then $\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3$ will be orthogonal to $\mathbf{v}_1$ and $\mathbf{v}_2$. Therefore, $\mathbf{v}_3=\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3$.

$$\mathbf{v}_3=\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3=\mathbf{w}_3-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_2}{\Vert\mathbf{v}_2\Vert^2}\mathbf{v}_2$$

If we repeat this process several times, we can get up to $\mathbf{v}_k$. Here $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ are orthogonal vectors, so we can normalize these vectors and then make an orthonormal basis as $\mathbf{q}_i=\dfrac{\mathbf{v}_i}{\Vert\mathbf{v}_i\Vert^2}$.

---

**Reference**

1. Anton, H., & Busby, R. C. (2003). Contemporary linear algebra. Wiley.
