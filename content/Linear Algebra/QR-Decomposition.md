---
title: "QR-Decomposition"
date: 2021-03-10
draft: false
---

**Gram-Schmidt Process**

Every nonzero subspaces in $R^n$ have its own prthonormal basis. Gram-Schmidt process is the process to change the basis of the nonzero subspace in $R^n$ to  orthonormal basis. Now we will check the process to find the orthonormal basis of subspace $W$ in $R^n$ whose basis is $\\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k\\}$.

{{<figure src="/la_fig_7.9.2.png" width="400" height="200">}}


If we let the first vector $\mathbf{w}_1$ as $\mathbf{v}_1$, then we can find $\mathbf{v}_2$, which is an orthogonal vector to $\mathbf{v}_1$. Now let $\text{proj}\_{W_1}\mathbf{w}_2$ as the projection of the second vector $\mathbf{w}_2$ on to the subspace spanned by $\mathbf{v}_1$, then $\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2$ is an orthogonal vector to $\mathbf{v}_1$. Thus, we can say that $\mathbf{v}_2=\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2$.

$$\mathbf{v}_2=\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2=\mathbf{w}_2-\dfrac{\mathbf{w}_2\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1$$

{{<figure src="/la_fig_7.9.3.png" width="400" height="200">}}

Similarly, we can find a vector $\mathbf{v}_3$ which is orthogonal to $\mathbf{v}_1$ and $\mathbf{v}_2$. Let $\text{proj}\_{W_2}\mathbf{w}_3$ as the projection of $\mathbf{w}_3$ onto the subspace spanned by $\mathbf{v}_1$ and $\mathbf{v}_2$, then $\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3$ will be orthogonal to $\mathbf{v}_1$ and $\mathbf{v}_2$. Therefore, $\mathbf{v}_3=\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3$.

$$\mathbf{v}_3=\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3=\mathbf{w}_3-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_2}{\Vert\mathbf{v}_2\Vert^2}\mathbf{v}_2$$

If we repeat this process several times, we can get up to $\mathbf{v}_k$. Here $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ are orthogonal vectors, so we can normalize these vectors and then make an orthonormal basis as $\mathbf{q}_i=\dfrac{\mathbf{v}_i}{\Vert\mathbf{v}_i\Vert^2}$.

---

**$QR$-Decomposition**

When column vectors of a matrix $A$ who has a full column rank are $\\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k\\}$, an orthonormal basis made through the Gram-Schmidt process are $\\{\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_k\\}$. Here we can express $\mathbf{w}_i$ as below.

$\begin{aligned}
\mathbf{w}_1&=(\mathbf{w}_1\cdot\mathbf{q}_1)\mathbf{q}_1 \\\\
\mathbf{w}_2&=(\mathbf{w}_2\cdot\mathbf{q}_1)\mathbf{q}_1+(\mathbf{w}_2\cdot\mathbf{q}_2)\mathbf{q}_2 \\\\
\vdots \\\\
\mathbf{w}_k&=(\mathbf{w}_k\cdot\mathbf{q}_1)\mathbf{q}_1+(\mathbf{w}_k\cdot\mathbf{q}_2)\mathbf{q}_2+\cdots+(\mathbf{w}_k\cdot\mathbf{q}_k)\mathbf{q}_k
\end{aligned}$

Now we can write down this equations with the matrix multiplication.

$\begin{bmatrix} \mathbf{w}_1 & \mathbf{w}_2 & \cdots & \mathbf{w}_k \end{bmatrix}=\begin{bmatrix} \mathbf{q}_1 & \mathbf{q}_2 & \cdots & \mathbf{q}_k \end{bmatrix}\begin{bmatrix} (\mathbf{w}_1\cdot\mathbf{q}_1)  & (\mathbf{w}_2\cdot\mathbf{q}_1) & \cdots & (\mathbf{w}_k\cdot\mathbf{q}_1) \\\\ 0 & (\mathbf{w}_2\cdot\mathbf{q}_2) & \cdots & (\mathbf{w}_k\cdot\mathbf{q}_2) \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & (\mathbf{w}_k\cdot\mathbf{q}_k) \end{bmatrix}$

We can see that the matrix $A$ can be decomposed to the multiplication of these two matrces. This is called a $QR$-decomposition of $A$.