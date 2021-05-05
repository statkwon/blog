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

---

**Application to Linear Regression**

이제 Gram-Schmidt Process를 $X$의 Column Vector들에 적용해보도록 하자.

$$X=\begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix}$$

![fig 1](/ml/fig/fig1.png)

첫 번째 벡터 $\mathbf{1}$(또는 $\mathbf{x}_0$)을 $\mathbf{z}_0$이라 하면, $\mathbf{z}_0$에 Orthogonal한 벡터 $\mathbf{z}_1$을 구할 수 있다. 두 번째 벡터 $\mathbf{z}_1$을 $\mathbf{1}$이 Span하는 공간에 Projection한 것을 $\text{proj}\_{X_0}\mathbf{x}_1$이라고 하면 $\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1$은 $\mathbf{z}_0$에 Orthogonal한 벡터가 된다. 따라서 $\mathbf{z}_1=\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1$이라고 할 수 있다.

$$\mathbf{z}_1=\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1=\mathbf{x}_1-\dfrac{\mathbf{x}_1\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert^2}\mathbf{z}_0$$

다음으로 $\mathbf{z}_0$와 $\mathbf{z}_1$에 대해 Orthogonal한 벡터 $\mathbf{z}_2$를 구할 수 있다. 마찬가지로 $\mathbf{z}_0$와 $\mathbf{z}_1$이 Span하는 공간에 $\mathbf{x}_2$를 Projection한 것을 $\text{proj}\_{X_1}\mathbf{x}_2$라고 하면 $\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2$는 $\mathbf{z}_0$와 $\mathbf{z}_1$에 모두 Orthogonal한 벡터가 된다. 따라서 $\mathbf{z}_2=\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2$이다.

$$\mathbf{z}_2=\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2=\mathbf{x}_2-\dfrac{\mathbf{x}_2\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert^2}\mathbf{z}_0-\dfrac{\mathbf{x}_2\cdot\mathbf{z}_1}{\Vert\mathbf{z}_1\Vert^2}\mathbf{z}_1$$

이런 과정을 반복하면 $\mathbf{z}_p$까지 구할 수 있다. $\begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix}$를 $\begin{bmatrix} \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_p \end{bmatrix}$로 굳이 바꾸어 준 이유는 $\mathbf{y}$를 각각의 $\mathbf{z}_i$에 Projection하여 $i$ 번째 $X$ 변수의 회귀 계수를 $\hat{\beta}_i=\dfrac{\mathbf{y}\cdot\mathbf{z}_i}{\Vert\mathbf{z}_i\Vert^2}$로 쉽게 계산할 수 있기 때문이다. 이유는 아래의 그림을 통해 확인할 수 있다.

![fig 2](/ml/fig/fig2.png)

우리는 $\hat{\mathbf{y}}$을 $a\mathbf{1}+b\mathbf{x}_1$라고 나타낼 수 있다. 이때 $b\mathbf{x}_1$은 $(c\mathbf{1}-a\mathbf{1})+d\mathbf{z}_1$과 같다. 이를 앞의 식에 대입해보면 $\hat{\mathbf{y}}$은 곧 $c\mathbf{1}+d\mathbf{z}_1$과 같다는 것을 확인할 수 있다. 즉, $\mathbf{y}$를 $\mathbf{x}_1$에 Projection 했을 때의 회귀 계수와 $\mathbf{z}_1$에 Projection 했을 때의 회귀 계수는 같은 값을 갖게 된다. 이에 더해, $\mathbf{z}_1, \ldots, \mathbf{z}_p$가 서로 Orthogonal 하기 때문에 $\mathbf{y}$를 $\mathbf{z}_1, \ldots, \mathbf{z}_p$에 Projection 했을 때의 회귀 계수는 $\mathbf{y}$를 각각의 $\mathbf{z}_i$에 Projection 했을 때의 회귀 계수와 같다.