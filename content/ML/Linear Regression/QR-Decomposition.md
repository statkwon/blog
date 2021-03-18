---
title: "QR-Decomposition"
date: 2021-03-10
draft: false
weight: 3
TableOfContents: true
---

## 1. Gram-Schmidt Process

모든 $n$차원 공간의 영공간이 아닌 Subspace는 Orthonormal Basis를 갖는다. Gram-Schmidt Process는 $n$차원 공간의 영공간이 아닌 Subspace의 Basis를 Orthonormal Basis로 바꿔주는 과정이다.

$\\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k\\}$를 Basis로 갖는 $n$차원 공간의 Subspace $W$의 Orthonormal Basis를 구해보자.

![LA fig 7.9.2](/ml/linear-regression/la_fig7.9.2.png)

첫 번째 벡터 $\mathbf{w}_1$을 $\mathbf{v}_1$이라 하면, $\mathbf{v}_1$에 Orthogonal한 벡터 $\mathbf{v}_2$를 구할 수 있다. 두 번째 벡터 $\mathbf{w}_2$를 $\mathbf{v}_1$이 Span하는 공간에 Projection한 것을 $\text{proj}\_{W_1}\mathbf{w}_2$라고 하면 $\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2$는 $\mathbf{v}_1$에 Orthogonal한 벡터가 된다. 따라서 $\mathbf{v}_2=\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2$라고 할 수 있다.

$$\mathbf{v}_2=\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2=\mathbf{w}_2-\dfrac{\mathbf{w}_2\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1$$

![LA fig 7.9.3](/ml/linear-regression/la_fig7.9.3.png)

다음으로 $\mathbf{v}_1$과 $\mathbf{v}_2$에 대해 Orthogonal한 벡터 $\mathbf{v}_3$를 구할 수 있다. 마찬가지로 $\mathbf{v}_1$과 $\mathbf{v}_2$가 Span하는 공간에 $\mathbf{w}_3$를 Projection한 것을 $\text{proj}\_{W_2}\mathbf{w}_3$라고 하면 $\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3$는 $\mathbf{v}_1$과 $\mathbf{v}_2$에 모두 Orthogonal한 벡터가 된다. 따라서 $\mathbf{v}_3=\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3$이다.

$$\mathbf{v}_3=\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3=\mathbf{w}_3-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_2}{\Vert\mathbf{v}_2\Vert^2}\mathbf{v}_2$$

이러한 과정을 반복하면 $\mathbf{v}_k$까지 구할 수 있다. 이때 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$는 Orthogonal한 벡터들이므로 각각의 벡터를 Normalize하여 Orthonormal Basis를 만들어줄 수 있다.

$$\mathbf{q}_i=\dfrac{\mathbf{v}_i}{\Vert\mathbf{v}_i\Vert^2}$$

## 2. $QR$-Decomposition

Full Column Rank를 갖는 어떤 행렬 $A$의 Column Vector가 $\\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k\\}$일 때, Gram-Schmidt 과정을 통해 이를 Orthonormal Basis로 바꾸어준 것을 $\\{\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_k\\}$라고 하자. 이때 $\mathbf{w}_i$를 다음과 같이 표현할 수 있다.

$\begin{aligned}
\mathbf{w}_1&=(\mathbf{w}_1\cdot\mathbf{q}_1)\mathbf{q}_1 \\\\
\mathbf{w}_2&=(\mathbf{w}_2\cdot\mathbf{q}_1)\mathbf{q}_1+(\mathbf{w}_2\cdot\mathbf{q}_2)\mathbf{q}_2 \\\\
\vdots \\\\
\mathbf{w}_k&=(\mathbf{w}_k\cdot\mathbf{q}_1)\mathbf{q}_1+(\mathbf{w}_k\cdot\mathbf{q}_2)\mathbf{q}_2+\cdots+(\mathbf{w}_k\cdot\mathbf{q}_k)\mathbf{q}_k
\end{aligned}$

이를 행렬곱의 형태로 표현하면 아래와 같다.

$$\begin{bmatrix} \mathbf{w}_1 & \mathbf{w}_2 & \cdots & \mathbf{w}_k \end{bmatrix}=\begin{bmatrix} \mathbf{q}_1 & \mathbf{q}_2 & \cdots & \mathbf{q}_k \end{bmatrix}\begin{bmatrix} (\mathbf{w}_1\cdot\mathbf{q}_1)  & (\mathbf{w}_2\cdot\mathbf{q}_1) & \cdots & (\mathbf{w}_k\cdot\mathbf{q}_1) \\\\ 0 & (\mathbf{w}_2\cdot\mathbf{q}_2) & \cdots & (\mathbf{w}_k\cdot\mathbf{q}_2) \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & (\mathbf{w}_k\cdot\mathbf{q}_k) \end{bmatrix}$$

따라서 $A$가 위와 같은 두 행렬의 곱으로 분해될 수 있다는 것을 확인할 수 있다. 이것을 $A$의 $QR$-Decomposition이라고 한다.

## 3. Application to Linear Regression

이제 Gram-Schmidt Process를 $X$의 Column Vector들에 적용해보도록 하자.

$$X=\begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix}$$

![fig 1](/ml/linear-regression/fig1.png)

첫 번째 벡터 $\mathbf{1}$(또는 $\mathbf{x}_0$)을 $\mathbf{z}_0$이라 하면, $\mathbf{z}_0$에 Orthogonal한 벡터 $\mathbf{z}_1$을 구할 수 있다. 두 번째 벡터 $\mathbf{z}_1$을 $\mathbf{1}$이 Span하는 공간에 Projection한 것을 $\text{proj}\_{X_0}\mathbf{x}_1$이라고 하면 $\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1$은 $\mathbf{z}_0$에 Orthogonal한 벡터가 된다. 따라서 $\mathbf{z}_1=\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1$이라고 할 수 있다.

$$\mathbf{z}_1=\mathbf{x}_1-\text{proj}\_{X_0}\mathbf{x}_1=\mathbf{x}_1-\dfrac{\mathbf{x}_1\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert^2}\mathbf{z}_0$$

다음으로 $\mathbf{z}_0$와 $\mathbf{z}_1$에 대해 Orthogonal한 벡터 $\mathbf{z}_2$를 구할 수 있다. 마찬가지로 $\mathbf{z}_0$와 $\mathbf{z}_1$이 Span하는 공간에 $\mathbf{x}_2$를 Projection한 것을 $\text{proj}\_{X_1}\mathbf{x}_2$라고 하면 $\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2$는 $\mathbf{z}_0$와 $\mathbf{z}_1$에 모두 Orthogonal한 벡터가 된다. 따라서 $\mathbf{z}_2=\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2$이다.

$$\mathbf{z}_2=\mathbf{x}_2-\text{proj}\_{X_1}\mathbf{x}_2=\mathbf{x}_2-\dfrac{\mathbf{x}_2\cdot\mathbf{z}_0}{\Vert\mathbf{z}_0\Vert^2}\mathbf{z}_0-\dfrac{\mathbf{x}_2\cdot\mathbf{z}_1}{\Vert\mathbf{z}_1\Vert^2}\mathbf{z}_1$$

이런 과정을 반복하면 $\mathbf{z}_p$까지 구할 수 있다. $\begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \cdots & \mathbf{x}_p \end{bmatrix}$를 $\begin{bmatrix} \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_p \end{bmatrix}$로 굳이 바꾸어 준 이유는 $\mathbf{y}$를 각각의 $\mathbf{z}_i$에 Projection하여 $i$ 번째 $X$ 변수의 회귀 계수를 $\hat{\beta}_i=\dfrac{\mathbf{y}\cdot\mathbf{z}_i}{\Vert\mathbf{z}_i\Vert^2}$로 쉽게 계산할 수 있기 때문이다. 이유는 아래의 그림을 통해 확인할 수 있다.

![fig 2](/ml/linear-regression/fig2.png)

우리는 $\hat{\mathbf{y}}$을 $a\mathbf{1}+b\mathbf{x}_1$라고 나타낼 수 있다. 이때 $b\mathbf{x}_1$은 $(c\mathbf{1}-a\mathbf{1})+d\mathbf{z}_1$과 같다. 이를 앞의 식에 대입해보면 $\hat{\mathbf{y}}$은 곧 $c\mathbf{1}+d\mathbf{z}_1$과 같다는 것을 확인할 수 있다. 즉, $\mathbf{y}$를 $\mathbf{x}_1$에 Projection 했을 때의 회귀 계수와 $\mathbf{z}_1$에 Projection 했을 때의 회귀 계수는 같은 값을 갖게 된다. 이에 더해, $\mathbf{z}_1, \ldots, \mathbf{z}_p$가 서로 Orthogonal 하기 때문에 $\mathbf{y}$를 $\mathbf{z}_1, \ldots, \mathbf{z}_p$에 Projection 했을 때의 회귀 계수는 $\mathbf{y}$를 각각의 $\mathbf{z}_i$에 Projection 했을 때의 회귀 계수와 같다.