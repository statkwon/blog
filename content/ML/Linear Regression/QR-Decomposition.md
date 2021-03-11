---
title: "QR-Decomposition"
date: 2021-03-10
draft: true
weight: 2
TableOfContents: true
---

## 1. Gram-Schmidt Process

모든 $n$차원 공간의 영공간이 아닌 Subspace는 Orthonormal Basis를 갖는다. Gram-Schmidt Process는 $n$차원 공간의 영공간이 아닌 Subspace의 Basis를 Orthonormal Basis로 바꿔주는 과정이다.

$\\{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_k\\}$를 Basis로 갖는 $n$차원 공간의 Subspace $W$의 Orthonormal Basis를 구해보자.

![LA fig 7.9.1](/ml/linear_regression/la_fig7.9.1.png)

1. $\mathbf{v}_1=\mathbf{w}_1$
2. $\mathbf{v}_2=\mathbf{w}_2-\text{proj}\_{W_1}\mathbf{w}_2=\mathbf{w}_2-\dfrac{\mathbf{w}_2\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1$
3. $\mathbf{v}_3=\mathbf{w}_3-\text{proj}\_{W_2}\mathbf{w}_3=\mathbf{w}_3-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_1}{\Vert\mathbf{v}_1\Vert^2}\mathbf{v}_1-\dfrac{\mathbf{w}_3\cdot\mathbf{v}_2}{\Vert\mathbf{v}_2\Vert^2}\mathbf{v}_2$
4. $\mathbf{v}_k$까지 동일한 과정을 반복한다.
5. $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$를 각각 Normalize 한다.

## 2. $QR$-Decomposition

## 3. Application to Linear Regression