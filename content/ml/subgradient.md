---
title: "Subgradient"
date: 2023-12-21
lastmod: 2023-12-21
categories:
  - "ML"
tags:
  - "Optimization"
  - "Subgradient"
sidebar: false
---

## Subgradient

convex & differentiable한 함수 $f$는 다음과 같은 식을 만족한다.

$$f(\mathbf{y})\geq f(\mathbf{x})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x}),\\;^\forall\mathbf{x}, \mathbf{y}$$

이에 기반하여 convex한 함수 $f$의 subgradient를 다음과 같은 식을 만족하는 $\mathbf{g}$로 정의한다.

$$f(\mathbf{y})\geq f(\mathbf{x})+\mathbf{g}^T(\mathbf{y}-\mathbf{x}),\\;^\forall\mathbf{y}$$

subgradient $\mathbf{g}$는 $f$가 $\mathbf{x}$에서 differentiable하지 않아도 구할 수 있으며, 여러 개 존재할 수 있다. 만약 $f$가 $\mathbf{x}$에서 differentiable하면 $\mathbf{g}$는 $\nabla f(\mathbf{x})$와 같아진다.

{{<figure src="/ml/subg1.png" width="400">}}

예를 들어, $f(x)=\vert x\vert$인 경우 $x$의 subgradient는 $g=\begin{cases} \text{sign}(x), & x\neq0 \\\\ c\in[-1, 1], & x=0 \end{cases}$가 된다.

---

**Reference**

1. https://convex-optimization-for-all.github.io/contents/chapter07/2021/03/25/07_01_subgradient/
