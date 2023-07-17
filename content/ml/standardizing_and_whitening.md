---
title: "Standardizing & Whitening"
date: 2023-07-16
categories:
  - "ML"
tags:
  - "Standardizing"
  - "Whitening"
sidebar: false
---

## Random Vector 관점

Standardizing(Normalization)은 mean이 $\boldsymbol{\mu}$이고 covariance가 $\Sigma$인 분포를 따르는 random vector $\mathbf{X}$의 분포가 mean이 $\mathbf{0}$이고 각 feature의 variance가 $1$인 분포가 되도록 처리하는 것 &rarr; 각 feature 별 연산이므로 feature 간 covariance에는 영향이 없다.

Whitening은 mean이 $\mathbf{0}$이고 covariance가 $\Sigma$인 분포를 따르는 random vector $\mathbf{X}$의 covariance가 identity matrix가 되도록 처리하는 것 &rarr; feature 간 covariance를 $0$으로 만들어 줄 수 있다.

$\mathbf{Y}=W\mathbf{X}$일 때, $\text{Cov}(\mathbf{Y})=W\text{Cov}(\mathbf{X})W^T=W\Sigma W^T$

$W\Sigma W^T$가 $I$가 되게 하려면 $W$가 $W^TW=\Sigma^{-1}$를 만족하도록 하면 된다.

$\Sigma$의 EVD를 통해 조건을 만족하는 $W$를 쉽게 찾을 수 있다. &rarr; $\Sigma=PDP^T$일 때, $W=D^{-1/2}P^T$

$\mathbf{Y}=D^{-1/2}P^T\mathbf{X}$ (PCA Whitening)

위 조건을 만족하는 $W$는 유일하지 않다. $W=PD^{-1/2}P^T$를 사용해도 된다.

$\mathbf{Y}=PD^{-1/2}P^T\mathbf{X}$ (ZCA Whitening)

## Sample Data 관점

Standardizing은 각 feature의 sample mean을 $0$, sample variance를 $1$이 되도록 처리하는 것 &rarr; 각 feature 별 연산이므로 feature 간 correlation에는 영향이 없다.

{{<figure src="/ml/whitening1.png" width="800">}}

Whitening은 데이터의 sample covariance가 identity matrix가 되도록 처리하는 것 &rarr; feature 간 correlation을 $0$으로 만들어 줄 수 있다.

$\text{Cov}(X)=PDP^T$일 때, $X_\text{whitened}=(D^{-1/2}V^TX^T)^T$

{{<figure src="/ml/whitening2.png" width="800">}}
