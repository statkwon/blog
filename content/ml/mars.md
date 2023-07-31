---
title: "MARS"
date: 2022-01-04
categories:
  - "ML"
tags:
  - "Spline"
sidebar: false
---

## Multivariate Adaptive Regression Splines

$\displaystyle f(\mathbf{x})=\beta\_0+\sum\_{m=1}^M\beta\_mh\_m(\mathbf{x})$, where $h\_m(\mathbf{x})=(X\_j-t)\_+\\;\text{or}\\; (t-X\_j)\_+$

MARS는 다음과 같은 특수한 형태의 Basis Function들의 선형 결합으로 반응변수를 추정하는 알고리즘이다.

$(x-t)_+=\begin{cases} x-t & \text{if}\\; x>t \\\\ 0 & \text{otherwise} \end{cases} \quad\text{and}\quad (t-x)\_+=\begin{cases} t-x & \text{if}\\; x<t \\\\ 0 & \text{otherwise} \end{cases}$

{{<figure src="/ml/mars1.png" width="600">}}

MARS는 Forward Modeling과 Backward Deletion의 두 가지 단계로 구성되어 있다.

## Forward Modeling

$(x-t)\_+$와 $(t-x)\_+$가 서로 대칭적인 구조를 가지므로 두 함수를 묶어 Reflected Pair라고 하자. Forward Modeling 단계에서는 이러한 Reflected Pair들의 집합

$\mathcal{C}=\\{(X_j-t)\_+, (t-X_j)\_+\\}$, where $t\in\\{x_{1j}, x_{2j}, \ldots, x_{Nj}\\}$ and $j=1, 2, \ldots, p$에서 특정한 기준에 따라 모형에 추가할 항을 선택한다.

$\mathcal{M}$을 모형에 포함되어 있는 Basis Function들의 집합이라고 할 때, $h_0(\mathbf{x})=1$을 시작점으로 하여 모형의 크기가 미리 지정한 값에 도달할 때까지 Error Sum of Squares를 가장 크게 증가시키는 $\hat{\beta}_{M+1}h_l(\mathbf{x})\cdot(X_j-t)\_++\hat{\beta}\_{M+2}h_l(\mathbf{x})\cdot(t-X_j)\_+$, where $h_l\in\mathcal{M}$을 순차적으로 모형에 추가한다. 이때 $(X_1-2)\_+^n$과 같은 고차항이 생성되는 것을 막기 위해 각각의 항은 최대 한 번씩만 선택할 수 있도록 제한을 둔다.

1st Stage: $f(\mathbf{x})=\hat{\beta}\_0$ \
2nd Stage: $f(\mathbf{x})=\hat{\beta}\_0+\hat{\beta}\_1(X_2-1)_++\hat{\beta}_2(1-X_2)\_+$ \
3rd Stage: $f(\mathbf{x})=\hat{\beta}\_0+\hat{\beta}\_1(X_2-1)\_++\hat{\beta}_2(1-X_2)\_++\hat{\beta}\_3(X_2-1)\_+(X_1-2)\_++\hat{\beta}_4(X_2-1)\_+(2-X_1)\_+$ \
$\quad\vdots$

추가적으로, 모형의 해석력을 유지하기 위해 이러한 과정을 통해 생성되는 Interaction Term의 차수에 제한을 두기도 한다.

## Backward Deletion

Forward Modeling 과정을 통해 생성된 모형은 그 크기가 굉장히 크기 때문에 Overfitting이 발생할 가능성이 높다. 따라서 Backward Deletion 과정을 통해 이러한 문제를 해결하게 된다. 우리는 앞서 생성된 모형에서 Error Sum of Squares를 가장 적게 증가시키는 항을 하나씩 제거하며 서로 다른 크기의 Submodel $\hat{f}_\lambda$의 배열을 구할 수 있다. 최적의 $\lambda$는 Cross-Validation을 사용하여 구할 수 있지만, 계산 비용을 줄이기 위하여 아래와 같은 형태의 Generalized Cross-Validation을 사용한다.

$\text{GCV}(\lambda)=\dfrac{\sum_{i=1}^N(y_i-\hat{f}_\lambda(x_i))^2}{(1-M(\lambda)/N)^2}$, where $M(\lambda)$ is the effective number of parameters in the model

즉, $\text{GCV}(\lambda)$를 최소화하는 $\lambda$에 대한 $\hat{f}_\lambda$를 최종 모형으로 선택한다.

## Advantages of MARS

- MARS는 특수한 형태의 Basis Function을 사용함으로써 모형에 필요한 파라미터의 개수가 제한적으로 늘어난다. 예를 들어, 서로 다른 두 항이 곱해지는 경우 두 항이 모두 0이 아닌 영역에서만 값게 된다. 이러한 특성은 고차원의 상황일수록 파라미터의 개수를 신중하게 늘려야한다는 점에서 장점으로 작용한다.
- 이러한 형태의 Basis Function은 계산 비용을 줄여준다는 이점을 갖기도 한다.
- Higher-Order Interaction을 추가하기 위해서는 그것의 Lower-Order Interaction이 반드시 모형에 먼저 포함되어 있어야한다는 점에서 MARS의 Forward Modeling 단계는 Hierarchical한 구조를 갖는다. 이러한 구조는 상호작용의 범위가 넓어질수록 지수적으로 증가하는 대안들을 탐색하는 소모적인 과정을 막아준다.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
