---
title: "PRIM"
date: 2022-01-03
categories:
  - "ML"
tags:
  - "Spline"
  - "PRIM"
sidebar: false
---

## Patient Rule Induction Method

PRIM은 CART 알고리즘과 같이 Feature Space를 박스 형태의 영역들로 분할하는 알고리즘이다. 하지만 Binary Split을 사용하지는 않는다. CART 알고리즘이 Splitting과 Pruning의 두 가지 단계로 이루어졌듯이, PRIM 알고리즘은 Top-Down Peeling과 Bottom-Up Pasting의 두 가지 단계로 구분된다.

{{<figure src="/ml/prim1.png" width="600">}}

## Top-Down Peeling

모든 데이터가 한 박스 안에 담겨있다고 생각해보자. 박스의 한 면을 선택하고 전체 데이터 중 비율 $\alpha$ 만큼의 데이터가 제외될 때까지 선택된 면을 기준으로 박스의 크기를 줄인다. 이때 줄어든 박스에 속한 데이터의 반응변수의 평균을 최대화하는 것을 기준으로 면을 선택한다. 이후 박스 안에 속한 데이터의 수가 최소 기준에 도달할 때까지 동일한 과정을 반복한다.

## Bottom-Up Pasting

Pasting 단계에서는 Peeling 단계를 통해 생성된 박스를 다시 확장하는 작업이 이루어진다. 확장된 박스의 반응변수의 평균이 더 이상 증가하지 않을 때까지 모든 면에 대해 박스를 확장한다. 이후 서로 다른 크기의 확장된 박스들의 배열에 대해 Cross-Validation을 통해 최적의 박스 크기를 결정한다.

이후 기존 박스에 속한 데이터를 제외한 나머지 데이터에 대해 Peeling & Pasting 과정을 반복하며 여러 개의 박스를 생성한다. 결과적으로 생성된 각각의 박스들은 $(a\leq X\_i\leq b)$ and $(c\leq X\_j\leq d)$와 같은 형태를 갖게 된다.

## Pros and Cons of PRIM

PRIM은 범주형 변수나 결측치를 처리함에 있어 CART 알고리즘과 유사한 방식을 사용한다. Regression과 Classification 문제에 모두 사용할 수 있지만, Classification의 경우 오직 Binary Case에 한해서만 사용이 가능하다. Multiclass 문제일 경우 Baseline을 정하고 해당 Baseline을 기준으로 One vs One 식의 적용만이 가능하다. Patient Rule이라는 이름에서 알 수 있듯이, PRIM은 Feature Space를 천천히 분할함으로써 더 좋은 분할을 찾을 수 있다는 장점을 갖는다. CART 알고리즘과 비교했을 때, CART 알고리즘은 모든 데이터를 소진하기까지 $\log_2{(N)}-1$ 번의 분할이 가능한 반면, PRIM은 대략 $-\log{(N)}/\log{(1-\alpha)}$ 번의 분할이 가능하다.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
