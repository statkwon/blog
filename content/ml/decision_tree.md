---
title: "Decision Trees"
date: 2022-01-02
categories:
  - "ML"
tags:
  - "Tree"
sidebar: false
---

의사결정나무는 Feature Space를 여러 개의 영역으로 분할한 후, 각 영역의 종속변수를 어떤 상수나 범주로 예측하는 알고리즘이다. 이때 영역을 분할하는 방법으로 Recursive Binary Split을 사용한다. 의사결정나무와 관련된 여러 알고리즘이 있지만, 이 글에서는 그 중 가장 유명한 CART(Classification and Regression Tree) 알고리즘만을 다룰 것이다.

{{<figure src="/ml/dt1.png" width="300">}}

## Regression Trees

Squared Error Loss를 기준으로 하는 경우, Regression Tree의 식을 다음과 같이 표현할 수 있다.

$\displaystyle f(\mathbf{x})=\sum_{m=1}^Mc_mI(\mathbf{x}\in R_m)$, where $\hat{c}_m=\text{ave}(y_i\vert\mathbf{x}_i\in R_m)$

즉, Feature Space를 $M$개의 영역으로 나누고, 각 영역의 반응변수를 해당 영역에 속한 $y$값들의 평균으로 예측한다.

이상적인 영역 분할 기준은 최종 모형의 Sum of Squares $\sum(y_i-f(x_i))^2$를 최소화하는 방향이다. 하지만 이는 실질적으로 계산이 불가능하므로 Greedy Algorithm이라는 대안을 사용한다. Greedy Algorithm은 모든 가능한 경우에 대해서 아래의 식을 최소화하는 변수 $j$와 Split Point $s$를 반복적으로 찾는 방식이다.

$\displaystyle R\_1(j, s)=\{X\vert X\_j\leq s\} \quad\text{and}\quad R\_2(j, s)=\{X\vert X\_j&gt;s\} \quad\text{s.t.}\quad \min\_{j, s}\left[\min\_{c_1}\sum\_{\mathbf{x}\_i\in R\_1(j, s)}(y\_i-c\_1)^2+\min\_{c_2}\sum\_{\mathbf{x}\_i\in R\_2(j, s)}(y\_i-c\_2)^2\right]$

이때 분할된 두 영역 $R_1$과 $R_2$에서 각 영역의 반응변수는 $\hat{c}_1=\text{ave}(y_i\vert \mathbf{x}_i\in R_1(j, s))$와 $\hat{c}_2=\text{ave}(y_i\vert \mathbf{x}_i\in R_2(j, s))$로 예측한다.

분할을 할 수록 Tree의 크기는 커진다. 그렇다면 Tree를 얼마나 키우는 것이 적절할까? 너무 큰 Tree는 Overfitting을 발생시킬 수 있고, 너무 작은 Tree로는 데이터의 구조를 포착하기 어렵다. 따라서 적절한 크기를 결정하기 위해 Cost-Complexity Pruning이라는 방식을 사용한다. Terminal Node의 크기가 최소 노드 크기보다 작아지지 않을 때까지 키운 Tree를 $T_0$라고 하자. 이 Tree의 Internal Node를 가지치기 함으로써 Subtree인 $T\subset T_0$를 얻을 수 있다. 임의의 Tree $T$의 Terminal Node의 개수를 $\vert T\vert$라고 하자.

$\displaystyle C\_\alpha(T)=\sum\_{m=1}^{\vert T\vert}\sum\_{\mathbf{x}\_i\in R\_m}(y\_i-\hat{c}\_m)^2+\alpha\vert T\vert$, where $\hat{c}\_m=\dfrac{1}{N\_m}\sum\_{\mathbf{x}\_i\in R\_m}y\_i$

우리는 Weakest Link Pruning이라는 가지치기 규칙을 사용함으로써 각각의 $\alpha$값에 대해 $C_\alpha(T)$를 최소화하는 Subtree $T\_\alpha\subseteq T\_0$를 찾을 수 있다. Weakest Link Pruning은 $T\_0$에서 시작해서 Root Tree가 될 때까지 $\sum\_{\mathbf{x}\_i\in R\_m}(y\_i-\hat{c}\_m)^2$을 가장 적게 증가시키는 Internal Node를 순차적으로 제거하는 방식이다. 이 과정을 통해 얻어지는 Subtree들 중에 반드시 $T\_\alpha$가 존재한다. 이때 최적의 $\alpha$는 Cross Validation을 통해 추정할 수 있다. 즉, 우리의 최종 Tree는 $T\_{\hat{\alpha}}$이 된다.

## Classification Trees

$\displaystyle f(\mathbf{x})=\sum\_{m=1}^Mk(m)I(\mathbf{x}\in R\_m)$, where $k(m)=\text{argmax}\_k\hat{p}\_{mk}$

앞선 상황에서 Splitting과 Pruning의 기준이 되는 Impurity Measure만 바꾸어주면 Classification Tree를 만들 수 있다. Regression Tree에서는 Squared Error Loss를 사용하였지만, 이것은 분류 문제에 적합하지 않다. 분류 문제에 적합한 Measure로 다음과 같은 것들이 있다. 편의를 위해 $\hat{p}\_{mk}=\dfrac{1}{N_m}\sum_{\mathbf{x}_i\in R_m}I(y_i=k)$라고 하자.

- Misclassification Error: $\dfrac{1}{N_m}\sum_{i\in R_m}I(y_i\neq k(m))$
- Gini Index: $\sum\_{k\neq k'}\hat{p}\_{mk}\hat{p}\_{mk'}=\sum\_{k=1}^K\hat{p}\_{mk}(1-\hat{p}\_{mk})$
- Cross-Entropy (or Deviance): $-\sum\_{k=1}^K\hat{p}\_{mk}\log{\hat{p}\_{mk}}$

Misclassification Error는 Node 내의 불순도를 반영하지 못하기 때문에 Tree를 키울 때는 Gini Index나 Cross-Entropy를 기준으로 사용한다. 가지치기를 할 때는 세 가지 Measure를 모두 사용할 수 있지만, 주로 Misclassification Error를 사용한다.

## Splitting Categorical Predictors

순서가 정해지지 않은 $q$개의 범주를 갖는 변수에 대해 Feature Space를 분할하는 경우, 총 $2^{q-1}-1$개의 경우의 수가 존재한다. 따라서 범주의 개수가 너무 많은 경우에는 사실상 계산이 불가능하다. 하지만 Regression(Squared Error Loss 기준) 또는 Binary Classification 문제의 경우 이 계산을 단순화할 수 있다. Regression 문제에서는 범주별 반응변수의 평균에 따라 범주의 순위를 정할 수 있다. Binary Classification 문제에서는 범주별 반응변수의 값이 $1$인 경우가 차지하는 비율에 따라 범주의 순위를 정할 수 있다. 이후 해당 변수를 순서가 정해진 변수로 취급하여 분할하면 그것이 곧 최적의 분할이 된다. Multicategory Classification 문제의 경우에는 이 방식을 적용할 수 없다. 범주의 개수가 많은 경우 가능한 분할의 결과가 지수적으로 증가하기 때문에 최적의 분할을 찾는다는 관점에서 선호도가 높다. 하지만 범주의 개수가 지나치게 많은 경우에는 역으로 Overfitting이 발생할 수 있기 때문에 그러한 변수는 제외하는 것이 좋다.

## Pros and Cons of Tree Algorithm

**Pros**

1. 일반적으로 데이터에 결측치가 존재하는 경우 결측치를 제거하거나 평균 등의 값으로 대체하는 방식을 사용한다. 의사결정나무에서는 이러한 방식을 사용하지 않고도 결측치 문제를 해결할 수 있다. 결측치가 포함된 변수가 범주형 변수인 경우 결측치가 'Missing'이라는 새로운 범주에 속하는 것으로 간주할 수 있다. 보다 일반적인 방식으로는 Surrogate Variable을 활용하는 것이 있다. Feature Space를 분할할 때 결측치를 포함하지 않는 데이터만을 고려하여 최적의 변수와 Split Point를 찾는다. 그리고 최적의 분할과 가장 유사한 결과를 얻을 수 있는 Surrogate Variable과 Split Point의 배열을 구한다. 이후 전체 데이터를 사용하여 Feature Space를 분할할 때, 선택된 분할에 결측치가 포함되어 있는 경우 앞서 구한 Surrogate Split을 대신 사용한다. 이 방식은 결측치가 포함된 변수와 그렇지 않은 변수 간의 Correlation이 높을 수록 더 효과적이다.
2. Feature Space가 $X_j≤s$ 또는 $X_j&gt;s$와 같은 형태로 분할되기 때문에 충분히 해석이 가능하다. 물론 이러한 Binary Split 대신 $\sum a_jX_j≤s$와 같은 변수들의 선형 결합 형태의 분할도 가능하다. 이 방식은 모형의 예측력을 높여주지만 예측 결과를 해석하는 것이 어려워진다는 단점을 갖는다.

**Cons**

1. Tree의 Hierarchical한 구조로 인해 모형의 분산이 크다. 따라서 사용하는 데이터가 조금만 달라지더라도 Tree의 구조가 크게 바뀌게 된다. 이러한 문제를 해결하기 위해 고안된 방식으로 Bagging 등이 있다.
2. Indicator Function으로 구성된 $f(\mathbf{x})$의 식에서도 알 수 있듯이 모형의 유연성이 떨어진다.
3. Additive Structure를 반영하기 어렵다.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
