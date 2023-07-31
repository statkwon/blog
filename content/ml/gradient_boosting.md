---
title: "Gradient Boosting"
date: 2022-01-13
categories:
  - "ML"
tags:
  - "Tree"
  - "Boosting"
sidebar: false
draft: true
---

## Optimization Problem for Trees

$\displaystyle T(x;\Theta)=\sum\_{j=1}^J\gamma\_jI(x\in R\_j)$, where $\Theta=\\{R\_j, \gamma\_j\\}\_1^J$

tree는 feature space를 $J$개의 disjoint한 영역 $R\_j$로 분할한 뒤, 각 영역마다 부여된 값인 $\gamma\_j$를 사용하여 예측을 하는 방식으로 동작한다.

하지만 다음과 같은 loss function을 최소화하는 parameter $\Theta$를 찾는 것은 쉽지 않다.

$\displaystyle \hat{\Theta}=\underset{\Theta}{\text{argmin}}\sum\_{j=1}^J\sum\_{x\_i\in R\_j}L(y\_i, \gamma\_j)$

따라서 최적화 문제를 다음과 같은 두 파트로 나누어 풀게 된다.

1. $R\_j$를 안다는 전제 하에 $\gamma\_j$를 찾는다.
2. $R\_j$를 찾는다.

1번은 쉽다. regression 문제의 경우 $\hat{\gamma}\_j=\bar{y}\_j$, classification 문제의 경우 $\hat{\gamma\_j}=\underset{k}{\text{argmax}}\sum\_iI(y_i=k)$가 된다. \
2번은 어렵다. 따라서 [Decision Trees](/ml/decision_trees)에서 언급한 것과 같이 greedy algorithm을 사용하여 근사해를 구한다.

## Gradient Boosting

AdaBoost를 적합하기 위해서는 매 step마다 loss를 최소화하는 tree를 찾아야 한다. 그 전에 다음과 같은 문제를 생각해볼 수 있다.

우리의 목표는 loss function $\displaystyle L(f)=\sum\_{i=1}^NL(y\_i, f(x\_i))$를 최소화하는 $f$를 찾는 것이다. 이 문제를 아래와 같은 수치 해석의 관점으로 바라보고

$\hat{\mathbf{f}}=\underset{\mathbf{f}}{\text{argmin}}\\;L(\mathbf{f})$, where $\mathbf{f}=\begin{bmatrix} f(x\_1) & f(x\_2) & \cdots & f(x\_N) \end{bmatrix}^T$

$\mathbf{f}\_m=\mathbf{f}\_{m-1}-\rho\_m\nabla L(\mathbf{f}\_{m-1})$과 같은 Gradient Descent 방식을 사용하여 최적화 할 수 있다.

결과적으로 최적해는 $\mathbf{f}\_M=\mathbf{f}\_0-\rho\_1\nabla L(\mathbf{f}\_0)-\rho\_2\nabla L(\mathbf{f}\_1)-\cdots-\rho\_M\nabla L(\mathbf{f}\_{M-1})$과 같은, 매 step마다 모델에 negative gradient를 더해가는 형태가 된다.

이는 AdaBoost가 매 step마다 loss를 최소화하는 tree를 찾아 모델에 더해가는 것과 유사하므로, $\mathbf{t}\_m=\begin{bmatrix} T(x_1;\Theta\_m) & T(x_2;\Theta\_m) & \cdots & T(x_N;\Theta\_m) \end{bmatrix}^T$가 negative gradient인 $-\rho\_m\nabla L(\mathbf{f}\_{m-1})$과 가까워지도록 최적화하는 방식을 고려해볼 수 있다.

즉, $\displaystyle \tilde{\Theta}\_m=\underset{\Theta}{\text{argmin}}\sum\_{i=1}^N(-g\_{im}-T(x\_i;\Theta))^2$이 된다.

$\mathbf{f}\_0=\mathbf{h}\_0$, $\mathbf{h}\_m=-\rho\_m\nabla L(\mathbf{f}\_{m-1})$이라고 하면, $\mathbf{f}\_M=\mathbf{h}\_0+\mathbf{h}\_1+\cdots+\mathbf{h}\_M$으로 쓸 수 있다.

이때 squared loss를 사용하는 regression 문제의 경우 $-\rho\_m\nabla L(\mathbf{f}\_{m-1})$이 $m-1$ 번째 step에서의 모델의 residual과 같으므로 $\mathbf{h}\_m$이 이 residual을 타겟으로 적합한 모형으로 볼 수 있다. 즉, 매 step마다 이전 step의 residual을 타겟으로 모델을 적합하여 더해나가는 방식이 된다.

**Algorithm**
1. Initialize $f\_0(x)=\underset{\gamma}{\text{argmin}}\sum\_{i=1}^NL(y\_i, \gamma)$.
2. For $m=1$ to $M$:
    1. For $i=1, 2, \ldots, N$ compute $r\_{im}=-\left[\dfrac{\partial L(y\_i, f(x\_i))}{\partial f(x\_i)}\right]\_{f=f\_{m-1}}$.
    2. Fit a regression tree to the targets $r\_{im}$ giving terminal regions $R\_{jm}$, $j=1, 2, \ldots, J\_m$.
    3. For $j=1, 2, \ldots, J\_m$ compute $\displaystyle \gamma\_{jm}=\underset{\gamma}{\text{argmin}}\sum\_{x\_i\in R\_{jm}}L(y\_i, f\_{m-1}(x\_i)+\gamma)$.
    4. Update $f\_m(x)=f\_{m-1}(x)+\sum\_{j=1}^{J\_m}\gamma\_{jm}I(x\in R\_{jm})$.
3. Output $\hat{f}(x)=f\_M(x)$.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. [https://convex-optimization-for-all.github.io/contents/chapter06/2021/03/20/06_04_gradient_boosting/](https://convex-optimization-for-all.github.io/contents/chapter06/2021/03/20/06_04_gradient_boosting/)
3. [https://youtu.be/d6nRgztYWQM](https://youtu.be/d6nRgztYWQM)
