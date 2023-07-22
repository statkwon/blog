---
title: "Bayesian Statistics"
date: 2021-02-10
categories:
  - "ML"
tags:
  - "Bayesian"
sidebar: false
---

## Introduction

**Bayesian Learning**

통계적 추론은 모집단의 일부를 통해 모집단의 일반적인 특성을 알아내기 위한 과정이다. 이때 대부분의 경우 모집단의 수치적 특성을 모수 $\theta$로 표현한다. 하지만 데이터가 주어지기 전까지, 모수 $\theta$의 값은 불확실하다. $y$라는 데이터셋이 주어진다면, 이러한 모수에 대한 불확실성을 줄여나갈 수 있다. Bayesian Inference는 이러한 불확실성의 변화를 측정하는 것에 목적이 있다.

> Sample Space $\mathcal{Y}$: 가능한 모든 데이터셋들의 집합
> 
> Parameter Space $\Theta$: 가능한 모든 모수 값들의 집합
>
> Prior Distribution $p(\theta)$: $\theta$($\theta\in\Theta$)가 참값(모집단의 특성)이라는 믿음
>
> Sampling Model $p(y|\theta)$: $\theta$($\theta\in\Theta$)가 참값이라는 전제 하에 샘플링을 했을 때, 데이터셋 $y$($y\in\mathcal{Y}$)가 나올 것이라는 믿음
>
> Posterior Distribution $p(\theta|y)$: 데이터셋 $y$를 관찰한 후의 $\theta$($\theta\in\Theta$)가 참값이라는 믿음

$p(\theta|y)=\dfrac{p(y|\theta)p(\theta)}{\int_{\Theta}p(y|\tilde{\theta})p(\tilde{\theta})d\tilde{\theta}}$

위와 같은 Bayes' Rule은 우리의 믿음이 새로운 정보롤 관찰한 후 어떻게 바뀌어야 하는지 설명해준다.

## Why Bayes?

$p(\theta)$와 $p(y|\theta)$가 한 사람의 합리적인 믿음을 나타낸다면, Bayes' Rule은 새로운 정보 $y$가 주어졌을 때 $\theta$에 대한 그 사람의 믿음을 업데이트하는 최적의 방법이다. 하지만 실질적인 데이터 분석 상황에서 Prior Belief를 정확하게 수치화하는 것은 어렵기 때문에, 많은 경우 $p(\theta)$는 계산의 편의를 위한 방식으로 결정된다.

그럼에도 불구하고 Bayesian Methods를 사용할 수 있는 이유는 무엇일까?

1. "All models are wrong, but some are useful." - 설령 $p(\theta)$가 Prior Belief를 정확하게 반영하지 못하더라도, $p(\theta)$가 우리의 믿음과 비슷하다면 $p(\theta|y)$ 역시 Posterior Belief에 대한 좋은 근사치가 될 수 있다.
2. 관심사가 우리의 믿음이 아닌 경우도 있다. 데이터가 서로 다른 Prior Belief를 가진 사람들의 다양한 믿음을 어떻게 업데이트 할 것인지에 대해 관심이 있는 경우, Parameter Space의 전 영역에 걸쳐 분산되어 있는 Prior Distribution을 사용할 수 있다.
3. 많은 복잡한 통계 문제들을 해결함에 있어, Non-Bayesian 문제라고 하더라도, Bayesian Methods를 사용하는 것은 매우 효과적이다.

### Estimating the Probability of a Rare Event

한 도시의 코로나 확진자 비율 $\theta$에 관심이 있다고 하자.

관련 연구에 따르면 여러 도시들의 확진자 비율은 $0.05$에서 $0.20$ 사이이고, 평균 비율은 $0.10$이다. 따라서 우리의 Prior Belief는 $(0.05, 0.20)$에서 대부분의 확률을 갖고, 평균이 $0.10$인 분포를 따라야 한다. 이러한 조건을 만족하는 분포는 굉장히 많지만, 계산의 편의를 고려하여 $\text{beta}(2, 20)$을 사용할 것이다.

이제 모수를 추정하기 위해 $20$명의 표본을 뽑아야 한다. $20$명의 표본 중 확진자 수를 $Y$라고 하면, $\theta$가 참값이라는 전제 하에 $Y$에 대한 합리적인 Sampling Model은 $\text{binomial}(20, \theta)$이라고 할 수 있다.

Prior Distribution이 $\theta\sim\text{beta}(2, 40)$이고 Sampling Model이 $Y|\theta\sim\text{binomial}(20, \theta)$인 경우의 Posterior Distribution은 $\text{beta}(2+y, 40-y)$가 된다. (이유는 3장에서 다룬다.)

[Simulation Link](https://ysuks.shinyapps.io/Bayes/)

위 링크로 들어가면 위와 같은 조건 하에서 $0$부터 $20$ 사이의 $y$값에 따른 Posterior Distribution의 변화를 확인할 수 있다. 표본에서 관찰되는 확진자 수가 증가할 수록 $E[\theta|Y=y]$가 커지게 되는데, Posterior Distribution이 데이터셋을 관찰한 후의 $\theta$가 참값이라는 믿음이라는 점에서, 표본에서 더 많은 확진자가 발견될 수록 도시의 확진자 비율을 높게 추정하는 것은 상당히 직관적인 결과라고 할 수 있다.

이에 더해 Posterior Expectation의 식을 살펴보면

$\begin{align}
E[\theta|Y=y]&=\dfrac{a+y}{a+b+n} \\\\
&=\dfrac{n}{a+b+n}\dfrac{y}{n}+\dfrac{a+b}{a+b+n}\dfrac{a}{a+b} \\\\
&=\dfrac{n}{w+n}\bar{y}+\dfrac{w}{w+n}\theta_0
\end{align}$

Sample Mean $\bar{y}$와 Prior Expectation $\theta_0$의 가중 평균인 것을 확인할 수 있다. 즉, 더 많은 수의 표본을 뽑을 수록 표본 평균이 Posterior Expectation에 미치는 영향이 커지게 된다. 더 많은 데이터를 관찰했을 때 그 결과가 우리의 믿음의 변화에 미치는 영향이 커지는 것 역시 직관적이다.

이러한 Bayesian Methods의 장점은 표본의 갯수가 적을 때에도 충분히 활용 가능하다는 점이다. 빈도론적 관점에서는 표본 비율 $\bar{y}=\dfrac{y}{n}$을 사용하여 모비율 $\theta$를 추정한다. 이때 추정의 불확실성을 ($95$%) 신뢰구간으로 나타내는데, 일반적으로 $\bar{y} \pm 1.96\sqrt{\bar{y}(1-\bar{y}/n)}$의 Wald 신뢰 구간을 사용한다. 하지만 Wald의 신뢰구간은 $n$이 작은 경우 추정의 불확실성을 정확히 표현하지 못한다.

$\hat{\theta} \pm 1.96\sqrt{\hat{\theta}(1-\hat{\theta}/n)}$, where

$\hat{\theta}=\dfrac{n}{n+4}\bar{y}+\dfrac{4}{n+4}\dfrac{1}{2}$

이러한 문제점을 보완하기 위해 여러 대안이 제시되었고, 그중 하나가 위와 같은 Adjusted Wald 신뢰구간이다. 하지만 이러한 신뢰구간은 곧 Bayesian Inference와 밀접한 연관을 가지고 있다. 위 식에서의 $\hat{\theta}$는 Prior Distribution이 $\text{beta}(2, 2)$를 따를 때의 Posterior Expectation과 같다. 이때 $\text{beta}(2, 2)$는 평균이 $0.5$인 분포로, Prior Information이 약한 경우를 의미한다.

이외에도 앞서 보았던 것처럼 Prior Expectation은 표본의 갯수를 고려한 결과이므로 $n$이 적은 상황에서도 모수에 대한 합리적인 추정치로 사용하는 것이 가능하다.

### Building a Predictive Model

예측 모형을 만드는 경우에도 Bayesian Methods를 사용할 수 있다.

$Y_i=\beta_1x_{i, 1}+\beta_2x_{i, 2}+\cdots+\beta_64x_{i, 64}+\sigma\epsilon_i$

$64$개의 설명 변수를 사용하여 당뇨 진행에 대한 예측 모형을 만드는 상황을 생각해보자.

우선 $342$명의 환자에 대한 데이터를 트레인셋으로 사용하여 회귀 모형의 모수를 추정하고, 나머지 $100$명의 데이터를 테스트셋으로 사용하여 모형을 평가할 것이다.

이때 $65$개의 모수에 대한 정확한 Joint Prior Distribution을 찾는 것은 불가능하다. 따라서 대부분의 변수가 당뇨 진행에 영향을 미치지 않는다는 우리의 믿음을 반영하기 위해 각각의 회귀 계수가 $0$일 확률이 $50$%가 되게 하는 분포를 사용한다.

빈도론적 관점에서는 이러한 경우 OLS를 사용하여 회귀 계수를 추정한다. 하지만 OLS는 표본의 크기가 적은 경우 좋은 성능을 보이지 못한다. 위의 예시에서도 $\boldsymbol{\beta}\_{\text{OLS}}$보다 $\boldsymbol{\beta}_{\text{Bayes}}$를 사용했을 때의 Prediction Error가 더 낮게 나타났다. 이러한 문제점에 대한 대안으로 Lasso 회귀가 제시된다.

$\text{SSR}(\boldsymbol{\beta}:\lambda)=\sum\_{i=1}^n(y\_i-\mathbf{x}\_i^T\boldsymbol{\beta})^2+\lambda\sum\_{j=1}^p|\beta\_j|$

하지만 Lasso 회귀 계수는 결국 각각의 $\beta_j$에 대한 Prior Distribution이 Double-Exponential 분포를 따르는 경우의 Posterior Mode와 같다는 점에서 Bayesian Estimate와 연관되어 있다.

---

**Reference**

1. Hoff, P. D. (2009). A first course in Bayesian statistical methods (Vol. 580). New York: Springer.
