---
title: "Maximum Likelihood Estimation"
date: 2023-07-17
categories:
  - "ML"
tags:
  - "Likelihood"
  - "MLE"
sidebar: false
---

MLE는 통계학에서 모수를 추정하기 위한 방법 중 하나이다. 말 그대로 likelihood를 최대화하는 추정치를 사용하여 모수를 추정하는 방식이다.

likelihood는 모수의 함수이다. 즉, $L(\theta)$는 데이터가 주어졌을 때 $f(x;\theta)$로부터 해당 데이터가 sampling 되었을 가능성을 의미한다.

{{<figure src="/ml/mle1.png" width="400">}}

따라서 데이터 $x_1, \ldots, x_n$이 동일한 분포로부터 independently sampling 되었다면, $L(\theta)=\prod\_{i=1}^nf(x\_i;\theta)$가 된다.

MLE는 다음과 같은 좋은 성질을 갖는다.

- consistency: $\hat{\theta}\_n\overset{p}{\rightarrow}\theta$
- asymptotic normality: $\sqrt{n}(\hat{\theta}\_n-\theta)\overset{d}{\rightarrow}N(0, \theta^2)$
- MLE의 분산은 Cramér-Rao lower bound와 근사적으로 같다.
