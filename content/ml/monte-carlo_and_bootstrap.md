---
title: "Monte-Carlo & Bootstrap"
date: 2023-07-16
categories:
  - "ML"
tags:
  - "Monte-Carlo"
  - "Bootstrap"
sidebar: false
---

## Monte-Carlo Method
Monte-Carlo 방법은 random sampling을 통해 풀고자 하는 문제에 대한 numerical result를 얻는 방식이다.

예를 들어, $X\sim\text{Unif}(2, 4)$일 때, $\mathbb{E}[X]$의 analytical한 해는 다음과 같다.

$\displaystyle \mathbb{E}[X]=\int\_2^4x\cdot\dfrac{1}{2}dx=3$

만약 closed-form으로 해를 구할 수 없는 경우라면 $\text{Unif}(2, 4)$로부터의 random sampling을 통해 numerical한 해를 구할 수 있다.

```py
np.random.seed(0)
X = np.random.uniform(2, 4, size=1000)
np.mean(X) # 2.9918430687435653
```

## Bootstrap Method
Bootstrap은 데이터를 사용하여 sampling distribution을 근사하는 방식이다.

Parametric Bootstrap은 데이터의 분포는 알지만 모수는 알지 못하는 경우 모수의 추정치를 사용하여 sampling distribution을 근사한다.

```py
np.random.seed(0)
X = np.random.normal(0, 1, 1000)

mu = np.mean(X)
sigma = np.var(X)

X_pbs = np.random.normal(mu, sigma, 300)
```

Non-parametric Bootstrap은 데이터의 분포를 알지 못하는 경우 데이터를 사용한 random sampling with replacement로 sampling distribution을 근사한다.

```py
np.random.seed(0)
X_npbs = np.random.choice(X, size=300, replace=True)
```

{{<figure src="/ml/bootstrap1.png" width="900">}}
