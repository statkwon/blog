---
title: "Curse of Dimensionality"
date: 2021-02-15
categories:
  - "ML"
tags:
  - "Dimension"
sidebar: false
---

차원의 저주는 고차원의 Feature Space에서 데이터를 다룰 때 발생하는 현상들을 의미한다. 이 글에서는 차원의 저주의 대표적인 세 가지 현상에 대해 소개할 것이다.

**1. Feature Space의 차원이 높아질 수록 Neighborhood의 범위가 넓어진다.**

{{<figure src="/ml/cod1.png" width="300">}}

Feature Space가 $p$차원의 Unit Hypercube이고, 그 안에 데이터가 균등하게 분포되어있는 상황을 생각해보자. 우리는 Feature Space에 속한 임의의 데이터에 대해, 전체 데이터 중 비율 $r$을 차지하는 만큼의 데이터를 Neighborhood로 사용할 것이다. 이때 Feature Space가 Unit Hypercube이므로, Neighborhood의 한 모서리의 평균 길이는 $r^{1/p}$이 된다. 아래의 그래프는 $r$을 $x$축, $r^{1/p}$을 $y$축으로 갖는 그래프이다.

{{<figure src="/ml/cod2.png" width="400">}}

2차원의 Feature Space에서 전체 데이터 중 $10\\%$의 데이터를 포함하는 Neighborhood의 한 모서리의 평균 길이는 약 $0.3$이지만, 10차원의 Feature Space에서 이러한 Neighborhood의 한 모서리의 평균 길이는 약 $0.8$이다. 전체 Feature Space(Unit Hypercube)의 한 모서리의 길이가 $1$인 것을 생각했을 때, 한 모서리의 길이가 $0.8$인 Neighborhood는 사실상 Feature Space의 전역에 걸쳐있다고 할 수 있다. 즉, Feature Space의 차원이 높아질 수록 Neighborhood가 차지하는 범위가 넓어짐으로써, 관심 데이터와의 거리가 가까운 데이터의 집합이라는 특성을 잃게 된다. 물론 $r$의 크기를 엄청나게 줄임으로써 $r^{1/p}$의 값을 줄일 수는 있지만, 이러한 경우 매우 적은 양의 데이터를 사용하여 관심 데이터에 대한 반응변수를 추정하는 것이므로 추정의 분산이 매우 커지게 되어 유의미한 해결책이 되지 못한다.

**2. Feature Space의 차원이 높아질 수록 데이터가 Feature Space의 가장자리로 쏠린다.**

{{<figure src="/ml/cod3.png" width="400">}}

이번에는 Feature Space가 $p$차원의 Unit Ball이고, 그 안에 데이터가 균등하게 분포되어있는 상황을 생각해보자. 지금부터 Unit Ball의 중심과 중심으로부터 가장 가까운 데이터 사이의 거리의 중앙값을 구하는 과정을 살펴볼 것이다. 우선 데이터가 균등하게 분포해 있으므로, 어떤 데이터가 중심으로부터 거리 $r$ 이내에 위치할 확률을 $\frac{\text{반지름이 }r\text{인 Ball의 부피}}{\text{Unit Ball의 부피}}$로 나타낼 수 있다. $p$차원 공간에서 반지름이 $r$인 Ball의 부피를 구하는 공식은 $\dfrac{\pi^{p/2}}{\Gamma\left(\frac{p}{2}+1\right)}r^p$이다. 따라서 어떤 데이터가 중심으로부터 거리 $r$ 이내에 위치할 확률은 다음와 같다.

$\text{P}(R≤r)=\dfrac{\pi^{p/2}r^p/\Gamma\left(\frac{p}{2}+1\right)}{\pi^{\frac{p}{2}}/\Gamma\left(\frac{p}{2}+1\right)}=r^p$, where $0≤r≤1$

이것을 Unit Ball의 중심과 데이터 사이의 거리에 대한 Cumulative Distribution Function $F_R(r)$로 생각한다면, 이 식을 $r$에 대하여 미분함으로써 Unit Ball의 중심과 데이터 사이의 거리에 대한 Probability Density Function $f_R(r)=pr^{p-1}$, where $0≤r≤1$을 구할 수 있다. 이때 우리가 관심을 가지고 있는 대상은 Unit Ball의 중심과 중심으로부터 가장 가까운 데이터 사이의 거리이므로, $R$에 대한 PDF를 사용하여 구한 $R$의 1st Order Statistic $R_{(1)}$의 PDF 및 CDF를 사용할 것이다.

$f_{R_{(1)}}(r)=n(1-r^p)^{n-1}pr^{p-1} \quad (0≤r≤1)$

$F_{R_{(1)}}(r)=1-(1-r^p)^n \quad (0≤r≤1)$

이제 $F_{R_{(1)}}(r)=\dfrac{1}{2}$을 만족시키는 $r$의 값을 구함으로써 Unit Ball의 중심과 중심으로부터 가장 가까운 데이터 사이의 거리의 중앙값의 식 $\text{median}(R)=\left(1-\dfrac{1}{2}^{1/n}\right)^{1/p}$을 구할 수 있다.

{{<figure src="/ml/cod4.png" width="400">}}

위와 같이 $p$를 $x$축, $\text{median}(r)$을 $y$축으로 갖는 그래프를 그려보면, 10차원의 Feature Space에서 중심과 중심과 가장 가까운 데이터 사이의 거리가 약 $0.52$로 중심보다 가장자리와의 거리가 더 가까움을 알 수 있다. 즉, Feature Space의 차원이 높아질 수록 데이터가 Feature Space의 가장자리로 쏠리는 현상이 발생하게 된다. 이러한 현상이 문제가 되는 이유는 일반적으로 가장자리에서의 Prediction이 안쪽에서의 Prediction보다 어렵기 때문이다.

**3. Feature Space의 차원이 높아질 수록 데이터의 분포가 Sparse해진다.**

{{<figure src="/ml/cod5.jpeg" width="400">}}

다시 Feature Space가 $p$차원의 Unit Hypercube이고, 그 안에 데이터가 균등하게 분포되어있는 상황을 생각해보자. 우리는 한 방향에 놓인 데이터의 개수로 데이터의 밀도를 가늠해볼 수 있다. 예를 들어, 위 그림에서 2차원 공간에서의 한 방향에 놓인 데이터의 개수는 $9^{1/2}=3$개, 3차원 공간에서의 한 방향에 놓인 데이터의 개수는 $8^{1/3}=2$개로, 3차원 공간에서의 데이터의 밀도가 2차원 공간보다 낮다고 할 수 있다. 이를 일반화하면, 데이터의 밀도가 $n^{1/p}$에 비례한다고 할 수 있다. 아래의 그래프는 $p$를 $x$축으로, $n^{1/p}$을 $y$축으로 갖는 그래프이다.

{{<figure src="/ml/cod6.png" width="400">}}

$p$가 조금만 커져도 한 방향에 놓인 데이터의 개수가 $0$에 가까워짐을 확인할 수 있다. 즉, Feature Space의 차원이 높아질 수록 데이터의 분포가 Sparse해진다.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. [https://en.wikipedia.org/wiki/Volume_of_an_n-ball](https://en.wikipedia.org/wiki/Volume_of_an_n-ball)
