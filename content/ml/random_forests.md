---
title: "Random Forests"
date: 2022-01-15
categories:
  - "ML"
tags:
  - "Tree"
  - "Bagging"
  - "Random Forests"
sidebar: false
---

Bagging: 추정량의 분산을 줄이기 위한 방법 &rarr; DT와 같은 고분산 & 저편향 모형에 잘 작동함

Boosting: Committe of Weak Learners

Random Forests: Bagging의 수정 버전, De-Correlated Tree들의 평균

## Random Forests

Bagging에서 만들어지는 모든 tree들은 identically distributed &rarr; bias가 같음 &rarr; 분산을 줄이는 것만 가능

분산이 $\sigma^2$인 $B$개의 i.i.d. 확률변수들의 평균의 분산은 $\dfrac{1}{B}\sigma^2$이지만, pairwise correlation이 $\rho$인 i.d. $B$개의 확률변수들의 평균의 분산은 $\rho\sigma^2+\dfrac{1-\rho}{B}\sigma^2$이다.

{{<rawhtml>}}
<details><br>
<summary>Proof</summary>
$\begin{aligned}
\text{Var}\left(\dfrac{\sum_{i=1}^BX_i}{B}\right)&=\dfrac{1}{B^2}\sum_{i=1}^B\text{Var}(X_i)+\dfrac{1}{B^2}\sum_{i\neq j}^B\text{Cov}(X_i, X_j) \\
&=\dfrac{1}{B^2}\cdot B\sigma^2+\dfrac{1}{B^2}\cdot B(B-1)\sigma^2\rho \\
&=\dfrac{\sigma^2}{B}+\dfrac{B-1}{B}\sigma^2\rho \\
&=\rho\sigma^2+\dfrac{1-\rho}{B}\sigma^2
\end{aligned}$
</details><br>
{{</rawhtml>}}

$\rho$가 Bagging의 성능 개선(분산 감소)에 걸림돌이 됨 &rarr; 'Tree들 간의 correlation을 줄여서 성능을 개선해보자!'는 것이 Random Forest의 아이디어

**Algorithm**

Training set: $Z=(z\_1, z\_2, \ldots, z\_N)$ where $z\_i=(x\_i, y\_i)$
1. $B$개의 Bootstrap 데이터셋을 생성한다.
2. 각각의 $b$번째 Bootstrap 데이터셋에 대하여 Tree($T\_b$)를 적합한다. 이때 Tree를 키움에 있어 각 노드를 분할하기 전에 $m$(Regression $p/3$, Classification $\sqrt{p}$ 추천)개 변수를 랜덤하게 선택하고, 선택된 변수들만 사용하여 최적의 분할점을 찾는다. Tree가 최소 노드 크기($n\_\text{min}$, Regression $5$, Classification $1$ 추천)를 만족시킬 때까지 분할한다.
3. $B$개의 Tree들($\\{T\_b\\}\_1^B$)에 ensemble 기법을 적용한다.
    1. Regression: $\displaystyle
\hat{f}\_\text{rf}^B(x)=\dfrac{1}{B}\sum\_{b=1}^BT(x;\Theta\_b)$, where $\Theta\_b$ is the $b$th RF tree.
    2. Classification: $\hat{C}\_\text{rf}^B(x)=\text{majority vote}\;\\{\hat{C}(x;\Theta_b)\\}\_1^B$

직관적으로 $m$을 줄이면 ensemble에 사용된 tree들 간의 correlation이 줄어듦 &rarr; 분산이 줄어듦

이러한 방식이 모든 모형에서 잘 작동하는 것은 아님 &rarr; tree와 같이 비선형적인 모형에서 가장 효과적

RF와 다르게, Bagging은 linear statistics(ex. $\bar{x}$)의 분산을 일정 수준 이상 줄이지 못함

## Details of Random Forests

### OOB Sample

RF는 Bootstrap 기법을 활용하기 때문에 각각의 tree($T\_b$)를 적합할 때 사용되지 않는 데이터가 존재한다. 이러한 $z\_i=(x\_i, y\_i)$가 사용되지 않은 tree들에만 ensemble 모형을 적합하고, $z\_i$를 test set으로 사용하여 해당 ensemble 모형으로 test error를 계산한다. (이렇게 구한 OOB error는 $N$-fold CV error와 거의 동일) OOB error가 안정화되었을 때 모형의 학습을 종료한다.

### Variable Importance

RF도 Gradient Boosting과 같이 split criterion에 대한 개선도를 기준으로 변수 중요도를 구할 수 있다. But, RF만의 차별화된 방식이 존재한다.
1. 각각의 $b$번째 Tree에 대한 OOB error를 계산한다.
2. 동일한 상황에 대하여 $j$번째 변수에 대한 값으로 임의의 상수를 사용했을 때의 OOB error를 계산한다.
3. $B$개의 두 Error의 차이에 대한 평균, 표준편차를 구하고, 평균/표준편차를 $j$번째 변수의 중요도로 사용한다.

### Proximity Plots

{{<figure src="/ml/rf1.png" width="600">}}

RF를 적합할 때 각각의 tree에 대해 데이터가 같은 terminal node에 속할 때마다 해당 데이터 사이의 proximity를 한 단위씩 증가시키고, 그 결과를 $N\times N$ proximity matrix로 저장한 것을 MDS를 사용하여 2차원으로 표현한 것

&rarr; 기존 데이터의 차원이 높더라도 proximity plot을 사용하여 데이터 사이의 거리를 가늠할 수 있음

기존 feature space 상의 pure region에 속한 데이터일수록 proximity plot에서 끝 쪽에 위치한다.

## Overfitting

전체 변수의 개수는 많지만 유효한 변수들의 개수는 적은 경우, RF는 노드를 분할하기 위해 선택하는 변수의 개수($m$)가 작은 경우에 대하여 좋지 않은 성능을 보인다. (유효한 변수들이 선택될 확률이 줄어들기 때문)

사용하는 Tree의 개수($B$)를 늘리는 것은 overfitting을 일으키지 않는다.

$\displaystyle \hat{f}\_\text{rf}(\mathbf{x})=\text{E}\_\Theta[T(x;\Theta)]=\lim\_{B\rightarrow\infty}\hat{f}(x)\_\text{rf}^B$

RF는 $\text{E}\_\Theta[T(x;\Theta)]$로 $f$를 추정하는 알고리즘이지만, 우리가 이 평균값을 정확하게 계산하는 것은 불가능하기 때문에 앞서 보았던 것과 같이 $\displaystyle
\hat{f}\_\text{rf}^B(x)=\dfrac{1}{B}\sum\_{b=1}^BT(x;\Theta\_b)$와 같은 방식으로 평균값을 근사하는 것이다. 따라서 $B$를 키우는 것은 overfitting을 일으키지 않는다.

하지만 이것이 RF가 결코 overfitting 되지 않음을 의미하는 것은 아니다. 우리가 어떤 확률변수 $X$의 평균을 알고싶은 경우 $X$의 분포로부터 $n$개의 표본을 임의로 추출하여 해당 표본들의 평균으로 $X$의 평균을 추정하는 것처럼, RF 역시 $\Theta$에 대한 평균을 구하는 것이므로 $\Theta$의 분포로부터 $B$개의 표본을 임의로 추출하여 해당 표본들의 평균으로 $\text{E}_\Theta[T(\mathbf{x};\Theta)]$를 추정한다. 이때 $\Theta$의 분포가 training data에 의존하기 때문에 RF는 overfitting 될 수 있다.

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
