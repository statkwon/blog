---
title: "CH5 Resampling Methods"
date: 2021-02-10
TableOfContents: true
weight: 4
---

Resampling Methods는 Training Data로부터 반복적으로 표본을 추출하고, 추출한 표본을 사용하여 여러 번 모형을 적합하여 하나의 모형을 적합할 때는 얻을 수 없는 정보를 얻기 위해 사용하는 방법이다. 5장에서는 가장 흔히 사용되는 방식인 Cross-Validation과 Bootstrap에 대한 내용을 다룬다.

## 5.1. Cross-Validation

가지고 있는 Training Data의 크기가 작아 Test Error Rate를 직접적으로 추정하기 어려운 경우, Training Data의 일부를 홀드아웃하여 Test $\text{MSE}$를 추정하는 방법에 대해 알아보도록 하자.

### 5.1.1. The Validation Set Approach

Validation Set Approach는 데이터를 Trainint Set과 Validation Set, 동일한 크기의 두 부분으로 나누어 활용하는 방식이다. 우선 Training Set을 사용하여 모형을 적합하고, 적합한 모형으로 Validation Set에 대한 결과값을 예측하여 $\text{MSE}$를 계산한다. 이러한 $\text{MSE}$를 Test $\text{MSE}$의 추정치로 사용할 수 있다.

![FIGURE 5.1](/ISLR/FIGURE_5.1.png)

하지만 이러한 방식은 두 가지 문제점을 갖는다.

![FIGURE 5.2](/ISLR/FIGURE_5.2.png)

1. 위 그림에서 볼 수 있듯이, 계산된 $\text{MSE}$가 Training Set과 Validation Set에 따라 큰 차이를 보인다. 즉, 추정치의 분산이 크다.
2. 모형을 적합하는데 전체 데이터를 사용하지 않고, Training Set만을 사용하기 때문에 모형의 성능이 상대적으로 떨어질 수밖에 없다. 경우에 따라 Overfitting이 발생할 수 있다.

앞으로 다루게 될 방법들은 이러한 두 가지 문제점들을 보완한 것들이다.

### 5.1.2. Leave-One-Out Cross Validation

LOOCV는 데이터를 Training Set과 Validation Set으로 나누어 활용한다는 점에서는 Validation Set Approach와 동일하다. 하지만 동일한 크기로 나누지 않고, Validation Set에 오직 한 개의 데이터만을 사용한다는 점에서 앞선 방식과 차이점을 갖는다. 즉, $\left\\{(x_1, y_1), \ldots, (x_n, y_n)\right\\}$의 데이터가 있을 때, 이중 임의의 $n-1$개 데이터를 Training Set으로, 나머지 한 개 데이터를 Validation Set으로 사용한다.

![FIGURE 5.3](/ISLR/FIGURE_5.3.png)

$n-1$개의 데이터를 가지고 모형을 적합하고, 한 개의 데이터에 대한 예측값을 구하여 $\text{MSE}$를 계산하는 과정을 $n$번 반복하면 총 $n$개의 $\text{MSE}$를 얻을 수 있다. LOOCV는 이 $n$개의 $\text{MSE}$의 평균을 Test $\text{MSE}$의 추정치로 사용하는 방법이다.

$$\text{CV}_{(n)}=\frac{1}{n}\sum\_{i=1}^n{\text{MSE}_i}$$

LOOCV는 모형을 적합하는데 $n-1$개의 데이터를 사용하기 때문에, Validation Set Approach에 비해 모형의 Bias를 줄일 수 있다. 또한, 반복 시행에 따른 Training Set과 Validation Set의 차이가 크지 않기 때문에, Test $\text{MSE}$의 추정치의 분산이 작다는 점에서 보다 개선된 방식이라고 할 수 있다.

하지만 모형 적합을 $n$번 해야한다는 점에서 비용적인 문제가 존재한다. 모형을 한 번 적합하는데 오랜 시간이 걸리거나, 데이터의 크기($n$)가 매우 큰 경우, LOOCV가 비효율적일 수 있음을 인지하고 있어야 한다.

### 5.1.3. $k$-Fold Cross-Validation

$k$-Fold Cross-Validation은 데이터를 동일한 크기의 $k$개 그룹으로 나누는 방식이다. 그중 한 개를 Validation Set으로, 나머지 $k-1$개 그룹을 Training Set으로 사용한다. 각 그룹을 한 번씩 Validation Set으로 사용하며 모형 적합과 예측을 총 $k$번 반복하면, $k$개의 $\text{MSE}$를 계산할 수 있다. 이 $k$개의 $\text{MSE}$의 평균을 Test $\text{MSE}$의 추정치로 사용한다.

$$\text{CV}_{(k)}=\frac{1}{k}\sum\_{i=1}^k{\text{MSE}_i}$$

LOOCV는 $k$-Fold Cross-Validation의 특수한 케이스라고 할 수 있다. $k=n$인 경우 LOOCV와 동일한 결과를 갖는다. 일반적으로는 $k=5$ 또는 $k=10$을 사용하는데, 이러한 경우 $k=n$을 사용할 때보다 수행 시간을 줄일 수 있다는 장점을 갖는다.

![FIGURE 5.5](/ISLR/FIGURE_5.5.png)

### 5.1.4. Bias-Variance Trade-Off for $k$-Fold Cross-Validation

앞서 $k<n$인 $k$-Fold Cross-Validation은 계산 시간에 있어 LOOCV에 비해 이점을 갖는다고 하였다. 이 외에도, Bias-Varinace Trade-Off의 관점에 있어서도 $k$-Fold Cross-Validation이 더 우수한 방식이라고 할 수 있다. Bias Reduction의 관점에서만 생각한다면, 데이터의 대부분을 Training에 사용하는 LOOCV가 더 낮은 Bias를 갖는 것이 사실이다. 하지만, 매번 모형을 적합할 때마다 거의 유사한 데이터가 사용된다는 점에서 Variance는 $k$-Fold Cross-Validation에 비해 높게 나타날 수밖에 없다. 이러한 맥락에서, $k=5$ 또는 $k=10$을 사용하는 것이 Bias와 Variance를 모두 낮출 수 있는 최적의 수치라는 것이 경험적으로 증명되었다.

### 5.1.5. Cross-Validation on Classification Problems

Cross-Validation은 Classification Problems에 대해서도 동일한 방식으로 적용될 수 있다. 단지 $\text{MSE}$ 대신 Error Rate를 사용한다는 점에서만 차이가 있다. 예를 들어, Classification Setting에서 LOOCV를 사용하여 추정한 Error Rate는 다음과 같은 식으로 계산된다.

$$\text{CV}_{(n)}=\frac{1}{n}\sum\_{i=1}^n{\text{Err}_i} \qquad (\text{Err}_i=I(y_i \neq \hat{y}_i))$$

## 5.2. The Bootstrap

Bootstrap에 관한 내용은 코드와 함께 보는 편이 이해하기 수월하므로 코드 관련 포스트를 참고하도록 하자.