---
title: "6. Linear Model Selection and Regularization"
date: 2021-02-10
TableOfContents: true
weight: 5
---

선형 모형은 많은 전제조건을 필요로 하지만, 추론 영역에서 갖는 강점으로 인해 실생활의 문제들을 푸는데 있어 비선형 모형에 대해 경쟁력을 가지고 있다. 6장에서는 Method of Least Squares를 변형하여 선형 모형의 성능을 조금 더 개선할 수 있는 방법들에 대해 다루고 있다. 이러한 방법들은 예측 정확도와 모형의 설명력을 높이는데 기여할 수 있다.

## 6.1. Subset Selection

Subset Selection은 곧 Variable Selection을 위한 방법론이다.

### 6.1.1. Best Subset Selection

Best Subset Selection은 다음과 같은 단계를 거쳐 수행된다.

1. $\mathcal{M}_0$를 변수를 하나도 포함하지 않은 Null Model이라고 하자.
2. $k=1, 2, \ldots, p$일 때, 각각의 $k$값에 따라 전체 $p$개의 변수 중 $k$개의 변수를 사용하는 $\dbinom{p}{k}$개의 모형을 적합한다. $\dbinom{p}{k}$개의 모형 중 최고의 모형을 선택하고 $\mathcal{M}_k$라고 하자. 이때 $\text{RSS}$가 가장 작거나, $R^2$가 가장 큰 모형을 최고의 모형으로 선택한다.
3. $\mathcal{M}_0, \ldots, \mathcal{M}_p$ 중에서 최고의 모형을 선택한다. 이때 Cross-Validated Prediction Error나 $C_p$, $\text{AIC}$, $\text{BIC}$, adjusted $R^2$ 등을 기준으로 사용한다.

하지만 이러한 방식은 총 $2^p$개의 모형을 적합해야 하기 때문에 막대한 수행 시간을 필요로 한다.

### 6.1.2. Stepwise Selection

Stepwise Selection은 Best Subset Selection에 비해 시간적인 비용 소모가 적은 방법이다.

#### Forward Stepwise Selection

1. $\mathcal{M}_0$를 변수를 하나도 포함하지 않은 Null Model이라고 하자.
2. $k=0, \ldots, p-1$일 때, 각각의 $k$값에 따라 $\mathcal{M}_k$에서 한 개의 변수를 더 추가한 $p-k$개의 모형을 적합한다. $p-k$개의 모형 중 최고의 모형을 선택하고 $\mathcal{M}\_{k+1}$이라고 하자. 이때 $\text{RSS}$가 가장 작거나, $R^2$가 가장 큰 모형을 최고의 모형으로 선택한다.
3. $\mathcal{M}_0, \ldots, \mathcal{M}_p$ 중에서 최고의 모형을 선택한다. 이때 Cross-Validated Prediction Error나 $C_p$, $\text{AIC}$, $\text{BIC}$, adjusted $R^2$ 등을 기준으로 사용한다.

이러한 방식은 총 $1+p(p+1)/2$개의 모형만 적합해도 되기 때문에 수행시간을 줄일 수 있다. 하지만 변수를 제외하는 과정이 포함되어 있지 않기 때문에 항상 최고의 모형만을 선택하는 것을 보장하지 못한다는 단점이 있다. Forward Stepwise Selection은 $n<p$인 고차원의 데이터에도 사용할 수 있는 방식이다. 다만 이러한 경우, $p≥n$인 상황에 대해서는 Method of Least Squares를 사용할 수 없기 때문에 $\mathcal{M}_0, \ldots, \mathcal{M}\_{n-1}$까지의 모형만 적합할 수 있다.

#### Backward Stepwise Selection

1. $\mathcal{M}_k$를 $p$개의 모든 변수를 포함하는 Full Model이라고 하자.
2. $k=p, p-1, \ldots, 1$일 때, 각각의 $k$값에 따라 $\mathcal{M}_k$에서 한 개의 변수를 제외한 $k$개의 모형을 적합한다. $k$개의 모형 중 최고의 모형을 선택하고 $\mathcal{M}\_{k-1}$이라고 하자. 이때 $\text{RSS}$가 가장 작거나, $R^2$가 가장 큰 모형을 최고의 모형으로 선택한다.
3. $\mathcal{M}_0, \ldots, \mathcal{M}_p$ 중에서 최고의 모형을 선택한다. 이때 Cross-Validated Prediction Error나 $C_p$, $\text{AIC}$, $\text{BIC}$, adjusted $R^2$ 등을 기준으로 사용한다.

Forward Stepwise Selection과 마찬가지로 Backward Stepwise Selection은 총 $2^p$개의 모형을 적합하는 방식이다. 따라서 Best Subset Selection에 비해 시간적인 비용 소모를 줄일 수 있다. 또한 항상 최고의 모형 선택을 보장하지 못한다는 점 역시 동일하다. 하지만 Full Model을 적합할 수 있어야 하기 때문에, $n<p$인 데이터에 대해서는 사용할 수 없다는 점에서 차이를 갖는다.

#### Hybrid Approaches

Forward Selection과 Backward Elimination을 혼합하여 사용하는 것도 가능하다. 각 단계마다 변수를 추가할 뿐만 아니라, 모형을 개선하는데 더 이상 필요하지 않은 변수는 제외하는 방식이다. 이러한 접근법은 Best Subset Selection과 가장 유사한 결과를 도출함과 동시에 수행 시간을 줄일 수 있는 방식이다.

### 6.1.3. Choosing the Optimal Model

Subset Selection의 각 단계에서는 Test Error의 추정치를 기준으로 최고의 모형을 선택한다. 일반적으로 다음 두 가지 방법을 사용하여 Test Error를 추정하게 된다.

1. Overfitting으로 인한 Bias의 증가를 막게끔 변형한 Training Error를 사용하여 간접 추정하는 방식
2. Validation Set Approach나 Cross-Validation Approach를 사용하여 직접 추정하는 방식

#### $C_p$, $\text{AIC}$, $\text{BIC}$, and Adjusted $R^2$

$$C_p=\frac{1}{n}(\text{RSS}+2d\hat{\sigma}^2)$$

$C_p$ 통계량은 Training $\text{RSS}$에 $2d\hat{\sigma}^2$의 Penalty Term을 부과하여 Training Error가 Test Error를 과소추정하는 것을 방지하는 방식이다. 이 Penalty Term은 모형의 변수의 갯수가 늘어남에 따라 증가하며 Training $\text{RSS}$의 감소를 어느 정도 상쇄해주는 효과를 갖는다. $\hat{\sigma}^2$이 $\sigma^2$의 불편추정량인 경우, $C_p$는 Test $\text{MSE}$의 불편추정량이 된다.

$$\text{AIC}=\frac{1}{n\hat{\sigma}^2}(\text{RSS}+2d\hat{\sigma}^2)$$

$\text{AIC}$는 Method of Maximum Likelihood를 사용하여 적합된 모형을 위해 고안된 지표이다. $Y=\beta_0+\beta_1X_1+\cdots+\beta_pX_p+\epsilon$과 같이 Maximum Likelihood와 Least Squares가 동일한 경우, $\text{AIC}$는 위의 식과 같은 형태를 따른다.

$$\text{BIC}=\frac{1}{n}(\text{RSS}+\log{(n)}d\hat{\sigma}^2)$$

$\text{BIC}$는 $C_p$의 Penalty Term을 $\log{(n)}d\hat{\sigma}^2$로 변경한 지표이다. $n$이 $7$보다 큰 경우 항상 $\log{n}>2$가 되므로, $\text{BIC}$는 $C_p$에 비해 많은 변수를 포함하는 모형에 대해 더 많은 Penalty를 부과한다는 것을 알 수 있다.

$$\text{Adjusted } R^2=1-\frac{\text{RSS}/(n-d-1)}{\text{TSS}/(n-1)}$$

$R^2$가 변수가 추가될 때마다 계속해서 증가하는 것과는 달리, 변수를 추가함으로 인해 $\text{RSS}$가 감소하는 정도보다 $d$가 늘어나는 정도가 더 큰 경우, 전체적인 Adjusted $R^2$의 값은 감소하게 된다. 즉, 분자에 있는 $d$로 인해 모형에 불필요한 변수가 추가되는 경우 그 값이 감소한다.

{{<figure src="/islr_fig_6.2.png" width="600" height="400">}}

$C_p$, $\text{AIC}$, $\text{BIC}$는 값이 작을 수록, Adjusted $R^2$는 클 수록 더 좋은 모형임을 의미한다. 일반적으로는 $R^2$를 제외한 나머지 지표들을 위주로 사용한다.

#### Validation and Cross-Validation

앞서 언급하였듯이, 각 모형에 대한 Validation Set Error나 Cross-Validation Error를 기준으로 우수한 모형을 선정하는 방법이다. Test Error를 직접 추정할 수 있고, 모형에 대해 필요한 가정의 수가 상대적으로 적다는 점에서 장점을 갖는다. 또한 모형의 자유도를 파악하기 어렵거나, $\sigma^2$을 추정하기 어려운 경우에도 사용할 수 있다.

{{<figure src="/islr_fig_6.3.png" width="600" height="400">}}

이러한 방법을 사용할 때는 일반적으로 One-Standard-Error Rule을 적용한다. 위 그림에서 확인할 수 있듯이, Validation Set Error와 Cross-Validation Error는 변수의 갯수가 6개일 때 최소가 된다. 그러나 사실상 3개에서 11개까지는 변수의 갯수에 따른 Error의 차이가 거의 없다고 볼 수 있다. 따라서 Test $\text{MSE}$의 추정치가 Test $\text{MSE}$ 추정치의 최솟값 $\pm$ 최소 추정치의 Standard Error 사이에 있는 모형 중 변수의 갯수가 가장 적은 모형을 선택한다. 즉, One-Standard-Error Rule은 모형의 성능이 비슷한 경우, 가장 간단한 모형을 선택하는 규칙이다.

## 6.2. Shrinkage Methods

Shrinkage Methods는 계수의 추정치에 제약을 가하여 $0$에 가깝게 함으로써 분산을 줄이는 방법이다. Ridge와 Lasso 회귀가 가장 대표적인 것들이다.

### 6.2.1. Ridge Regression

$$\sum_{i=1}^n{\left(y_i-\beta_0-\sum_{j=1}^p{\beta_jx_{ij}}\right)^2}+\lambda\sum_{j=1}^p{\beta^2_j}=\text{RSS}+\lambda\sum_{j=1}^p{\beta^2_j}$$

Ridge Regression은 LSE와 유사한 방식으로 회귀계수를 추정하지만, 회귀계수에 대해 Penalty Term을 추가함으로써 LSE와는 다른 결과를 도출하는 방식이다. 위 식에서 볼 수 있듯이, 식에 포함되어 있는 $\text{RSS}$로 인해 기본적으로 데이터를 잘 적합하는 방향으로 회귀계수를 추정하게 된다. 하지만 $\lambda\sum_j{\beta^2_j}$의 Shrinkage Penalty를 추가함으로써 각각의 회귀계수 $\beta_j$가 $0$으로 줄어들게 하는 효과를 갖는다. 이때 $\lambda(\lambda>0)$는 Tuning Parameter라고 불리며, 이 두 가지 Term의 상대적인 영향력을 조절하는 역할을 한다. $\lambda=0$인 경우, Penalty Term은 아무런 영향력을 갖지 못하며, Ridge Regression은 LSE와 동일한 결과를 갖게 된다. 반면 $\lambda$가 커질 수록 Shrinkage Penalty의 영향력이 증가하며 회귀계수들이 $0$에 가까워지게 된다. 따라서 적절한 값의 $\lambda$를 사용하는 것이 매우 중요하다.

{{<figure src="/islr_fig_6.4.png" width="600" height="400">}}

좌측의 그래프에서는 $\lambda$값이 커짐에 따라 회귀계수들이 점점 $0$에 가까워지는 것을 확인할 수 있다. 반면 우측의 그래프에서는 $x$축에 $\lambda$ 대신 $||\hat{\beta}^R_\lambda||_2/||\hat{\beta}||_2$를 사용하고 있다. 이때 $||\beta||_2$를 $\ell_2$ Norm이라고 하고, $||\beta||_2=\sqrt{\sum_{j=1}^p{\beta^2_j}}$로 정의한다. 따라서 $\lambda=0$인 경우에는 Ridge 회귀계수의 추정치와 LSE가 동일하므로 $1$의 값을, $\lambda=\infty$인 경우에는 모든 Ridge 회귀계수의 추정치가 $0$이 되므로 $0$의 값을 갖게 된다. 즉, $||\hat{\beta}^R_\lambda||_2/||\hat{\beta}||_2$는 Ridge 회귀계수의 추정치가 $0$으로 줄어든 정도를 의미한다고 볼 수 있다. $0$에 가까울 수록 많이 줄어든 것이므로, 그래프의 좌측으로 갈 수록 회귀계수들이 $0$에 가까워진다.

$$\tilde{x}_{ij}=\frac{x\_{ij}}{\sqrt{\frac{1}{n}\sum\_{i=1}^n{(x\_{ij}-\bar{x}_j)^2}}}$$

$X_j\hat{\beta}^R_{j, \lambda}$는 $\lambda$뿐만 아니라 변수들의 Scale에도 큰 영향을 받는다. 따라서 Ridge Regression을 적합하기 전에 위의 식과 같은 Standard Sclaer를 사용하여 변수들의 Scale을 동일하게 맞추는 것이 좋다.

#### Why Does Ridge Regression Improve Over Least Squares?

LSE와 비교했을 때 Ridge Regression이 갖는 장점은 Bias-Variance Trade-Off에 근거하고 있다. $\lambda$가 커짐에 따라 Ridge Regression의 유연성이 감소하며 분산이 줄어드는 대신 Bias가 증가한다. 일반적으로 독립변수와 종속변수 사이에 선형 관계가 존재하는 경우, LSE는 Low Bias와 High Variance를 갖는다. 따라서 변수의 갯수 $p$가 데이터의 갯수 $n$만큼 많은 경우, LSE의 분산은 매우 커지게 된다. 더욱이 $p>n$이면 유일해를 갖지 않기 때문에 LSE를 사용할 수가 없다. 반면 Ridge Regression은 Bias를 약간 늘리는 대신 Variance를 대폭 감소시키기 때문에 이러한 경우에도 사용이 가능하다. 즉, Ridge Regression은 LSE가 High Variance를 갖는 경우에 유용하게 쓰일 수 있다. Best Subset Selection과 비교했을 때에도 수행 시간에 있어 강점을 갖는다. $2^p$개의 모형을 적합해야 하는 Best Subset Selection과 달리, Ridge Regression은 하나의 모형만을 적합하는 것으로 충분하다.

### 6.2.2. The Lasso

위와 같은 장점에도 불구하고, Ridge Regression은 항상 최종 모형으로 $p$개의 모든 변수를 포함하는 모형을 선택한다는 단점을 가지고 있다. Ridge Regression의 Penalty Term $\lambda\sum{\beta^2_j}$는 회귀계수들을 $0$에 가깝게 만들지만, 정확히 $0$이 되게 하지는 않는다. 이는 예측 정확도에 있어서는 문제가 되지 않지만, 변수의 갯수가 많은 경우에 모형 해석을 어렵게 한다.

Lasso Regression은 이러한 문제점에 대한 대안이 될 수 있다.

$$\sum_{i=1}^n{\left(y_i-\beta_0-\sum_{j=1}^p{\beta_jx_{ij}}\right)^2}+\lambda\sum_{j=1}^p{|\beta_j|}=\text{RSS}+\lambda\sum_{j=1}^p{|\beta_j|}$$

Lasso Regression은 $\ell_2$ Penalty 대신 $\lambda\sum_j{|\beta_j|}$의 $\ell_1$ Penalty를 사용한다. 이때 $\ell_1$ Norm은 $\sum{|\beta_j|}$로 정의된다. $\ell_1$ Penalty는 회귀계수의 추정치가 정확히 $0$이 되게 하는 효과를 갖는다. 따라서 Lasso Regression은 Best Subset Selection과 같은 Variable Selection 기능을 수행한다. 결과적으로 Lasso Regression을 사용하여 적합한 모형은 Ridge Regression에 의한 모형보다 해석하기가 수월하다. Lasso Regression 역시 적절한 $\lambda$값을 사용하는 것이 중요하다.

{{<figure src="/islr_fig_6.6.png" width="600" height="400">}}

위 그림에서 $\lambda$가 일정 값을 넘어서면 회귀계수의 추정치가 정확히 $0$이 되는 것을 확인할 수 있다.

#### Another Formulation for Ridge Regression and the Lasso

Ridge와 Lasso Regression의 회귀계수를 추정하기 위한 식을 각각 아래와 같은 방식으로 표현해볼 수 있다.

$$\underset{\beta}{\text{minimize}}\left\\{\sum_{i=1}^n{\left(y_i-\beta_0-\sum_{j=1}^p{\beta_jx_{ij}}\right)^2}\right\\} \quad \text{subject to} \quad \sum_{j=1}^p{|\beta_j|}≤s$$

$$\underset{\beta}{\text{minimize}}\left\\{\sum_{i=1}^n{\left(y_i-\beta_0-\sum_{j=1}^p{\beta_jx_{ij}}\right)^2}\right\\} \quad \text{subject to} \quad \sum_{j=1}^p{\beta^2_j}≤s$$

$p=2$인 경우라면, Ridge Regression은 $\beta^2_1+\beta^2_2≤s$를 만족하는 값들 중에서 $\text{RSS}$를 최소화하는 값으로 회귀계수를 추정할 것이다. 마찬가지로, Lasso Regression은 $|\beta_1|+|\beta_2|≤s$를 만족하는 값들 중에서 $\text{RSS}$를 최소화하는 값으로 회귀계수를 추정할 것이다.

Best Subset Selection 역시 이러한 패턴으로 나타낼 수 있다.

$$\underset{\beta}{\text{minimize}}\left\\{\sum_{i=1}^n{\left(y_i-\beta_0-\sum_{j=1}^p{\beta_jx_{ij}}\right)^2}\right\\} \quad \text{subject to} \quad \sum_{j=1}^p{I(\beta_j \neq 0)}≤s$$

각 방식의 제약 조건의 식을 비교해봄으로써 Ridge와 Lasso Regression이 Best Subset Selection에 비해 계산이 용이함을 다시 한 번 확인할 수 있다.

#### The Variable Selection Property of the Lasso

Ridge Regression과 달리 Lasso Regression이 회귀계수를 정확히 $0$으로 추정하게 되는 이유를 그래프를 통해 살펴보도록 하자.

{{<figure src="/islr_fig_6.7.png" width="600" height="400">}}

Ridge Regression의 제약 조건 $\beta^2_1+\beta^2_2≤s$는 2차원에서 원의 형태를 갖는다. 반면, Lasso Regression의 조건식 $|\beta_1|+|\beta_2|≤s$는 마름모 형태이다. 마름모와 달리 원에는 Sharp Points가 존재하지 않기 때문에, $\hat{\beta}$의 타원 영역과 정확히 축 위에서 만나는 것이 불가능하다. 마름모의 경우 축 위에 모서리가 위치하고 있기 때문에, $\hat{\beta}$의 타원과 각각의 모서리에서 만나게 된다. 이러한 성질은 보다 고차원의 경우에 있어서도 동일하게 적용된다.

#### Comparing the Lasso and Ridge Regression

그렇다면 Ridge와 Lasso 중 어느 한 쪽이 더 우월한 모형이라고 판단할 수 있을까? 답은 '불가능'이다. 변수의 갯수가 적고, 각 계수의 크기가 균일하지 않은 경우에는 Lasso Regression을 사용하는 것이 좋다. 하지만 변수의 갯수가 많고, 각 계수의 크기가 균일한 경우에는 Ridge Regression을 사용하는 것이 적합하다. 물론 일반적인 경우 모형에 포함되는 변수의 갯수를 정확하게 알지 못하므로, Cross-Validation 등의 방식을 사용하여 해당 데이터에 어느 방식이 더욱 적합한지 판단하게 된다.

### 6.2.3. Selecting the Tuning Parameter

Ridge와 Lasso Regression의 $\lambda$값을 결정할 때는 각각의 $\lambda$에 대한 Cross-Validation Error를 계산하여 가장 낮은 Error를 발생시키는 값을 선택하면 된다.

## 6.3. Dimension Reduction Methods

지금까지는 독립변수를 그대로 사용하는 방법들에 대해 살펴보았지만, 6.3장에서는 독립변수를 변형하여 Least Squares 모형을 적합하는 방식에 대해 다루고 있다. 이러한 방식을 Dimension Reduction Methods라고 한다.

$M<p$에 대해 $Z_1, Z_2, \ldots, Z_M$이 $p$개의 기존 변수들의 선형 결합을 나타낸다고 하자. 즉, 상수 $\phi_{1m}, \phi_{2m}, \ldots, \phi_{pm}$과 $m=1, \ldots, M$에 대해 $Z_m=\sum_{j=1}^p{\phi_{jm}X_j}$으로 나타내는 것이다. 이렇게 변형한 새로운 변수들을 가지고 선형 회귀 모형을 적합할 수 있다.

$$y_i=\theta_0+\sum_{m=1}^M{\theta_m}z_{im}+\epsilon_i \qquad i=1, \ldots, n$$

$\phi_{1m}, \phi_{2m}, \ldots, \phi_{pm}$로 적절한 값을 사용한다면, 이러한 Dimension Reduction Approach가 일반적인 회귀 모형보다 더 좋은 성능을 발휘할 수 있다. 이때 $\beta_0, \beta_1, \ldots, \beta_p$의 $p+1$개 변수에 대한 문제를 $\theta_0, \theta_1, \ldots, \theta_M$의 $M+1$개 변수에 대한 문제로 축소하는 것이므로, 이를 차원 축소라고 한다.

$$\sum_{m=1}^M{\theta_mz_{im}}=\sum_{m=1}^M{\theta_m}\sum_{j=1}^p{\phi_{jm}x_{ij}}=\sum_{j=1}^p\sum_{m=1}^M{\theta_m\phi_{jm}x_{ij}}=\sum_{j=1}^p{\beta_jx_{ij}}$$

$$\beta_j=\sum_{m=1}^M{\theta_m\phi_{jm}}$$

위의 식에서 볼 수 있듯이, $\beta_j$를 $\sum{\theta_m\phi_{jm}}$으로 나타낼 수 있다. 즉, Dimension Reduction은 $\beta_j$에 제약을 가하는 방식이라고 할 수 있다. 데이터의 갯수에 비해 변수의 갯수가 많은 경우, $p$차원의 문제를 $M$차원의 문제로 축소함으로써 분산을 줄일 수 있다.

모든 Dimension Reduction Methods는 두 가지 단계로 이루어진다. 우선 $Z_1, Z_2, \ldots, Z_m$과 같이 변형된 변수를 생성한다. 다음으로, 이 $M$개의 변수들을 사용하여 모형을 적합한다. 이때 $Z_1, Z_2, \ldots, Z_m$을 생성하는 방법으로 여러 가지가 존재한다. 6.3장에서는 그중 Principal Components와 Partial Least Squares에 대해 다룬다.

### 6.3.1. Principal Components Regression

(작성중)

## 6.4. Considerations in High Dimensions

### 6.4.1. High-Dimensional Data

대부분의 전통적인 통계 모형들은 변수의 갯수에 비해 데이터의 갯수가 훨씬 많은 Low-Dimensional Setting에 초점이 맞춰져 있다. 하지만 기술이 발전함에 따라 수집할 수 있는 변수의 갯수에 대한 제약이 줄어들면서 데이터의 갯수보다 변수의 갯수가 더 많은 상황이 빈번해졌다. 이를 High-Dimensional이라고 부르며, Least Squares 선형 회귀와 같은 고전적인 방법론들은  이러한 경우에 사용이 적합하지 않다.

### 6.4.2. What Goes Wrong in High Dimensions?

그렇다면 어떠한 점에서 $p>n$의 데이터에 기존의 방법론을 적용해서는 안되는 것일까? 아래의 그림을 통해 그 이유를 쉽게 이해할 수 있다.

{{<figure src="/islr_fig_6.22.png" width="600" height="400">}}

일반적인 선형 회귀 모형을 생각해보자. 데이터의 갯수보다 변수의 갯수가 많은 경우에 대해 선형 회귀 모형을 적합하면 변수 간의 실제 관계와는 무관하게 항상 Overfitting된 결과를 얻게될 것이다. 오른쪽 그림에서 볼 수 있듯이, 데이터가 두 개이고 변수가 두 개인 상황에서 회귀선을 적합하게 될 경우, 주어진 데이터에 꼭 맞는 선을 얻을 수 있다. 이때 Training $\text{MSE}$는 정확하게 $0$이 된다. 하지만 좌측의 더 많은 데이터를 사용하여 적합한 회귀선과 비교할 경우, 이러한 회귀선이 상당히 Overfitting 된 것임을 확인할 수 있다.

### 6.4.3. Regression in High Dimensions

따라서 High-Dimensional Setting에 대해서는 모형 선정에 더욱 신경을 써야 할 필요가 있다. 예를 들어, 앞서 배웠던 Stepwise Selection, Ridge Regression, Lasso Regression, PCA 등은 High-Dimensional Setting에 적용하기 유용한 방법들이다. 이러한 방법들을 사용하는 경우, 기본적인 Least Squares 방식보다 덜 유연한 모형을 적합함으로써 모형의 Overfitting을 피할 수 있다.