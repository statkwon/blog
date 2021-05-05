---
title: "ISLR CH2"
date: 2021-02-10
draft: false
TableOfContents: true
---

## 3.1. Simple Linear Regression

![FIGURE 3.1](/ISLR/FIGURE_3.1.png)

Simple Linear Regression은 변수 간의 선형 관계를 잘 나타내는 최적의 직선 $\beta_0+\beta_1X$를 찾는 작업이다. 직선의 식을 아는 경우, 새로운 데이터에 대한 예측을 할 수 있다. 따라서 우리는 Training Data를 사용하여 회귀계수 $\beta_0$, $\beta_1$을 추정할 것이다.

### 3.1.1. Estimating the Coefficients

가장 일반적인 방법은 Method of Least Squares이다. ISLR에 나와있는 LSE의 정의는 잘못되었다. 책에서는 $e_i=Y_i-\hat{Y}_i$, 즉 데이터가 추정된 회귀선으로부터 떨어진 정도를 나타내는 Residual의 제곱합을 최소화하는 추정량이 LSE라고 설명하고 있다. 하지만 LSE는 Residual이 아닌, Error의 제곱합을 최소화하는 추정량이다. 이때 Error는 $\epsilon_i=y_i-\beta_0-\beta_1x_i$로, 데이터가 모회귀선으로부터 떨어진 정도를 의미한다. 따라서 아래와 같이 표기를 수정해야 한다.

$$\text{SSE}=\sum_{i=1}^n(y_i-\beta_0-\beta_1x_i)^2$$

LSE를 사용하여 추정한 회귀 계수는 다음과 같다.

$$\hat{\beta}\_1=\frac{\sum{\left(X_i-\bar{X}\right)\left(Y_i-\bar{Y}\right)}}{\sum{\left(X_i-\bar{X}\right)^2}}$$

$$\hat{\beta}_0=\bar{Y}-\hat{\beta}_1\bar{X}$$

이와 관련된 증명이 궁금한 경우에는 [1. Linear Regression with One Predictor Variable](/Regression_1/)를 참고하자.

### 3.1.2. Assessing the Accuracy of the Coefficient Estimates

Method of Least Squares를 사용하여 추정한 회귀계수는 얼마나 정확한 추정치일까?

$$E(\hat{\beta}_0)=\beta_0 \qquad E(\hat{\beta}_1)=\beta_1$$

우선 LSE는 불편추정량이다. 불편추정량이라 함은, 구조적으로 추정치가 모수의 참값을 과대 추정하지도, 과소 추정하지도 않는다는 의미이다.

또한 LSE는 모든 불편 선형 추정량 중 분산이 가장 작은 추정량이다. $\hat{\beta}_0$과 $\hat{\beta}_1$의 분산은 아래와 같다.

$$\text{Var}(\hat{\beta}_0)=\sigma^2\left[\frac{1}{n}+\frac{\bar{X}^2}{\sum{\left(X_i-\bar{X}\right)^2}}\right] \qquad \text{Var}(\hat{\beta}_1)=\frac{\sigma^2}{\sum{(X_i-\bar{X})^2}}$$

일반적으로 $\sigma^2$의 참값을 알지 못하므로, $\sigma^2$의 불편추정량인 $\text{MSE}=\frac{\text{SSE}}{n-2}$를 대신 사용한다.

$$s(\hat{\beta}_0)=\text{MSE}\left[\frac{1}{n}+\frac{\bar{X}^2}{\sum{\left(X_i-\bar{X}\right)^2}}\right] \qquad s(\hat{\beta}_1)=\frac{\text{MSE}}{\sum{(X_i-\bar{X})^2}}$$

$\hat{\beta}_0$과 $\hat{\beta}_1$의 분포를 사용하여 신뢰구간을 구하거나 가설검정을 할 수 있다. 대부분의 경우 $\beta_1$에 대한 추론에만 관심이 있기 때문에 $\beta_0$에 관한 내용을 생략하도록 하겠다.

$\beta_1$에 대한 $95$% 신뢰구간은 다음과 같다. 아래의 식이 정확한 형태이고, 책에서는 $t_{0.025(0.975) ; n-2}$를 대략적인 값인 $2$로 표기하고 있다.

$$\left[\hat{\beta}_1-t\_{0.025 ; n-2} \times s(\hat{\beta}_1), \\ \hat{\beta}_1+t\_{0.975 ; n-2} \times s(\hat{\beta}_1)\right]$$



$\beta_1$에 대한 가설 검정의 경우, 다음과 같은 형태가 가장 많이 사용된다.

$$H_0:\beta_1=0 \quad \text{vs.} \quad H_1:\beta_1 \neq 0$$

귀무가설의 기각 여부를 판단하기 위해서 $t=\frac{\hat{\beta}_1-0}{s(\hat{\beta}_1)}$를 검정통계량으로 사용한다.

마찬가지로 위 내용과 관련된 증명들이 궁금하다면 [1. Linear Regression with One Predictor Variable](/1.-linear-regression-with-one-predictor-variable/)를 참고하자.

### 3.1.3. Assessing the Accuracy of the Model

이번에는 모형의 성능을 확인할 수 있는 지표들에 대해 알아보도록 하자.

**Mean Squarred Error**

$$\text{MSE}=\frac{\sum{\left(Y_i-\bar{Y}\right)^2}}{n-2}$$

$\text{MSE}$는 데이터가 추정된 회귀선으로부터 떨어져 있는 평균적인 정도를 의미한다. 예측값이 실제값과 유사한 경우 $\text{MSE}$가 작을 것이고, 실제값과 차이가 큰 경우에는 $\text{MSE}$가 크게 나타날 것이다.

**$\bf{R^2}$ Statistic**

$$R^2=\frac{\text{SSTO}-\text{SSE}}{\text{SSTO}}=1-\frac{\text{SSE}}{\text{SSTO}}$$

$\text{SSTO}$는 $\sum(y_i-\bar{y})^2$으로, $Y$의 전체 분산을 의미한다. 반면 $\text{SSE}$는 Regression을 수행한 이후에도 설명되지 못한 분산의 양을 의미한다. 따라서 $\text{SSTO}-\text{SSE}$은 Regression을 수행함으로써 설명된 분산의 양이 되고, 결과적으로 $R^2$는 $X$로 설명되지 못하는 $Y$의 분산의 비율을 나타내는 척도가 된다. $R^2$는 $X$와 $Y$의 상관계수를 제곱한 것과 같다.

## 3.2. Multiple Linear Regression

### 3.2.1. Estimating the Regression Coefficients

Multiple Linear Regression은 독립변수의 개수가 여러 개인 경우 사용할 수 있는 모형이다. 모형식은 다음과 같다.

$$Y=\beta_0+\beta_1X_1+\beta_2X_2+...+\beta_pX_p+\epsilon$$

Simple LInear Regression에서와 마찬가지로, 우리는 실제 회귀계수를 모르기 때문에, LSE를 사용하여 회귀계수를 추정한다.

$$\hat{Y}=\hat{\beta}_0+\hat{\beta}_1X_1+\hat{\beta}_2X_2+...+\hat{\beta}_pX_p$$

### 3.2.2. Some Important Questions

Multiple Linear Regression에서는 일반적으로 다음과 같은 질문들에 대한 답을 얻는 것이 중요하다.

#### 1. 독립변수 $X_1, X_2, ..., X_p$ 중에서 반응변수를 예측하는데 유의미한 변수가 하나라도 존재하는가?

Simple Linear Regression에서와 마찬가지로, 다음과 같은 가설검정을 통해 이 질문에 대한 답을 구할 수 있다.

$$H_0:\beta_1=\beta_2=\cdots=\beta_p=0 \quad \text{vs.} \quad H_a:\text{at least one } \beta_j \text{ is non-zero.}$$

이때 다음과 같은 검정통계량을 사용하여 귀무가설의 기각 여부를 판단하게 된다.

$$F=\frac{(\text{TSS}-\text{RSS})/p}{\text{RSS}/(n-p-1)}$$

전체 회귀 계수 중 $q$개의 회귀계수들에 대해서만 가설검정을 하는 경우도 존재한다.

$$H_0:\beta_{p-q+1}=\beta_{p-q+2}=\cdots=\beta_p=0$$

이러한 경우, 우리는 $q$개의 변수를 제외한 나머지 변수들을 가지고 두 번째 모형을 적합한다. 이 두 번째 모형의 $\text{SSE}$를 $\text{SSE}_0$라고 하면, $F$-통계량은 다음과 같다.

$$F=\frac{(\text{SSE}_0-\text{SSE})/q}{\text{SSE}/(n-p-1)}$$

#### 2. 유의미한 변수가 하나라도 존재한다면, 어떠한 변수들인가?

앞선 가설검정에서 귀무가설이 기각되었다면, 유의미한 변수가 하나라도 존재한다는 뜻이므로, 어떠한 변수가 유의미한 변수인지 알아볼 필요가 있다. 이상적인 방법으로는, 가능한 모든 경우의 수에 대해 각각의 모형을 적합하고 성능을 비교하는 것이 있다. 이때 성능을 판단하는 척도로 Mallow's $C_p$, AIC, BIC, adjusted $R^2$ 등을 사용한다. 하지만 변수의 개수가 $p$개일 때, 우리는 총 $2^p$개의 경우의 수를 갖는다. 따라서$p$가 매우 큰 경우, 모든 $2^p$개의 모형에 대해서 성능을 계산하고 비교하는 것은 불가능하다. 그렇기 때문에 모든 모형을 고려하지 않는 Forward Selection, Backward Selection, Mixed Selection과 같은 효율적인 방법을 사용해야 한다.

a) Forward Selection: 변수를 하나도 포함하지 않은 모형에서 시작하여 변수를 하나씩 추가하면서 기준 척도의 값을 비교하며 최적의 모형을 선택한다.

b) Backward Selection: 모든 변수를 포함한 모형에서 시작하여 변수를 하나씩 제거하면서 기준 척도의 값을 비교하며 최적의 모형을 선택한다.

c) Mixed Selection: 변수를 하나도 포함하지 않은 모형에서 시작하여 변수를 추가 또는 제거하면서 기준 척도의 값을 비교하며 최적의 모형을 선택한다.

#### 3. 모형의 성능이 어떠한가?

일반적으로 모형의 적합성을 측정할 때 $\text{MSE}$와 $R^2$를 사용한다.

$R^2$의 경우 모형과 관련이 없는 변수를 추가하더라도 무조건 값이 증가한다. 이는 Least Square Equations에 다른 변수를 추가하는 것이 트레이닝 데이터를 더욱 정확하게 적합하도록 만들기 때문이다.

$\text{MSE}$는 Multiple Linear Regression에서 $\sqrt{\frac{1}{n-p-1}\text{SSE}}$로 나타낼 수 있다. 새로운 변수를 추가함으로써 발생하는 $\text{SSE}$의 감소량이 $p$의 증가량보다 작은 경우 $\text{MSE}$가 증가한다.

#### 4. 예측의 정확도는 어떠한가?

다중 회귀 모형을 적합했다면, 새로운 데이터에 대한 반응변수의 값을 예측하는 것이 가능하다. 하지만 이러한 예측은 세 가지 불확실성을 동반한다. 첫 번째는 Reducible Error, 두 번째는 Model Bias, 세 번째는 Irreducible Error이다. 앞서 나온 Confidence Interval이 Reducible Error만 포함한다면, Prediction Interval은 Reducible Error와 Irreducible Error를 모두 포함한다.

## 3.3. Other Considerations in the Regression Model

### 3.3.1. Qualitative Predictors

지금까지는 독립변수가 수치형인 경우를 가정하였지만, 독립변수는 범주형이 될 수도 있다. 범주형 변수의 경우 더미 변수로 변환하여 사용하는 것이 바람직하다. 일반적으로 범주의 갯수보다 하나 적은 갯수의 더미 변수를 사용한다. 이때 더미 변수를 사용하지 않는 범주를 Baseline이라고 한다.

$$x_{i1}=\begin{cases} 1 & \mbox{if } i \mbox{th person is Asian} \\\ 0 & \mbox{if } i \mbox{th person is not Asian} \end{cases}$$
$$x_{i2}=\begin{cases} 1 & \mbox{if } i \mbox{th person is Caucasian} \\\ 0 & \mbox{if } i \mbox{th person is not Caucasian} \end{cases}$$

### 3.3.2. Extensions of the Linear Model

회귀 모형에서 가장 중요한 가정 두 가지는 Additive Assumption과 Linear Assumption이다. Additive Assumption은 어떤 변수 $X_j$가 $Y$에 미치는 여향이 다른 변수들과는 독립적임을 의미한다. Linear Assumption은 어떤 변수 $X_j$의 단위 변화에 따른 $Y$의 변화가 상수 단위로 일어나는 것을 의미한다.

#### 1. Removing the Additive Assumption

Additive Assumption을 배제하는 방법으로는 Interaction Term을 추가하는 것이 있다.

$$Y=\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_1X_2+\epsilon$$

#### 2. Non-Linear Relationships

Linear Assumption을 완화하는 방법에는 Polynomial Regression 등의 Non-Linear Relationship으로 모형을 확장하는 것이 있다.

$$Y=\beta_0+\beta_1X_1+\beta_2X^2_1+\epsilon$$

### 3.3.3. Potential Problems

회귀 모형을 적합하기 전에 항상 확인해야 할 것들이 있다.

#### 1. Non-Linearity of the Data

선형 회귀 모형은 기본적으로 독립 변수와 반응 변수 사이의 선형 관계를 전제로 한다. 만약 독립 변수와 반응 변수가 선형 관계에 있지 않다면, 모형의 성능이 매우 떨어지게 된다. 따라서 모형을 적합하기 전에 변수 간 관계를 미리 확인할 필요가 있다. Residual Plot은 데이터의 선형성을 확인하기에 가장 적합한 툴이다.

![FIGURE 3.9](/ISLR/FIGURE_3.9.png)

이상적인 경우, Residual Plot 상에서 어떠한 패턴도 드러나지 않아야 한다. 만약 특정한 패턴이 발견된다면, 선형 가정에 문제가 있다고 볼 수 있다. Residual Plot이 데이터의 비선형성을 보여준다면, 독립변수에 대해 $\log{X}$, $\sqrt{X}$, $X^2$ 등의 변환을 적용하여 비선형성을 완화할 수 있다.

#### 2. Correlation of Error Terms

선형 회귀 모형의 또 다른 중요한 가정은 Error Terms 사이의 독립성이다. 일반적으로 시계열 자료에서 이러한 독립성 가정이 위배되는 경우가 자주 발생한다. 마찬가지로 Residual Plot을 그려서 확인할 수 있다. 독립성 가정의 경우 데이터를 수집하는 과정에서부터 주의를 기울이는 것이 중요하다.

#### 3. Non-constant Variance of Error Terms

Error Terms의 등분산성 역시 선형 회귀 모형의 중요한 전제 사항이다. Residual Plot을 그렸을 때 Funnel Shape이 확인된다면 등분산성 가정이 위배된 것으로 판단할 수 있다. 이를 Heteroscedasticity라고 한다. 이러한 문제가 발생한 경우, 반응변수에 대해 $\log{Y}$, $\sqrt{Y}$ 등의 변환을 가함으로써 Error Terms의 이분산성을 완화시킬 수 있다.

![FIGURE 3.11](/ISLR/FIGURE_3.11.png)

#### 4. Outliers

반응변수의 관점에서 보았을 때, 회귀 모형으로부터 동떨어져 있는 값들을 Outlier라고 부른다. Residual Plot을 그렸을 때 상대적으로 높은 값을 보이는 점들이 Outlier라고 할 수 있다. 하지만 그 기준을 정하기 애매하기 때문에, 일반적으로 Studentized Residual Plot을 그려서 확인한다. Studentized Residual이 $3$보다 큰 경우 Outlier일 가능성이 높다.

![FIGURE 3.12](/ISLR/FIGURE_3.12.png)

#### 5. High Leverage Points

Outlier가 반응변수의 관점에서 보이는 비정상적인 값이라면, High Leverage Point는 종속변수의 관점에서 비정상적인 값을 찾은 것이다. 단순 선형 회귀에서는 산점도를 그려봄으로써 쉽게 찾아낼 수 있지만, 변수의 갯수가 많아지면 육안으로 구분하는 것이 불가능하다.  따라서 Leverage Statistic이라는 것을 구하게 된다.

$$h_i=\frac{1}{n}+\frac{\left(X_i-\bar{X}\right)^2}{\sum{\left(X_{i'}-\bar{X}\right)^2}}$$

일반적으로 Leverage Statistic이 $\frac{p+1}{n}$보다 큰 경우 High Leverage Point라고 판단한다.

![FIGURE 3.13](/ISLR/FIGURE_3.13.png)

#### 6. Collinearity

Collinearity는 두 개 이상의 변수들이 서로 연관되어있는 경우를 의미한다. Collinearity를 확인하기 가장 쉬운 방법은 Correlation Matrix를 그려보는 것이지만, 이러한 방법으로는 세 개 이상의 변수 사이의 Collinearity는 확인하지 못한다. 따라서 다음과 같은 식의 $\text{VIF}$를 계산하여 다중공선성 여부를 확인해야 한다.

$$\text{VIF}(\hat{\beta}_j)=\frac{1}{1-R^2\_{X_j|X\_{-j}}}$$

이때 $R^2_{X_j|X_{-j}}$는 $X_j$에 대해 나머지 변수들을 사용하여 적합한 회귀 모형의 $R^2$이다.