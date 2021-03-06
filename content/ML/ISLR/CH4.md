---
title: "4. Classification"
date: 2021-02-10
TableOfContents: true
weight: 3
---

4장에서는 반응변수가 범주형인 경우, 즉, Classification의 문제를 다룬다.

## 4.2. Why Not Linear Regression?

선형 회귀 모형은 반응변수가 범주형인 경우 사용이 적합하지 않다. 일반적으로 범주의 갯수가 세 개 이상인 경우 범주형 반응변수를 선형회귀분석에 적합한 형태의 수치형 변수로 변환하는 방법이 존재하지 않는다. 설령 범주의 갯수가 두 개라고 하더라도 선형 회귀 모형의 결과를 해석하는데 있어 문제가 발생한다. 두 가지 범주를 0과 1로 코딩하여 선형 회귀 모형을 적합한 경우, 몇몇 예측값은 0과 1의 범위를 벗어나는 값을 가질 수 있다.

## 4.3. Logistic Regression

{{<figure src="/islr_fig_4.2.png" width="600" height="400">}}

로지스틱 회귀 모형은 $X$가 주어졌을 때 $Y$가 $1$의 범주에 속할 조건부확률인 $P(Y=1|X)$에 대한 모형이다. $P(Y=1|X)$는 확률값이기 때문에 항상 $0$과 $1$ 사이의 값만을 갖는다. 로지스틱 회귀 모형에서는 확률값에 대해 임의의 Cutoff 값을 설정하여 범주를 구분한다. 예를 들어 $0.5$를 기준으로 삼은 경우, $P(Y=1|X)>0.5$면 $1$로 분류하고, $P(Y=1|X)<0.5$면 $0$으로 분류하는 식이다.

### 4.3.1. The Logistic Model

선형 회귀 모형에서는 $X$에 대해 $\beta_0+\beta_1X$의 선형 함수를 사용하였지만, 로지스틱 회귀 모형에서는 $0$과 $1$ 사이의 값만을 갖는 함수를 사용한다. 이러한 함수에는 여러 종류가 있지만 그중에서도 $\frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}$의 로지스틱 함수를 사용한다. 따라서 로지스틱 회귀 모형은 다음과 같은 형태를 갖는다.

$$P(Y=1|X)=\frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}$$

위 식을 변형하면 아래와 같은 식을 얻을 수 있다.

$$\frac{P(X)}{1-P(X)}=e^{\beta_0+\beta_1X}$$

$\frac{P(X)}{1-P(X)}$를 Odds라고 부르는데, 이는 어떤 결과가 발생할 가능성을 나타내는 지표이다. $0$에 가까울 수록 낮은 가능성을, $\infty$에 가까울수록 높은 가능성을 의미한다.

다시 위 식의 양 변에 $\log$를 취하면 아래와 같은 식을 얻을 수 있다.

$$\log{\left(\frac{P(X)}{1-P(X)}\right)}=\beta_0+\beta_1X$$

$\log{\left(\frac{P(X)}{1-P(X)}\right)}$를 Log-Odds 또는 Logit이라고 하며, 로지스틱 회귀 모형은 Logit에 대한 선형 함수의 형태라는 것을 확인할 수 있다. 따라서 $X$가 한 단위 증가함에 따라 Log-Odds가 $\beta_1$만큼 증가하는 것으로 해석할 수 있다. 또는 $X$가 한 단위 증가함에 따라 Odds가 $e^{\beta_1}$배 되는 것으로 해석하는 것도 가능하다.

### 4.3.2. Estimating the Regression Coefficients

Simple Logistic Regression에서 회귀계수 $\beta_0$와 $\beta_1$을 추정할 때는 Method of Maxiumu Likelihood를 사용한다. Method of Least Squares를 사용하는 것도 가능하지만, 일반적으로 Method of Maximum Likelihood를 사용하는 것을 선호한다.

$$\ell(\beta_0, \beta_1)=\prod_{i:y_i=1}{P(X_i)}\prod_{i':y_{i'}=0}{\left(1-P(X_{i'})\right)}$$

Method of Maximum Likelihood는 위와 같은 Likelihood Function을 최대화하는 $\hat{\beta}_0$와 $\hat{\beta}_1$을 찾는 방법이다. 로지스틱 회귀 모형뿐만 아니라 다른 많은 비선형 모형들을 적합할 때에도 자주 사용된다.

### 4.3.3. Making Predictions

$$\hat{P}(Y=1|X)=\frac{e^{\hat{\beta}_0+\hat{\beta}_1X}}{1+e^{\hat{\beta}_0+\hat{\beta}_1X}}$$

추정한 회귀 계수를 사용하여 새로운 데이터에 대한 $Y$값을 예측한다. 이때 독립변수로 수치형 변수가 아닌 범주형 변수를 사용하는 것 역시 가능하다. 선형 회귀 모형과 같은 방식으로 범주형 변수를 더미 변수로 변환하여 모형을 적합하면 된다.

### 4.3.4. Multiple Linear Regression

$$P(Y=1|X)=\frac{e^{\beta_0+\beta_1X_1+\cdots+\beta_pX_p}}{1+e^{\beta_0+\beta_1X_1+\cdots+\beta_pX_p}}$$

여러 개의 독립변수를 사용하여 모형을 적합할 수도 있다. 여러 개의 변수를 사용하는 경우 변수 간의 Confounding Effect가 모형에 영향을 주게 된다.

### 4.3.5. Logistic Regression for $>2$ Response Classes

반응변수의 범주의 갯수가 세 개 이상인 경우에 대해서도 로지스틱 회귀 모형을 적합하는 것이 가능하다. 하지만 이러한 경우에는 아래서 다룰 Discriminant Analysis를 사용하는 것이 더욱 일반적이다.

## 4.4. Linear Discriminant Analysis

로지스틱 회귀 모형과 비교했을 때 Linear Discriminant Analysis가 갖는 장점은 다음과 같다.

- 범주가 명확하게 구분되는 경우, 로지스틱 회귀 모형에 대한 모수 추정은 상당히 불안정한 반면, Linear Discriminant Analysis는 이러한 문제로부터 자유롭다.
- 데이터의 갯수가 적고 독립변수들이 어느정도 정규분포를 따르는 경우 Linear Discriminant 모형이 로지스틱 회귀 모형에 비해 더욱 안정적이다.
- 앞서 언급했듯이, 세 개 이상의 범주를 갖는 경우에 대해서는 Linear Discriminant Analysis가 더 인기 있는 모형이다.

### 4.4.1. Using Bayes' Theorem for Classification

데이터를 총 $K(K≥2)$개의 그룹으로 분류하는 상황을 가정하자. 베이즈 정리를 사용하여 우리가 구하고자 하는 사후 확률 $P(Y=k|X=x)$를 다음과 같은 식으로 나타낼 수 있다.

$$P(Y=k|X=x)=\frac{\pi_kf_k(x)}{\sum_{l=1}^K{\pi_lf_l(x)}}$$

여기서 $\pi_k$는 $P(Y=k)$, 즉, 랜덤하게 뽑힌 데이터가 $k$번째 그룹에 속할 사전 확률을 나타낸다. $f_k(X)$는 $P(X=x|Y=k)$로, $k$번째 그룹에 속한 데이터의 $X$값에 대한 밀도 함수이다. $\pi_k$와 $f_k(X)$를 추정하여 위 식에 대입함으로써 $P(Y=k|X=x)$의 값을 구할 수 있다. $\pi_k$를 추정하는 것은 간단하다. Training Data에서 각각의 범주가 차지하는 비율을 계산하기만 하면 된다. 하지만 $f_k(X)$를 추정하는 것이 까다롭다.

### 4.4.2. Linear Discriminant Analysis for $p=1$

$f_k(X)$를 추정하기 위해서는 우선 그 형태에 대한 가정이 필요하다. $f_k(X)$가 평균 $\mu_k$와 분산 $\sigma^2_k$를 갖는 정규분포를 따른다고 가정하자. 또한 $\sigma^2_1=\cdots=\sigma^2_K=\sigma^2$를 가정하자. 이러한 가정들을 바탕으로 $P(Y=k|X=x)$를 다음과 같은 식으로 나타낼 수 있다.

$$P(Y=k|X=x)=\frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\frac{1}{2\sigma^2}\left(x-\mu_k\right)^2\right)}}{\sum_{l=1}^K\pi_l\frac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\frac{1}{2\sigma^2}\left(x-\mu_l\right)^2\right)}}$$

Bayes Classifier는 데이터를 위 식을 최대화하는 $k$로 분류하는 것이므로, 위 식을 최대화하는 $k$값을 찾을 것이다. 우선 위 식의 양 변에 $\log$를 취하여 정리한다. 정리한 식은 아래의 식과 같은 형태를 갖는다.

$$\delta_k(x)=x\cdot\frac{\mu_k}{\sigma^2}-\frac{\mu^2_k}{2\sigma^2}+\log{(\pi_k)}$$

하지만 우리는 $\mu_k$, $\sigma^2$, $\pi_k$를 전부 알지 못한다. 따라서 $\delta_k(x)$에 $\mu_k$, $\sigma^2$, $\pi_k$의 추정치를 대신 사용한다. 각각의 추정치는 다음과 같은 방식으로 구할 수 있다.

$$\hat{\mu}_k=\frac{1}{n_k}\sum\_{i:y_i=k}{x_i}$$

$$\hat{\sigma}^2=\frac{1}{n-K}\sum_{k=1}^K\sum_{i:y_i=k}{\left(x_i-\hat{\mu}_k\right)^2}$$

$$\hat{\pi}_k=\frac{n_k}{n}$$

이때 $n_k$는 $k$번째 범주에 속한 데이터의 갯수를 의미한다. 결과적으로 LDA는 $\hat{\delta}_k(x)=x\cdot\frac{\hat{\mu}_k}{\hat{\sigma}^2}-\frac{\hat{\mu}^2_k}{2\hat{\sigma}^2}+\log{\left(\hat{\pi}_k\right)}$를 최대화하는 범주 $k$로 데이터를 분류하게 된다. 여기서 참고로, Linear Discriminant Analysis라고 부르는 이유는 이러한 $\hat{\delta}_k(x)$가 $x$에 대한 선형 함수이기 때문이다.

### 4.4.3. Linear Discriminant Analysis for $p>1$

$X \sim N(\mu, \mathbf{\Sigma})$와 같이 $X=(X_1, X_2, \ldots, X_p)$가 다변량 정규분포를 따르는 경우에도 LDA를 사용할 수 있다. 이러한 경우 $k$번째 범주의 데이터들은 $N(\mu_k, \mathbf{\Sigma})$를 따른다. $p=1$인 경우와 마찬가지로, $\frac{\pi_kf_k(x)}{\sum_{l=1}^K{\pi_lf_l(x)}}$에 $f_k(X=x)$를 대입하여 아래와 같은 식을 얻을 수 있다.

$$\delta_k(x)=x^T\mathbf{\Sigma}^{-1}\mu_k-\frac{1}{2}\mu^T_k\mathbf{\Sigma}^{-1}+\log{\pi_k}$$

위 식을 최대화하는 $k$값으로 범주를 분류한다.

{{<figure src="/islr_fig_4.6.png" width="600" height="400">}}

뿐만 아니라, 위 그림과 같이 각각의 범주를 구분하는 Decision Boundary를 구하는 것도 가능하다. $k$번째 범주와 $l$번째 범주를 구분하는 Decision Boundary는 $\delta_k(x)=\delta_l(x)$, 즉, $x^T\mathbf{\Sigma}^{-1}\mu_k-\frac{1}{2}\mu^T_k\mathbf{\Sigma}^{-1}\mu_k=x^T\mathbf{\Sigma}^{-1}\mu_l-\frac{1}{2}\mu^T_l\mathbf{\Sigma}^{-1}\mu_l$를 만족하는 $x$값들의 집합이다. 이때 여전히 $\mu_1, \ldots, \mu_k$, $\mathbf{\Sigma}$, $\pi_1, \ldots, \pi_k$의 값을 알지 못하므로 각각의 추정치를 대신 사용한다. 위 그림에서 LDA Decision Boudnary(직선)가 Bayes Decision Boundary(점선)와 매우 유사한 것을 확인할 수 있다.

**cf. Confusion Matrix & ROC Curve**

Regression 문제와 달리, Classification 문제에서는 모형의 성능을 측정할 때 주의해야할 점이 있다. 범주가 두 개인 경우, Binary Classifier는 두 가지 종류의 Error를 발생시키기 때문에, 모형의 성능을 판단할 때 두 가지 Error 모두 고려해주어야 한다. Confusion Matrix를 통해 이를 쉽게 확인할 수 있다.

기본적인 형태의 Confusion Matrix는 역학 용어를 기반으로 한다. Positive는 양성을, Negative는 음성을 뜻한다. 따라서 True Positive는 양성으로 예측하였고, 실제로 양성인 경우, True Negative는 음성으로 예측하였고, 실제로 음성인 경우, False Positive는 양성으로 예측하였지만 실제로 음성인 경우, False Negative는 음성으로 예측하였지만 실제로 양성인 경우에 해당한다. 이때 실제로 양성인 경우에 대해 정확하게 예측한 확률을 Sensitivity, 실제로 음성인 경우에 대해 정확하게 예측한 확률을 Specificity라고 한다. 만약 어느 한 쪽의 지표가 더 중요한 상황이라면, 범주를 구분하는 기준값을 조절하여 더 중요한 쪽의 정확도를 높일 수 있다. $P(Y=1|X=x)>c$에서 $c$를 threshold라고 부르는데, 이것이 바로 앞서 말한 범주를 구분하는 기준이 된다. 예를 들어, Sensitivity보다 Specificity를 우선시 해야하는 경우 threshold값을 높임으로써 False Positive인 경우를 줄일 수 있다. threshold값은 분류하고자 하는 데이터의 도메인 지식에 따라 결정해야한다.

$$\text{Sensitivity}=\frac{\text{TP}}{\text{FN}+\text{TP}} \qquad \text{Specificity}=\frac{\text{TN}}{\text{TN}+\text{FP}}$$

$1-\text{Specificity}$를 $x$축으로, $\text{Sensitivity}$를 $y$축으로 하는 그래프인 ROC Curve는 각 threshold값에 따른 두 가지 Error를 동시에 표현할 수 있다는 점에서 유용하게 사용된다. 또한 모형의 전체적인 성능을 ROC Curve 아래의 면적, AUC로 판단할 수도 있다.

{{<figure src="/islr_fig_4.8.png" width="400" height="200">}}

### 4.4.4. Quadratic Discriminant Analysis

QDA는 LDA와 마찬가지로 변수들이 정규분포를 따름을 가정하지만, 각 범주별로 서로 다른 Covariance Matrix를 갖는다는 점에서 LDA와 차이가 있다. 즉, $k$번째 범주에 속한 데이터들은 $N(\mu_k, \mathbf{\Sigma}_k)$를 따른다. 이러한 가정 하에서 $\delta_k(x)$를 구해보면 다음과 같다.

$$\begin{align} \delta_k(x)&=-\frac{1}{2}(x-\mu_k)^T\mathbf{\Sigma}^{-1}_k(x-\mu_k)-\frac{1}{2}\log{|\mathbf{\Sigma}_k|}+\log{\pi_k}
\\\\
&=-\frac{1}{2}x^T\mathbf{\Sigma}^{-1}_kx+x^T\mathbf{\Sigma}^{-1}_k\mu_k-\frac{1}{2}\mu^T_k\mathbf{\Sigma}^{-1}_k\mu_k-\frac{1}{2}\log{|\mathbf{\Sigma}_k|}+\log{\pi_k} \end{align}$$

위 식을 최대화하는 $k$값으로 범주를 분류한다. 앞선 경우와 마찬가지로, $\delta_k(x)$가 $x$에 대한 Quadratic Function이기 때문에 Quadratic Discriminant Analysis라고 부른다.

{{<figure src="/islr_fig_4.9.png" width="600" height="400">}}

만약 Training Data의 갯수가 부족한 경우라면 일반적으로 분산을 줄일 수 있는 LDA의 성능이 QDA보다 좋다고 할 수 있다. 하지만 Training Data가 충분하여 모형의 분산에 대해 크게 신경을 쓰지 않아도 되는 경우나, Covariance Matrix가 모든 범주에 대해 동일하다는 가정이 적합하지 않은 경우에는 QDA를 사용하는 것이 좋다.