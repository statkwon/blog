---
title: "2. Statistical Learning"
date: 2020-09-01
TableOfContents: true
weight: 1
---

## 2.1. What is Statistical Learning?

### 2.1.1. Why Estimate $f$?

우리가 $f$를 추정하는 목적은 예측과 해석을 하기 위함이다.

$$\hat{Y}=\hat{f}(X)$$

**1. 예측** <br>
$f$의 추정치 $\hat{f}$를 안다면, 우리가 가진 $X_1, X_2, ..., X_p$를 투입하여 $Y$의 예측값 $\hat{Y}$을 구할 수 있다. 이러한 경우는 예측의 정확도만 높으면 되기 때문에 $\hat{f}$을 Black Box로 생각해도 무방하다. 예측 정확도는 두 가지 종류의 Error에 의해 영향을 받는다. 우선 우리가 $f$를 추정하는 과정에서, $f$와 완벽하게 같은 값을 구하지 못하기 때문에 Error가 발생한다. 이러한 Error는 더 적합한 모형을 적용해보며 줄여나갈 수 있기 때문에 Reducible Error라고 한다. 다른 하나는 $X$와 $Y$의 식에 포함되어 있던 $\epsilon$이다. 애초에 설명할 수 없는 부분이었기 때문에 이를 Irreducible Error라고 한다.

$$E(Y-\hat{Y})^2=E[f(X)+\epsilon-\hat{f}(X)]^2=\underbrace{[f(X)-\hat{f}(X)]^2}\_{Reducible}+\underbrace{\text{Var}(\epsilon)}\_{Irreducible}$$

{{<rawhtml>}}
<details>
<summary>Proof</summary>
$\begin{align} E(Y-\hat{Y})^2&=E[f(X)+\epsilon-\hat{f}(X)]^2
\\
&=E[(f(X)-\hat{f}(X))-\epsilon]^2
\\
&=E[(f(X)-\hat{f}(X))^2-2\epsilon(f(X)-\hat{f}(X))+\epsilon^2]
\\
&=E[f(X)-\hat{f}(X)]^2-E[2\epsilon(f(X)-\hat{f}(X))]+E(\epsilon^2)
\\
&=[f(X)-\hat{f}(X)]^2+\text{Var}(\epsilon) \end{align}$
</details>
{{</rawhtml>}}

그러므로 우리는 Reducible Error를 줄이는 것을 목표로 $f$를 추정할 것이다.

**2. 해석** <br>
단순한 $Y$의 예측값이 아닌, $X$와 $Y$의 관계를 해석하고자 하는 경우가 존재한다. 예를 들어, 독립변수들이 반응변수와 관련이 있는지, 관련이 있다면 어떠한 관계인지 등을 알아볼 수 있다. 이러한 경우 $f$는 Black Box로 간주되어서는 안되며, 그 정확한 형태를 파악해야 한다.

### 2.1.2. How Do We Estimate $f$?
대부분의 Statistical Learning 모형은 Parametric과 Non-Parametric으로 구분할 수 있다.

**1. 모수적 방법** <br>
모수적 방법은 우선 $f$의 형태에 대한 가정을 토대로 모형을 선정하고, 이후 Training Data를 사용하여 해당 모형을 적합하는 방식이다. 이러한 방식은 $f$를 통째로 추정하는 문제를 몇몇 모수들을 추정하는 문제로 단순화해준다. 하지만 $f$에 대한 가정이 실제와 많이 어긋나는 경우, 추정의 정확도가 매우 떨어지게 된다. 상대적으로 유연한 가정을 적용함으로써 이를 해결할 수는 있지만, 유연한 모형의 경우 더 많은 모수를 필요로 하기 때문에 과적합 문제가 발생할 수 있다.

**2. 비모수적 방법** <br>
비모수적 방법은 $f$의 형태에 대한 가정을 하지 않고, 그저 실제 Data Points에 최대한 근접할 수 있는 $f$를 추정한다. $f$의 형태에 대한 제약이 없기 때문에 더욱 다양하게 $f$를 추정하는 것이 가능하다. 하지만 모수가 아닌 $f$ 자체를 추정해야 하기 때문에 정확한 추정을 위해서는 매우 많은 양의 관측치를 확보해야 한다.

### 2.1.3. The Trade-Off Between Prediction Accuracy and Model Interpretability

{{<figure src="/islr_fig_2.7.png" width="400" height="200">}}

모델의 예측 정확도와 해석력은 Trade-Off 관계에 놓여있다. 따라서 해석이 목표인 경우에는 상대적으로 덜 유연한 모형을, 예측이 목표인 경우에는 유연한 모형을 사용하는 것이 좋다.

### 2.1.4. Supervised Versus Unsupervised Learning

대부분의 Statistical Learning 모형은 Supervised와 Unsupervised로 구분할 수 있다.

**1. 지도학습** <br>
지도학습은 독립변수에 상응하는 반응변수가 존재하는 경우를 말한다. 앞서 다뤘던 예측과 해석의 문제들은 모두 지도학습의 영역에 속한다. <br>
ex) Linear Regression, Logistic Regression, GAM, Boosting, SVM

**2. 비지도학습** <br>
비지도학습은 독립변수에 상응하는 반응변수가 존재하지 않는 경우를 말한다. 이러한 경우 데이터 간의 관계를 이해하는데 초점을 맞추어 볼 수 있다. <br>
ex) Cluster Analysis

**3. Semi-Supervised Learning** <br>
총 $n$개의 관측치가 있을 때, 그 중 $m$개에 대해서는 반응변수가 존재하고, 나머지 $n-m$개에 대해서는 반응변수가 존재하지 않는 경우가 있다. 이러한 경우를 Semi-Supervised Learning이라고 부르는데, 이 책에서는 다루지 않는다.

### 2.1.5. Regression Versus Classification Problems

변수는 수치형과 범주형으로 구분할 수 있다. 이때 수치형 변수에 관한 문제들을 Regression Problems, 범주형 변수에 관한 문제들을 Classification Problems라고 한다. 우리는 변수의 형태에 따라 적절한 모형을 선택하게 되는데, KNN이나 Boosting 등의 일부 모형들은 두 가지 경우에 대해 모두 사용이 가능하기도 하다.

모든 데이터에 대해 적합한 모형은 존재하지 않는다. 따라서 우리는 매번 주어진 데이터에 대한 최적의 모형을 선택해야 하기 때문에, 모형의 성능에 대한 척도가 필요하다. 일반적으로 모형의 성능은 해당 모형을 사용하여 예측한 값이 실제값에 얼마나 가까운지를 기준으로 판단한다.

### 2.2.1. Measuring the Quality of Fit

$$MSE=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{f}(x_i))^2$$

Regression Problems에서 가장 많이 사용하는 척도는 Mean Squared Error (MSE)이다. 위의 식은 Training Data를 가지고 계산된 Training MSE를 구하는 식이다. 하지만 우리는 Training Data가 아닌, Previously Unseen Test Data에 대한 모형의 성능을 확인해야 하기 때문에 아래와 같은 식을 통하여 Test MSE를 계산한다.

$$Ave(y_0-\hat{f}(x_0))^2$$

이때 $(x_0, y_0)$는 Test Data이다. 일반적으로 많은 모형들이 Training MSE를 최소화하는 방향으로 모수를 추정하기 때문에 Test MSE는 상대적으로 Training MSE보다 높게 나타난다. 우리는 Training MSE가 아닌, Test MSE가 가장 낮은 모형을 선택해야 한다.  만약 별도의 Test Data를 가지고 있는 경우라면 각 모형별 Test MSE를 계산하여 비교해볼 수 있겠지만, 그렇지 않은 경우에는 다른 방법을 찾아야 한다. 이러한 경우, 결코 낮은 Training MSE가 낮은 Test MSE를 보장해주지 않는다는 점을 주의해야 한다.

{{<figure src="/islr_fig_2.9.png" width="400" height="200">}}

위 그림을 보면 Training MSE의 경우 모형의 유연성이 증가함에 따라 단조적으로 감소하지만, Test MSE의 경우 U자 형태로 나타나고 있다. 즉, 모델의 유연성이 증가하면 Training MSE는 항상 줄어들지만, Test MSE는 그렇지 않을 수도 있다는 것을 의미한다. 따라서 별도의 Test Data가 없는 경우에는 Cross Validation과 같이 Training Data를 이용해 Test MSE를 추정하는 방식을 주로 사용하게 된다.

### 2.2.2. The Bias-Variance Trade-Off

$$E(y_0-\hat{f}(x_0))^2=\text{Var}(\hat{f}(x_0))+[\text{Bias}(\hat{f}(x_0))]^2+\text{Var}(\epsilon)$$

위 식과 같이 Test MSE의 기대값은 모형의 분산, 모형의 편향의 제곱, 그리고 $\epsilon$의 분산의 합으로 나타낼 수 있다. 따라서 Test MSE를 줄이기 위해서는 Low Variance와 Low Bias를 동시에 만족하는 모형을 사용해야 한다.

{{<rawhtml>}}
<details>
<summary>Proof</summary>
$\begin{align} E(y_0-\hat{f}(x_0))^2&=E[f(x_0)+\epsilon-\hat{f}(x_0)]^2
\\
&=E[(f(x_0)-\hat{f}(x_0))+\epsilon]^2
\\
&=E[(f(x_0)-\hat{f}(x_0))^2+2\epsilon(f(x_0)-\hat{f}(x_0))+\epsilon^2]
\\
&=E[f(x_0)-\hat{f}(x_0)]^2+E[2\epsilon(f(x_0)-\hat{f}(x_0))]+E(\epsilon^2)
\\
&=E[f(x_0)-\hat{f}(x_0)]^2+\text{Var}(\epsilon)
\\
&=E[f(x_0)-E(\hat{f}(x_0))-(\hat{f}(x_0)-E(\hat{f}(x_0)))]^2+\text{Var}(\epsilon)
\\
&=E[(f(x_0)-E(\hat{f}(x_0)))^2-2(f(x_0)-E(\hat{f}(x_0)))(\hat{f}(x_0)-E(\hat{f}(x_0)))+(\hat{f}(x_0)-E(\hat{f}(x_0)))^2]+\text{Var}(\epsilon)
\\
&=E[E(\hat{f}(x_0))-f(x_0)]^2-2(f(x_0)-E(\hat{f}(x_0)))E[\hat{f}(x_0)-E(\hat{f}(x_0))]+E[\hat{f}(x_0)-E(\hat{f}(x_0))]^2+\text{Var}(\epsilon)
\\
&=(f(x_0)-E(\hat{f}(x_0)))^2+E[\hat{f}(x_0)-E(\hat{f}(x_0))]^2+\text{Var}(\epsilon)
\\
&=[\text{Bias}(\hat{f}(x_0))]^2+\text{Var}(\hat{f}(x_0))+\text{Var}(\epsilon) \end{align}$
</details>
{{</rawhtml>}}

**1. 분산** <br>
Training Data를 사용하여 적합한 모형으로 $\hat{f}(x_0)$를 구하는 것이므로 사용하는 데이터가 달라질 경우 예측값도 달라지게 된다. 이때 사용하는 데이터에 따라 예측값이 달라지는 정도가 모형의 분산이다. 직관적으로 생각해 보았을 때, 투입하는 데이터에 따라 매번 다른 결과가 나오는 것은 좋지 않은 상황이다. 모형의 분산은 당연히 낮을 수록 좋다. 일반적으로 유연한 모형일 수록 모형의 분산이 높아진다.

**2. 편향** <br>
모형의 편향은 실제 현상과 우리가 사용하는 모형 사이의 차이를 의미한다. 예를 들어, 어떠한 요인들 간에 선형 관계를 가정하였는데, 실제로는 선형 관계가 아닌 경우 모형의 편향이 크다고 할 수 있다. 일반적으로 유연한 모형일 수록 모형의 편향이 낮아진다.

정리하자면, 유연한 모형일 수록 분산은 높아지고 편향은 낮아지게 된다. 우리는 Test MSE를 낮추기 위해 분산과 편향이 동시에 낮은 모형을 사용하여야 하므로, 분산과 편향의 적절한 타협점을 찾아내야 한다.

### 2.2.3. The Classification Setting

$$\frac{1}{n}\sum_{i=1}^nI(y_i \neq \hat{y}_i)$$

Classification Problems에서는 Error Rate를 모형 평가 척도로 사용한다. 위의 식은 Training Error Rate를 나타낸다. 하지만 Regression Problems에서와 마찬가지로 Classification Problems 역시 Test Error Rate를 기준으로 모형을 선택해야 한다. Test Error Rate의 식은 아래와 같다.

$$\text{Ave}(I(y_0 \neq \hat{y}_0)$$

**Bayes Classifier** <br>
$$\text{P}(Y=j|X=x_0)$$

Bayes Classifier는 위와 같은 조건부 확률을 가장 크게 하는 방향으로 데이터를 분류한다. 이 확률이 최대가 되는 지점들을 연결한 것이 바로 Bayes Decision Boundary이고, Bayes Decision Boundary를 기준으로 데이터가 분류된다. Bayes Classifier를 사용하였을 때 발생하는 Error Rate를 Bayes Error Rate라고 하며, 이는 항상 Test Error Rate의 최소값에 해당한다. Bayes Error Rate의 식은 아래와 같다.

$$1-E(\underset{j}{\text{max}}\text{P}(Y=j|X))$$

**K-Nearest Neighbors (KNN)** <br>
앞서 언급했듯이 Bayes Classifier를 사용하는 경우 Test Error Rate를 최소화할 수 있으므로, 모든 Classification Problems에 대해 Bayes Classifier를 사용하는 것이 이상적이다. 하지만 실제로는 $Y|X$의 분포를 알지 못하기 때문에 조건부 확률을 계산하는 것이 불가능하다. 그렇기 때문에 많은 방식들이 $Y|X$의 분포를 추정하여 조건부 확률을 계산하고자 한다. K-Nearest Neighbors (KNN)는 이러한 방식의 대표적인 예시이다.

{{<figure src="/islr_fig_2.14.png" width="400" height="200">}}

$$\text{P}(Y=j|X=x_0)=\frac{1}{K}\sum_{i\in\mathcal{N}_0}I(y_i=j)$$

KNN은 위와 같은 방식으로 조건부 확률을 추정한다. Training Data 중에서 $x_0$와 가까운 $K$개의 점을 찾고, 그 $K$개의 점들 가운데 $j$로 분류된 점들의 비율을 조건부 확률의 추정치로 사용한다. 이때 적절한 $K$의 값을 선택하는 것이 중요하다. $K$가 커질 수록 모형의 유연성이 감소하게 된다. 이러한 방식은 매우 단순해보이지만 종종 Bayes Classifier에 근접한 결과를 얻어내기도 한다.

---

**Reference**

An Introduction to Statistical Learning