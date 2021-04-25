---
title: "Supervised Learning"
draft: true
tableofcontents: true
---

## 1. What is Our Goal?

우리는 $Y$와 $X$의 관계를 알고 싶어한다. 이 관계를 어떠한 한 함수 $f$로 나타내보자. 우리는 모든 데이터를 가지고 있지 않기 때문에, 이 함수의 정확한 형태를 아는 것이 불가능하다. 만약 모든 데이터를 가지고 있다면, 굳이 $Y$와 $X$의 관계를 함수로 표현할 필요 없이 주어진 데이터를 이용하여 예측과 분류를 하면 될 것이다. 하지만 대부분의 경우 우리는 모든 데이터를 가지고 있지 않다. 따라서 우리는 주어진 상황에 맞는 최적(optimal)의 함수를 선택해야 한다.

자, 그러면 우리가 최적의 함수를 찾았다고 생각해보자. 그럼에도 불구하고, 사실 우리는 $Y$에 대한 정확한 예측과 분류를 하지는 못한다. 우리가 찾은 함수는 '최적의' 함수이지, '정답'이 아니다. 아무리 정답에 가까운 함수의 형태를 찾아낸다고 하더라도, $Y$에 대한 예측이나 분류를 할 때마다 우리가 통제할 수 없는 Error($\epsilon$)가 발생한다. 따라서 우리는 $Y$와 $X$의 관계를 다음과 같이 표현하게 된다.

$$Y=f(X)+\epsilon$$

통계학에서는 $\epsilon$이 평균이 $0$이고 분산이 $\sigma^2$인 분포를 따르는 Random Variable이라고 가정한다. 자연스레 $Y$ 역시 평균이 $f(X)$이고 분산이 $\sigma^2$인 분포를 따르는 Random Variable이 된다.

이제 $f$의 형태를 결정하는 방법에 대해서 이야기해보자. $f$의 형태는 단순하게 결정되지 않으며, 많은 요인들의 영향을 받는다. 첫 번째는 '어떤 모형을 사용하는지(which model)'이다. Simple Linear Regression부터 Neural Network까지, 세상에는 다양한 모형이 존재한다. 두 번째는 '어떤 변수를 사용하는지(which variable)'이다. 결코 모든 변수가 유용하지는 않다. 마지막은 '어떤 하이퍼파라미터를 사용하는지(which hyper-paramter)'이다.

우리가 $f$의 형태를 결정하기 위해서는 이러한 세 가지 요인들을 결정하는 기준이 필요하다. 우리가 어떤 목적을 가지고 $Y$와 $X$의 관계를 알고자 하는지를 생각해보면, 이 기준을 정하는 것은 어렵지 않다. 크게 두 가지의 목적을 생각해볼 수 있다. 첫 번째는 새로운 데이터가 주어졌을 때, 해당 데이터를 사용하여 예측과 분류를 하려는 것이다. 이러한 경우 예측 정확도나 분류 정확도가 기준이 될 수 있다. 두 번째는 $Y$와 $X$의 관계를 해석하려는 것이다. 이러한 경우에는 정확도보다는 해석력을 더 우선시하게 된다. 해석력의 경우 아직까지는 수치화된 기준이 없는 주관의 영역이기 때문에, 이 블로그에서는 우선 정확도를 기준으로 하는 상황에 초점을 맞추고, 해석력과 관련된 이야기는 그 이후에 다루어 볼 계획이다.

## 2. How to Minimize the Test Error?

새로운 데이터에 대한 예측 정확도나 분류 정확도를 보통 간단하게 Test Error라고 하거나, 각각 Prediction Error, Classification Error라고 한다. 우리는 어떤 방식으로 이 Test Error를 계산할 것인지에 대한 측도를 결정해야 한다. 새로운 데이터를 $X$, 그에 대한 예측 또는 분류 결과를 $Y$라고 했을 때, 일반적으로 예측 문제에서는 $(Y-f(X))^2$의 Squared Error를, 분류 문제에서는 $I\_{Y\neq\hat{Y}}$의 $0$-$1$ Error를 사용한다. 물론 이는 하나의 데이터에 대한 Error를 계산한 것에 불과한다. 앞으로 주어질 모든 데이터에 대해 Test Error를 최소화하기 위해서는 Expected Test Error를 최소화해야한다. 하지만 이는 아직 주어지지 않은 데이터에 대한 것이므로, 우리가 계산할 수 없는 값이다. 방법이 없는 것은 아니다. 우리는 주어진 데이터를 활용하여 이 값을 추정해볼 수 있다. 주어진 데이터 중 일부를 Test 데이터셋으로 삼고 그에 대한 Test Error를 계산함으로써 그 값을 Expected Test Error의 추정치로 사용할 수 있다. 

$$\dfrac{1}{n}\sum_{i=1}^n(y_i-\hat{f}(\mathbf{x}_i))^2\overset{p}{\longrightarrow}\text{E}[(Y-\hat{f}(X))^2]$$

$$\dfrac{1}{n}\sum_{i=1}^nI_{y_i\neq\hat{y}_i}\overset{p}{\longrightarrow}\text{E}[I_{Y\neq\hat{Y}}]$$

WLLN에 따라 Test 데이터셋의 크기가 커질 수록 상응하는 Test Error의 추정치는 Expected Test Error로 확률 수렴하게 된다. 즉, 우리는 주어진 데이터 중 일부를 활용하여 Test Error의 추정치를 계산함으로써 이를 $f$의 형태를 결정하는 기준으로 삼을 수 있다.

## 3. Statistical Decision Theory

Decision Theory를 통해 Expected Test Error를 최소화하는 $f$가 어떤 의미를 갖는지 살펴볼 수 있다. Decision Theory에서는 Error보다는 Loss라는 표현을 주로 사용한다. 우선 수식을 통해 Expected Prediction Error를 최소화하는 $f$를 찾아보자.

$\displaystyle\text{E}[(y-\hat{f}(\mathbf{x}_0))^2]=\int(y-f(\mathbf{x}_0))^2f\_{Y|X}(y|x)dy$

$\displaystyle\underset{f}{\text{argmin}}$

$\begin{aligned}
\dfrac{\partial}{\partial f}\int(y-f(x))^2f_{Y|X}(y|x)dy&=\int\dfrac{\partial}{\partial f}\left\\{(y-f(x))^2f_{Y|X}(y|x)\right\\}dy \\\\
&=-2\int(y-f(x))f_{Y|X}(y|x)dy=0
\end{aligned}$

$\begin{aligned}
\therefore f(x)&=\int yf_{Y|X}(y|x)dy \\\\
&=\text{E}[Y|X=x]
\end{aligned}$

## 4. Bias-Variance Decomposition

우리는 Expected Test Error를 수식적으로 분해함으로써 몇 가지 유용한 정보를 얻을 수 있다.

$\begin{aligned}
\text{E}[(y-\hat{f}(\mathbf{x}_0))^2]&=\text{E}[(f(\mathbf{x}_0)+\epsilon_0-\hat{f}(\mathbf{x}_0))^2] \\\\
&=\text{E}[(f(\mathbf{x}_0)-\hat{f}(\mathbf{x}_0))^2]+2\text{E}[(f(\mathbf{x}_0)-\hat{f}(\mathbf{x}_0))\epsilon_0]+\text{E}[\epsilon_0^2] \\\\
&=\text{E}[(f(\mathbf{x}_0)-\hat{f}(\mathbf{x}_0))^2]+\text{Var}(\epsilon_0) \\\\
&=\text{E}[(f(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)]-(\hat{f}(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)]))^2]+\text{Var}(\epsilon_0) \\\\
&=\text{E}[(f(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)])^2]+\text{E}[(\hat{f}(\mathbf{x}_0)-\text{E}[\hat{f}(\mathbf{x}_0)])^2]+\text{Var}(\epsilon_0) \\\\
&=[\text{Bias}(\hat{f}(\mathbf{x}_0))]^2+\text{Var}(\hat{f}(\mathbf{x}_0))+\text{Var}(\epsilon_0)
\end{aligned}$

위의 식을 보면, Expected Prediction Error는 우리가 사용하려는 함수 $\hat{f}$의 Bias와 Variance, 그리고 Error의 Variance로 분해되는 것을 확인할 수 있다.

우리는 Error의 Variance를 줄일 수 없다. 즉, Expected Test Error는 항상 $0$보다 큰 값을 갖는다. 이는 앞서 최적의 함수 $f$를 찾았다고 하더라도 $Y$에 대한 정확한 예측을 할 수 없다고 한 것을 수식으로 확인할 수 있는 부분이다. 결과적으로 Expected Test Error를 줄이기 위해서는 Bias와 Variance가 모두 작은 함수를 사용해야 한다.


이러한 과정이 Machine Learning이라고 불리는 이유는, Training 데이터셋을 사용하여 함수의 Parameter를 스스로 학습하는 구조 때문이다. 함수의 Parameter는 Training Error를 최소화하는 방향으로 학습된다. 예를 들어, Linear Regression을 사용하는 경우, 

$$\mathbf{Y}=f(X)+\boldsymbol{\epsilon}$$

$\mathbf{Y}$를 $n\times 1$ Column Vector, $X$를 $n\times p$ Matrix로 생각해보자. 통계학에서는 $\epsilon_1, \epsilon_2, \ldots, \epsilon_n$이 평균이 $0$이고 분산이 $\sigma^2$인 분포를 따르는 Random Sample이라고 가정한다. 자연스레 $Y_1, Y_2, \ldots, Y_n$ 역시 평균이 $f(X)$이고 분산이 $\sigma^2$인 분포를 따르는 Random Sample이 된다.

$$\mathbf{Y}=\begin{bmatrix} Y_1 \\\\ Y_2 \\\\ \vdots \\\\ Y_n \end{bmatrix} \quad X=\begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1p} \\\\ x_{21} & x_{22} & \cdots & x_{2p} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ x_{n1} & x_{n2} & \cdots & x_{np} \end{bmatrix} \quad \boldsymbol{\epsilon}=\begin{bmatrix} \epsilon_1 \\\\ \epsilon_2 \\\\ \vdots \\\\ \epsilon_n \end{bmatrix}$$
