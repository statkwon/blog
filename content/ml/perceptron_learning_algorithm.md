---
title: "Perceptron Learning Algorithm"
date: 2021-04-18
categories:
  - "ML"
tags:
  - "Perceptron"
sidebar: false
---

Perceptron Learning Algorithm은 서로 다른 Class에 속한 데이터를 가장 잘 구분하는 Hyperplane을 찾기 위한 알고리즘이다. 이 글에서는 $y\in\\{-1, 1\\}$의 Binary Outcome에 대한 Classification 문제를 가정한다.

{{<figure src="/ml/pla1.png" width="400">}}

Perceptron Learning Algorithm은 오분류된 데이터와 Decision Boundary 사이의 거리를 최소화하는 Hyperplane을 찾는 방식으로 작동한다. 일반적으로 $p$차원 공간 속의 Hyperplane은 다음과 같의 정의된다.

$L=\\{\mathbf{x}\in\mathbb{R}^p:f(\mathbf{x})=0\\}$, where $f(\mathbf{x})=\beta_0+\boldsymbol{\beta}^T\mathbf{x}$

따라서 오분류된 데이터와 Hyperplane 사이의 거리를 $d_i$라고 할 때, 우리가 찾는 Hyperplane은

$\displaystyle \underset{f}{\text{argmin}}\sum_{i\in\mathcal{M}}d_i$, where $\mathcal{M}$ indexes the set of misclassified points

라고 표현할 수 있다. 하지만 실제로 최적해를 구할 때는 $d_i$ 대신 그 값이 거리에 비례하면서 계산 비용이 더 적은 $-y_if(\mathbf{x}_i)$를 사용할 수 있다.

$\displaystyle \underset{\boldsymbol{\beta}, \beta_0}{\text{argmin}}\sum_{i\in\mathcal{M}}-y_i(\beta_0+\boldsymbol{\beta}^T\mathbf{x}_i)$, where $\mathcal{M}$ indexes the set of misclassified points

간단한 Linear Algebra를 통해 $-y_if(\mathbf{x}_i)$가 $\mathbf{x}_i$와 Decision Boundary(Hyperplane) 사이의 거리에 비례한다는 것을 쉽게 확인할 수 있다.

{{<figure src="/ml/pla2.png" width="400">}}

$\boldsymbol{\beta}$가 Hyperplane에 Orthogonal한 벡터라는 것은 자명한 사실이다. 따라서 $\boldsymbol{\beta}^*=\boldsymbol{\beta}/\Vert\boldsymbol{\beta}\Vert$라고 할 때, 임의의 $\mathbf{x}\in\mathbb{R}^p$와 $\mathbf{x}_0\in L$에 대해서 $\mathbf{x}$와 Hyperplane $L$ 사이의 거리를 다음과 같이 나타낼 수 있다.

$\begin{aligned}
{\boldsymbol{\beta}^*}^T(\mathbf{x}-\mathbf{x}_0)&=\dfrac{1}{\Vert\boldsymbol{\beta}\Vert}\boldsymbol{\beta}^T\mathbf{x}-\dfrac{1}{\Vert\boldsymbol{\beta}\Vert}\boldsymbol{\beta}^T\mathbf{x}_0 \\\\
&=\dfrac{1}{\Vert\boldsymbol{\beta}\Vert}\boldsymbol{\beta}^T\mathbf{x}+\dfrac{1}{\Vert\boldsymbol{\beta}\Vert}\beta_0 \quad (\because\mathbf{x}_0\in L) \\\\
&=\dfrac{1}{\Vert\boldsymbol{\beta}\Vert}f(\mathbf{x}) \\\\
&=\dfrac{1}{\Vert f'(\mathbf{x})\Vert}f(\mathbf{x})
\end{aligned}$

이때, $f(\mathbf{x})$는 실제 $y$가 $1$인 경우 음수의 값을, $-1$인 경우 양수의 값을 가지므로, $f(\mathbf{x})$에 $-y$를 곱해주면 그 값이 항상 양수가 되어 $\mathbf{x}$와 Hyperplane 사이의 거리에 비례한다고 말할 수 있다.

이제 앞서 정의한 문제의 최적해 $\boldsymbol{\beta}^\*$, $\beta_0^*$를 구함으로써 우리가 원하는 Hyperplane을 찾을 수 있다. Perceptron Learning Algorithm은 최적해를 구하기 위한 방법으로 다음과 같은 Stochastic Gradient Descent 알고리즘을 사용한다.

$\begin{pmatrix} \boldsymbol{\beta}^{(k+1)} \\\\ \beta_0^{(k+1)} \end{pmatrix}=\begin{pmatrix} \boldsymbol{\beta}^{(k)} \\\\ \beta_0^{(k)} \end{pmatrix}+\rho\begin{pmatrix} y_i\mathbf{x}_i \\\\ y_i \end{pmatrix}$

## Some Critical Issues of Algorithm

Perceptron Learning Algorithm은 굉장히 단순한 형태의 알고리즘이기 때문에 몇 가지 문제점을 가지고 있다.

1. Stochastic Gradient Descent 알고리즘의 Initial Point에 따라 Hyperplane의 형태가 바뀐다. (해가 유일하지 않다.)
2. 데이터가 Linear Decision Boundary를 사용하여 완벽하게 분할이 불가능한 경우 알고리즘이 수렴하지 않는다.
3. 상황에 따라 알고리즘이 수렴하기까지 굉장히 오랜 시간이 소요될 수 있다.

이후 첫 번째 문제점을 해결하기 위해 [Optimal Seperating Hyperplane](/ml/optimal_seperating_hyperplanes)이, 두 번째 문제점을 해결하기 위해 [Support Vector Classifier](/ml/support_vector_classifier) 등과 같은 알고리즘이 탄생하였다.

## Python Code for Perceptron Learning Algirhtm

Github Link: [MyPerceptronLearningAlgorithm.ipynb](https://github.com/statkwon/ML_Study/blob/master/MyPerceptronLearningAlgorithm.ipynb)

```py
import numpy as np

class MyPerceptronLearningAlgorithm:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
    
    def fit(self, X_train, y_train, lr):
        ones = np.transpose(np.array([1]*len(X_train)))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        beta = np.array([0]*X_train.shape[1])
        beta0 = 0
        status = True ; tmp = 1
        while status:
            status = False
            yhat = np.array([-1 if i < 0 else 1 for i in (X_train.dot(beta) + ones.dot(beta0))])
            for i in range(len(X_train)):
                if y_train[i] != yhat[i]:
                    beta = beta + lr*y_train[i]*X_train[i]
                    beta0 = beta0 + lr*y_train[i]
                    tmp += 1
                    status = True
            if tmp > len(X_train)*self.max_iter:
                break
        self.beta_new = beta
        self.beta0_new = beta0
        
    def predict(self, X_test):
        ones = np.transpose(np.array([1]*len(X_test)))
        X_test = np.array(X_test)
        return np.array([-1 if i < 0 else 1 for i in (X_test.dot(self.beta_new) + self.beta0_new)])
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
