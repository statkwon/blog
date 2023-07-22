---
title: "LDA-Classification"
date: 2021-03-05
categories:
  - "ML"
tags:
  - "LDA"
sidebar: false
---

Linear Discriminant Analysis(이하 LDA)는 Multiclass 분류 문제를 해결하기 위해 고안된 모형이다. 이 글에서는 LDA의 메커니즘을 크게 두 가지 관점으로 나누어서 정리하고 있다.

## Perspecitve of Bayes Classifier

LDA는 각 범주별 Posterior Probability를 추정하고, 해당 확률값이 가장 큰 범주로 데이터를 분류한다는 점에서 Bayes Classifier를 추정한 모형으로 볼 수 있다. 이때 LDA는 Posterior Probability를 직접적으로 추정하지 않고, Bayes' Rule을 사용하여 $f(Y\_k\vert\mathbf{x})$와 $\text{P}(Y\_k)$를 추정하는 방식을 사용한다.

$\text{P}(Y\_k\vert\mathbf{x})\approx f(\mathbf{x}\vert Y\_k)\text{P}(Y\_k)$

지금부터 편의를 위해 $\text{P}(Y\_k)$를 $\pi\_k$로 표기하도록 하겠다. LDA는 $\mathbf{x}\vert Y\_k$가 다변량 정규분포를 따르며, $k$값에 관계없이 각 분포의 공분산 행렬은 모두 $\Sigma$로 동일함을 가정한다.

$\mathbf{x}\vert Y_k\sim N(\boldsymbol{\mu}_k, \Sigma) \quad\Rightarrow\quad f(\mathbf{x}\vert Y_k)=\dfrac{1}{(2\pi)^{p/2}\vert\Sigma\vert^{1/2}}e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)}$

분류 경계선은 서로 다른 범주에 대한 Posterior Probability가 같은 지점에서 형성되므로, $\\{\mathbf{x}:\text{P}(Y_k\vert\mathbf{x})=\text{P}(Y_l\vert\mathbf{x})\\}$의 형태를 갖는다. 간단한 연산을 통해 분류 경계선의 식을 다음과 같이 변형할 수 있다.

$\begin{aligned}
&\\{\mathbf{x}:\text{P}(Y_k\vert\mathbf{x})=\text{P}(Y_l\vert\mathbf{x})\\} \\\\
&\Leftrightarrow \\{\mathbf{x}:f(\mathbf{x}\vert Y_k)\pi_k=f(\mathbf{x}\vert Y_l)\pi_l\\} \\\\
&\Leftrightarrow \\{\mathbf{x}:\log{f(\mathbf{x}\vert Y_k)}+\log{\pi_k}=\log{f(\mathbf{x}\vert Y_l)}+\log{\pi_l}\\} \\\\
&\Leftrightarrow \left\\{\mathbf{x}:-\dfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)+\log{\pi_k}=-\dfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_l)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}_l)+\log{\pi_l}\right\\} \\\\
&\Leftrightarrow \left\\{\mathbf{x}:\mathbf{x}^T\Sigma^{-1}\boldsymbol{\mu}_k-\dfrac{1}{2}\boldsymbol{\mu}_k^T\Sigma^{-1}\boldsymbol{\mu}_k+\log{\pi_k}=\mathbf{x}^T\Sigma^{-1}\boldsymbol{\mu}_l-\dfrac{1}{2}\boldsymbol{\mu}_l^T\Sigma^{-1}\boldsymbol{\mu}_l+\log{\pi_l}\right\\}
\end{aligned}$

마지막 줄의 $\delta_k(\mathbf{x})=\mathbf{x}^T\Sigma^{-1}\boldsymbol{\mu}_k-\dfrac{1}{2}\boldsymbol{\mu}_k^T\Sigma^{-1}\boldsymbol{\mu}_k+\log{\pi_k}$를 Discriminant Function이라고 부른다. 당연하게도, Posterior Probability가 가장 큰 범주로 데이터를 분류하는 것은 곧 $\delta_k(\mathbf{x})$의 값이 가장 큰 범주로 데이터를 분류하는 것과 같다.

여기서 우리는 모수 $\boldsymbol{\mu}_k$, $\Sigma$, $\pi_k$의 값을 알지 못한다. 하지만 앞서 정규분포를 가정하였으므로, MLE를 사용하여 이 값들을 추정할 수 있다. 따라서 추정된 Discriminant Function을 $\hat{\delta}_k(\mathbf{x})=\mathbf{x}^T\hat{\Sigma}^{-1}\hat{\boldsymbol{\mu}}_k-\dfrac{1}{2}\hat{\boldsymbol{\mu}}_k^T\hat{\Sigma}^{-1}\hat{\boldsymbol{\mu}}_k+\log{\hat{\pi}_k}$로 나타낼 수 있다.

## Perspective of Distance between Data and Centroid

{{<figure src="/ml/lda-cls1.png" width="300">}}

앞서 설정한 것과 동일한 가정 하에, 데이터와 각 분포의 중심 사이의 거리를 기준으로 그 거리가 가장 가까운 범주로 데이터를 분류하는 방식을 생각해볼 수 있다. 이때 거리를 측정하는 척도로 Mahalanobis 거리를 사용할 경우, 분류 경계선은 다음 식과 같이 각 중심과의 거리가 같은 지점에서 형성된다.

$\left\\{\mathbf{x}:\sqrt{(\mathbf{x}-\boldsymbol{\mu}_k)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)}=\sqrt{(\mathbf{x}-\boldsymbol{\mu}_l)^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}_l)}\right\\}$

물론 위 식은 각 범주에 속할 확률 $\pi_k$가 모두 동일한 경우에 해당한다. 설령 확률이 서로 같지 않더라도, 각각의 값에 비례하는 수치를 양변에 더해줌으로써 일반화할 수 있다. 이제 우리는 이 식과 'Perspective of Bayes Classifier'에서 구한 분류 경계선의 식을 비교함으로써, LDA가 데이터와 분포의 중심 사이의 거리에 기반한 분류 모형이라는 사실을 확인할 수 있다. 이는 분류하려는 데이터와 특정 범주의 분포 사이의 거리가 가까울 수록 해당 범주로 분류될 확률이 높아진다는 점에서 상당히 직관적인 방식이라고 할 수 있다.

## Whitening Transformation for LDA

주어진 데이터에 Whitening Transformation(또는 Sphering Transformation)을 적용함으로써 Discriminant Function을 추정하기 위환 계산 과정을 보다 간소화할 수 있다. $W^TW=\hat{\Sigma}^{-1}$를 만족하는 Whitening Matrix $W$를 구하는 방법은 여러가지지만, 이 글에서는 $\hat{\Sigma}$에 대한 Eigen Decomposition을 사용하여 $W$를 구하는 방법에 대해 다룬다.

$\hat{\Sigma}=VDV^T=VD^{1/2}D^{1/2}V^T$

위와 같은 식을 통해 $W=D^{-1/2}V^T$인 경우 $W^TW=\hat{\Sigma}^{-1}$를 만족함을 어렵지 않게 확인할 수 있다. 따라서 Data Matrix $X$에 $D^{-1/2}V^T$를 곱함으로써 해당 데이터가 Identity Matrix를 공분산 행렬로 갖는 분포로부터 생성된 것처럼 변형할 수 있다. 즉, Sphered된 Data Matrix를 $X^*=XD^{-1/2}V^T$로 나타낼 수 있다. $X^\*$를 사용하여 추정한 Discriminant Function의 식은 다음과 같다.

$\hat{\delta}_k(\mathbf{x}^\*)={\mathbf{x}^\*}^T\mathbf{x}^\*-\dfrac{1}{2}{\hat{\boldsymbol{\mu}}_k^\*}^T\hat{\boldsymbol{\mu}}_k^\*+\log{\hat{\pi}_k}$

기존의 식과 비교했을 때 공분산 행렬이 생략된 형태로 식이 보다 간소화되었음을 확인할 수 있다.

또한 이러한 변환을 데이터와 분포 사이의 거리의 관점에서 생각해보면, 각 분포의 공분산 행렬이 Identity Matrix로 변환되는 것이므로 변형된 공간에서의 Mahalnobis 거리는 곧 Euclidean 거리와 일치함을 알 수 있다.

(Whitening Transformation에 대한 자세한 설명이 필요한 경우 [이 글](/ml/standardizing_and_whitening)을 참조)

## How to Get Nonlinear Boundaries

QDA, FDA, RDA

## Python Code for LDA

Github Link: [Python_Code_for_LDA.ipynb](https://github.com/statkwon/ML_Study/blob/master/Python_Code_for_LDA.ipynb)

```py
import warnings
import numpy as np

def _class_means(X, y):
    classes = np.unique(y)
    means = np.array([np.mean(X[y == i], axis=0) for i in classes])
    return means

def _class_cov(X, y, priors):
    _, p = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    cov = np.zeros((p, p))
    cov_ = np.array([np.cov(X[y == i], rowvar=False) for i in classes])
    for i in range(n_classes):
        cov += priors[i] * cov_[i]
    return cov

def softmax(X):
    X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

class LinearDiscriminantAnalysis:
    """My Linear Discriminant Analysis"""
    
    def __init__(self, priors=None):
        self.priors = priors
    
    def _solve(self, X, y):
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_)
        self.coef_ = self.means_.dot(np.linalg.inv(self.covariance_))
        self.intercept_ = -0.5 * np.diag(self.means_.dot(self.coef_.T)) + np.log(self.priors_)
    
    def fit(self, X, y):
        X = np.array(X) ; y = np.array(y)
        self.classes_ = np.unique(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)
        
        if n_samples == n_classes:
            raise ValueError('데이터의 개수는 범주의 개수보다 많아야합니다.')
        
        if self.priors is None:
            self.priors_ = np.array([sum(y == i) / len(y) for i in self.classes_])
        else:
            self.priors_ = np.array(self.priors)
        
        if any(self.priors_ < 0):
            raise ValueError('사전 확률은 0보다 커야합니다.')
        if not np.isclose(sum(self.priors_), 1):
            warnings.warn('사전 확률의 합이 1이 아닙니다. 값을 재조정합니다', UserWarning)
            self.priors_ = self.priors_ / sum(self.priors_)
        
        self._solve(X, y)
        
        return self
    
    def decision_function(self, X):
        X = np.array(X)
        scores = X.dot(self.coef_.T) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores
    
    def predict(self, X):
        decision = self.decision_function(X)
        y_pred = self.classes_.take(decision.argmax(1))
        return y_pred
    
    def predict_proba(self, X):
        decision = self.decision_function(X)
        return softmax(decision)
    
    def predict_log_proba(self, X):
        prediction = self.predict_proba(X)
        prediction[prediction == 0] += np.finfo(prediction.dtype).tiny
        return np.log(prediction)
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. [https://en.wikipedia.org/wiki/Mahalanobis_distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
3. [https://en.wikipedia.org/wiki/Whitening_transformation](https://en.wikipedia.org/wiki/Whitening_transformation)
