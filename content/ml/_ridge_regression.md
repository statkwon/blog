---
title: "Ridge Regression"
date: 2021-02-25
categories:
  - "ML"
tags:
  - "Regularization"
sidebar: false
---

## Ridge Regression

Subset selection methods can sometimes cause high variance due to its discrete characteristic. As an alternative, shrinkage methods such as ridge regression can be used.

Ridge regression shrinks the regression coefficients by imposing a penalty on their size. The ridge solutions are not equivariant under scaling of the inputs, and so one normally standardizes the inputs. So until now we will assume that $X$ is a standardized matrix. Coefficients of ridge regression is related to the restricted minimization problem as below.

$\begin{aligned}
\hat{\beta}^{\text{ridge}}&=\underset{\beta}{\text{argmin}}\left\\{\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2+\lambda\sum_{j=1}^p\beta_j^2\right\\} \\\\
&=\underset{\beta}{\text{argmin}}\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2 \\; \text{subject to} \\; \sum_{j=1}^p\beta_j^2≤t
\end{aligned}$

We can solve this problem with a matrix notation and get the ridge coefficient as $\hat{\beta}^{\text{ridge}}=(X^TX+\lambda I)^{-1}X^Ty$.

$\begin{aligned}
\dfrac{\partial}{\partial\beta}(y-X\beta)^T(y-X\beta)+\lambda\beta^T\beta&=\dfrac{\partial}{\partial\beta}y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta+\lambda\beta^T\beta \\\\
&=-2X^Ty+2X^TX\beta+2\lambda\beta
\end{aligned}$

Here, $X^TX+\lambda I$ is always invertible for any $\lambda>0$ because the positive semi-definite matrix $X^TX$ becomes positive definite when a positive constant is added to the diagonal of $X^T$. Therefore, we can express our fitted value $\hat{y}$ as $X\hat{\beta}^{\text{ridge}}=X(X^TX+\lambda I)^{-1}X^Ty$.

## Geometric Interpretation of Ridge Regression

We can make some geometric interpretation using $X=UDV^T$, the singular value decomposition of $X$. Let's think about the case of linear regerssion first.

$\begin{aligned}
\hat{y}&=X(X^TX)^{-1}X^Ty \\\\
&=UDV^T(VDU^TUDV^T)^{-1}VDU^Ty \\\\
&=UDV^T(V^T)^{-1}(D^2)^{-1}V^{-1}VDU^Ty \\\\
&=UU^Ty \\\\
&=\sum_{i=1}^pu_iu_i^Ty \\\\
&=\dfrac{y\cdot u_1}{u_1\cdot u_1}u_1+\cdots+\dfrac{y\cdot u_p}{u_p\cdot u_p}u_p
\end{aligned}$

This is same as to project $y$ onto each vector in the orthogonal basis of column space of $X$ and make sum of them.

{{<figure src="/ml/ridge1.png" width="300">}}

Now we will apply this concept to ridge regression.

$\begin{aligned}
\hat{y}&=X(X^TX+\lambda I)^{-1}X^Ty \\\\
&=UDV^T(VDU^TUDV^T+\lambda I)^{-1}VDU^Ty \\\\
&=UDV^T(VD^2V^T+\lambda I)^{-1}VDU^Ty \\\\
&=UD\\{V^{-1}(VD^2V^T+\lambda I)(V^T)^{-1}\\}DU^Ty \\\\
&=U(D^2+\lambda I)^{-1}U^Ty \\\\
&=\sum_{i=1}^pu_i\dfrac{d_i^2}{d_i^2+\lambda}u_i^Ty \\\\
&=\dfrac{d_1^2}{d_1^2+\lambda}\dfrac{y\cdot u_1}{u_1\cdot u_1}y+\cdots+\dfrac{d_p^2}{d_p^2+\lambda}\dfrac{y\cdot u_p}{u_p\cdot u_p}y
\end{aligned}$

Ridge regression is similar to the linear regression, but differs in that it makes a shrinkage to the direction of each $u_i$ by the amount of $\dfrac{d_i^2}{d_i^2+\lambda}$. Thus, it means that a greater amount of shrinkage is applied to the coordinates of basis vectors with smaller $d_j^2$.

## Python Code for Ridge Regression

Github Link: [MyRidgeRegression.ipynb](https://github.com/statkwon/ML_Study/blob/master/MyRidgeRegression.ipynb)

```py
import numpy as np

class MyRidgeRegerssion:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X_train, y_train):
        ones = np.ones(len(X_train))
        X_train = np.array(X_train)
        X_train = np.column_stack((np.ones(len(X_train)), X_train))
        y_train = np.array(y_train)
        self.beta = np.linalg.inv(np.transpose(X_train).dot(X_train)+self.alpha*np.identity(X_train.shape[1])).dot(np.transpose(X_train)).dot(y_train)
        
    def predict(self, X_test):
        ones = np.ones(len(X_test))
        X_test = np.array(X_test)
        X_test = np.column_stack((np.ones(len(X_test)), X_test))
        return X_test.dot(self.beta)
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
