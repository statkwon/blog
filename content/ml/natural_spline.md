---
title: "Natural Spline"
date: 2021-03-12
categories:
  - "ML"
tags:
  - "Spline"
sidebar: false
---

Adding another constraint to regression spline, we can fit natural splines. We will reduce the degree beyond the boundary to $\dfrac{M-1}{2}$.

The number of parameters to fit natural splines is $(M+1)\times(K-1)-\left(\dfrac{M-1}{2}+1\right)\times2-M\times K=K$ and we can see that it is independent of $M$.

Natural Cubic Spline is the most common one.

## Natural Cubic Spline

$N_1(X)=1, \quad N_2(X)=X, \quad N_{k+2}(X)=d_k(X)-d_{K-1}(X)$

$d_k(X)=\dfrac{(X-\xi_k)\_+^3-(X-\xi_K)_+^3}{\xi_k-\xi_K}$

We will make a proof for the formula above.

$\displaystyle f(X)=\sum_{j=1}^4\beta_jX^{j-1}+\sum_{k=1}^K\theta_k(X-\xi_k)_+^3$

1\) $(-\infty, \xi_1) \quad\Rightarrow\quad f(X)=\beta_1+\beta_2X+\beta_3X^2+\beta_4X^3$

Because of the degree constraint beyond the boundary, $\beta_3$ and $\beta_4$ should be $0$.

2\) $(\xi_K, \infty) \quad\Rightarrow\quad \displaystyle f(X)=\beta_1+\beta_2X+\sum_{k=1}^K\theta_k(X^3-3X^2\xi_k-3X\xi_k^2-\xi_k^3)$

Because of the constraint beyond the boundary, $\displaystyle\sum_{k=1}^K\theta_k=0$ and $\displaystyle\sum_{k=1}^K\theta_k\xi_k=0$.

3\) $\displaystyle\theta_K=-\sum_{k=1}^{K-1}\theta_k$

4\) $\displaystyle\theta_{K-1}=\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}$

$\begin{aligned}
\sum_{k=1}^K\theta_k\xi_k&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}+\theta_K\xi_K \\\\
&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}-\sum_{k=1}^{K-1}\theta_k\xi_K \\\\
&=\sum_{k=1}^{K-2}\theta_k\xi_k+\theta_{K-1}\xi_{K-1}-\sum_{k=1}^{K-2}\theta_k\xi_K-\theta_{K-1}\xi_K=0
\end{aligned}$

$\displaystyle\Leftrightarrow\theta_{K-1}(\xi_K-\xi_{K-1})=\sum_{k=1}^{K-2}\theta_k(\xi_k-\xi_K)$

$\therefore\displaystyle\theta_{K-1}=\sum_{k=1}^{K-2}\dfrac{\theta_k(\xi_k-\xi_K)}{\xi_K-\xi_{K-1}}$

5\) $\displaystyle f(X)=\beta_1+\beta_2X+\sum_{k=1}^{K-2}\phi_k\left\\{d_k(X)-d_{K-1}(X)\right\\}$

Let $\displaystyle f(X)=\beta1+\beta_2X+\sum_{k=1}^K\theta_k(X-\xi_k)_+^3=\beta_1+\beta_2X+g(X)$.

$\begin{aligned}
g(X)&=\sum\_{k=1}^{K-1}\theta\_k(X-\xi\_k)\_+^3+\theta\_K(X-\xi\_K)\_+^3 \\\\
&=\sum\_{k=1}^{K-1}\theta\_k(X-\xi\_k)\_+^3-\sum\_{k=1}^{K-1}\theta\_k(X-\xi\_K)\_+^3 \\\\
&=\sum\_{k=1}^{K-1}\theta_k\left\\{(X-\xi\_k)\_+^3-(X-\xi\_K)\_+^3\right\\} \\\\
&=\sum\_{k=1}^{K-2}\theta\_k\left\\{(X-\xi\_k)\_+^3-(X-\xi\_K)\_+^3\right\\}+\theta\_{K-1}\left\\{(X-\xi\_{K-1})\_+^3-(X-\xi\_K)\_+^3\right\\} \\\\
&=\sum\_{k=1}^{K-2}\theta\_k\left\\{(X-\xi\_k)\_+^3-(X-\xi\_K)\_+^3\right\\}+\sum\_{k=1}^{K-2}\dfrac{\theta\_k(\xi\_k-\xi\_K)}{\xi\_K-\xi\_{K-1}}\left\\{(X-\xi\_{K-1})\_+^3-(X-\xi\_K)\_+^3\right\\} \\\\
&=\sum\_{k=1}^{K-2}\theta\_k(\xi\_k-\xi\_K)\left\\{\dfrac{(X-\xi\_k)\_+^3-(X-\xi\_K)\_+^3}{\xi\_K-\xi\_k}-\dfrac{(X-\xi\_{K-1})\_+^3-(X-\xi\_K)\_+^3}{\xi\_K-\xi\_{K-1}}\right\\} \\\\
&=\sum\_{k=1}^{K-2}\phi\_k\left\\{d\_k(X)-d\_{K-1}(X)\right\\}
\end{aligned}$

$\begin{aligned}
\therefore f(X)&=\beta\_1+\beta\_2X+g(X) \\\\
&=\beta\_1+\beta\_2X+\sum\_{k=1}^{K-2}\phi\_k\left\\{d\_k(X)-d\_{K-1}(X)\right\\}
\end{aligned}$

## Python Code for Natural Cubic Spline

Github Link: [MyNaturalCubicSpline.ipynb](https://github.com/statkwon/ML_Study/blob/master/MyNaturalCubicSpline.ipynb)

```py
import numpy as np

class MyNaturalCubicSpline:
    def fit(self, X_train, y_train, k):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.k = np.array(k)
        X_train_new = np.column_stack((np.ones(len(X_train)), X_train))
        d_Km1 = (np.where(np.power(X_train-k[-2], 3) < 0, 0, np.power(X_train-k[-2], 3))-np.where(np.power(X_train-k[-1], 3) < 0, 0, np.power(X_train-k[-1], 3)))/(k[-2]-k[-1])
        for i in range(len(k)-2):
            d = (np.where(np.power(X_train-k[i], 3) < 0, 0, np.power(X_train-k[i], 3))-np.where(np.power(X_train-k[-1], 3) < 0, 0, np.power(X_train-k[-1], 3)))/(k[i]-k[-1])
            X_train_new = np.column_stack((X_train_new, d-d_Km1))
        self.beta = np.linalg.inv(np.transpose(X_train_new).dot(X_train_new)).dot(np.transpose(X_train_new)).dot(y_train)
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        X_test_new = np.column_stack((np.ones(len(X_test)), X_test))
        d_Km1 = (np.where(np.power(X_test-self.k[-2], 3) < 0, 0, np.power(X_test-self.k[-2], 3))-np.where(np.power(X_test-self.k[-1], 3) < 0, 0, np.power(X_test-self.k[-1], 3)))/(self.k[-2]-self.k[-1])
        for i in range(len(self.k)-2):
            d = (np.where(np.power(X_test-self.k[i], 3) < 0, 0, np.power(X_test-self.k[i], 3))-np.where(np.power(X_test-self.k[-1], 3) < 0, 0, np.power(X_test-self.k[-1], 3)))/(self.k[i]-self.k[-1])
            X_test_new = np.column_stack((X_test_new, d-d_Km1))
        return X_test_new.dot(self.beta)
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
