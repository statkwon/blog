---
title: "Regression Spline"
date: 2021-03-10
categories:
  - "ML"
tags:
  - "Spline"
sidebar: false
---

We can obtain regression splines by adding a continuity constraint to piecewise polynomials.

Regression splines are often called the $M$th order spline, a piecewise polynomial of degree $M$, that is continuous and has continuous derivatives of orders $1, \ldots, M-1$ at its know points.

## $M$th order spline with $K$ knots

$h_j(X)=X^{j-1} \quad (j=1, 2, \ldots, M+1)$

$h_{M+1+l}(X)=(X-\xi_l)_+^M \quad (l=1, 2, \ldots, K)$

$f(X)=\beta\_1+\beta\_2X\cdots+\beta\_{M+1}X^M+\beta\_{M+2}(X-\xi\_1)\_+^M+\cdots+\beta\_{M+K+1}(X-\xi\_K)\_+^M$

We will make a proof for the case when $M=3$ and $K=2$.

$f(X)=\beta_1+\beta_2X+\beta_3X^2+\beta_4X^3+\beta_5(X-\xi_1)_+^3+\beta_6(X-\xi_2)\_+^3$

We will show the continuity of $f(X)$, $f'(X)$, $f''(X)$ at the knots $\xi_1$, $\xi_2$ ($\xi_1<\xi_2$).

1\) Continuity of $f(X)$

$\begin{aligned}
f(\xi_1-h)&=\beta_1+\beta_2(\xi_1-h)+\beta_3(\xi_1-h)^2+\beta_4(\xi_1-h)^3+\beta_5(\xi_1-h-\xi_1)_+^3+\beta_6(\xi_1-h-\xi_2)\_+^3 \\\\
&=\beta_1+\beta_2(\xi_1-h)+\beta_3(\xi_1-h)^2+\beta_4(\xi_1-h)^3
\end{aligned}$

$f(\xi_1+h)=\beta_1+\beta_2(\xi_1+h)+\beta_3(\xi_1+h)^2+\beta_4(\xi_1+h)^3+\beta_5(\xi_1+h-\xi_1)_+^3+\beta_6(\xi_1+h-\xi_2)\_+^3$

$\displaystyle\lim_{h\rightarrow0}f(\xi_1-h)=\lim_{h\rightarrow0}f(\xi_1+h)=\beta_1+\beta_2\xi_1+\beta_3\xi_1^2+\beta_4\xi_1^3$

$\displaystyle\therefore\lim_{x\rightarrow\xi_1}f(x)=\beta_1+\beta_2\xi_1+\beta_3\xi_1^2+\beta_4\xi_1^3=f(\xi_1)$

$f(X)$ is continuous at $\xi_1$ and we can show that $f(X)$ is continuous at $\xi_2$ by a similar way.

2\) Continuity of $f'(X)$ at $\xi_1$, $\xi_2$

$\begin{aligned}
f'(\xi\_1^-)&=\lim\_{h\rightarrow0}\dfrac{f(\xi\_1)-f(\xi\_1-h)}{h} \\\\
&=\lim\_{h\rightarrow0}\dfrac{\beta\_2h+2\beta\_3\xi\_1h+3\beta\_4\xi\_1^2h+O(h^2)}{h} \\\\
&=\beta\_2+2\beta\_3\xi\_1+3\beta\_4\xi\_1^2
\end{aligned}$

$\begin{aligned}
f'(\xi\_1^+)&=\lim\_{h\rightarrow0}\dfrac{f(\xi\_1+h)-f(\xi\_1)}{h} \\\\
&=\lim\_{h\rightarrow0}\dfrac{\beta\_2h+2\beta\_3\xi\_1h+3\beta\_4\xi\_1^2h+\beta\_5(\xi\_1+h-\xi\_1)\_+^3+\beta\_6(\xi\_1+h-\xi\_2)\_+^3+O(h^2)}{h} \\\\
&=\beta\_2+2\beta\_3\xi\_1+3\beta\_4\xi\_1^2
\end{aligned}$

$\therefore\lim_{x\rightarrow\xi_1}f'(x)=\beta_2+2\beta_3\xi_1+3\beta_4\xi_1^2=f'(\xi_1)$

$f'(X)$ is continuous at $\xi_1$ and we can show that $f'(X)$ is continuous at $\xi_2$ by a similar way.

3\) Continuity of $f''(X)$ at $\xi_1$, $\xi_2$

Similarly, $\lim_{x\rightarrow\xi_1}f''(x)=6\beta_4\xi_1^2=f''(\xi_1)$ and $\lim_{x\rightarrow\xi_2}f''(x)=6\beta_4\xi_2^2=f''(\xi_2)$.

We showed that $f(X)$, $f'(X)$ and $f''(X)$ are all continuous at $\xi_1$, $\xi_2$. Thus, $f(X)$ represents a cubic spline with two knots.

The number of parameters to fit regression spline is $(M+1)\times(K+1)-M\times K=M+K+1$.

However, regression spline still has a problem that adding a continuity constraint cannot fix the irregularity beyond the boundaries.

## Python Code for Regression Spline

Github Link: [MyRegressionSpline.ipynb](https://github.com/statkwon/ML_Study/blob/master/MyRegressionSpline.ipynb)

```py
import numpy as np

class MyRegressionSpline:
  def fit(self, X_train, y_train, m, k):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    self.m = m
    self.k = np.array(k)
    X_train_new = np.power(X_train, 0)
    for i in range(1, m+1):
      X_train_new = np.column_stack((X_train_new, np.power(X_train, i)))
    for i in range(len(k)):
      X_train_new = np.column_stack((X_train_new, np.where(np.power(X_train-k[i], m) < 0, 0, np.power(X_train-k[i], m))))
    self.beta = np.linalg.inv(np.transpose(X_train_new).dot(X_train_new)).dot(np.transpose(X_train_new)).dot(y_train)
  
  def predict(self, X_test):
    X_test = np.array(X_test)
    X_test_new = np.power(X_test, 0)
    for i in range(1, self.m+1):
      X_test_new = np.column_stack((X_test_new, np.power(X_test, i)))
    for i in range(len(self.k)):
      X_test_new = np.column_stack((X_test_new, np.where(np.power(X_test-self.k[i], self.m) < 0, 0, np.power(X_test-self.k[i], self.m))))
    return X_test_new.dot(self.beta)
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
