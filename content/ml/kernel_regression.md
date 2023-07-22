---
title: "Kernel Regression"
date: 2021-03-17
categories:
  - "ML"
tags:
  - "Kernel"
sidebar: false
---

Kernel smoothing is a meomry-based method to add a flexibility in estimating the regression function by fitting a different but simple model seperately at each query point $x_0$ with only those observations close to the target point $x_0$. This localization is achieved via kernel $K_\lambda(\mathbf{X}_0, \mathbf{X})$, which assigns a weight to $\mathbf{X}$ based on its distance from $\mathbf{X}_0$.

## Kernel Functions

$K\_\lambda(\mathbf{X}\_0, \mathbf{X})=D\left(\dfrac{d(\mathbf{X}\_0, \mathbf{X})}{h\_\lambda(\mathbf{X}\_0)}\right)$

Kernel function can be divided by three parts: distance function, width function, and weighting function. A few famous distances such as euclidean distance or mahalanobis distance can be selected for the distance function. There are two types of width, metric window width $h\_\lambda(\mathbf{X}\_0)=\lambda$, which is just a constant and nearest-neighbor window width $h\_k(\mathbf{X}\_0)=d(\mathbf{X}\_0, \mathbf{X}\_{[k]})$, where $\mathbf{X}\_{[k]}$ is the $k$th closest point to $\mathbf{X}\_0$. Lastly, there exist several candidates of weighting function $D(t)$. We will only cover about some of them below.

1\) Epanechnikov: $D(t)=\begin{cases} \dfrac{3}{4}(1-t)^2 & \vert t\vert\leq1 \\\\ 0 & \text{o.w.} \end{cases}$

2\) Tri-Cube: $D(t)=\begin{cases} (1-\vert t\vert^3)^3 & \vert t\vert\leq1 \\\\ 0 & \text{o.w.} \end{cases}$

3\) Gaussian: $D(t)=\dfrac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}t^2}, \quad -\infty<t<\infty$

4\) Uniform: $D(t)=\begin{cases} \dfrac{1}{2} & \vert t\vert\leq1 \\\\ 0 & \text{o.w.} \end{cases}$

## Nadaraya-Watson Kernel Regression

Nadaraya-Watson kernel regression estimates $\hat{f}(\mathbf{X}\_0)$ with a weighted average of the observations close to $\mathbf{X}\_0$. Here the weights are determined by the distances of the observations from $\mathbf{X}\_0$.

$\hat{f}(\mathbf{X}\_0)=\dfrac{\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)y\_i}{\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)}$

This can be interpreted as using the kernel density estimation for $f(\mathbf{X}\_0)$ and $f(\mathbf{X}\_0, y\_0)$ to estimate $\text{E}[Y\_0\vert\mathbf{X}\_0]$.

$\begin{aligned}
\text{E}[Y\_0\vert\mathbf{X}\_0]&=\int y\_0\dfrac{f(\mathbf{X}\_0, y)}{f(\mathbf{X}\_0)}dy\_0 \\\\
&=\int\dfrac{y\_0\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)K\_\lambda(y\_0, y\_i)}{\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)}dy\_0 \\\\
&=\dfrac{\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)\int y\_0K\_\lambda(y\_0, y\_i)dy\_0}{\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)} \\\\
&=\dfrac{\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)y\_i}{\sum\_{i=1}^nK\_\lambda(\mathbf{X}\_0, \mathbf{X}\_i)}
\end{aligned}$

The continuity of $\hat{f}(\mathbf{X})$ depends on which kernel to use. If we use the continuous kernel like epanechnikov kernel, the result curve will be smooth, while the result for the discontinuous kernel like nearest-neighbor kernel is rigid.

We need to select an appropriate value for $\lambda$(or $k$) because it has a decisive effect on the bias and variance of the model. As $\lambda$ becomes larger, the variance of the model gets lower, while the bias gets higher. Thus, we call $\lambda$ as a smoothing parameter.

Usually metric window width tends to keep the bias of the estimate constant, but the variance is inversely proportional to the local density. Nearest-neighbor window width exhibits the opposite behavior; the variance stays constant and the absolute biase varies inversely with local density. These are intuitive results because the number of points included in the metric window is proportional to the local density, while the length of nearest-neighbor window is inversely proportional to the local density.

However, both methods suffers from boundary issues. Local density around the boundaries is naturally low and this makes the metric window to contain less points and the nearest-neighbor window to get wider. Also, locally weighted averages can be badly biased on the boundaries of the domain, because of the asymmetry of the kernel in that region. This bias can also occur in the interior of the domain, if the $\mathbf{X}$ values are not equally spaced, but is usually less severe.

## Python Code for Nadaraya-Watson Kernel Regression

Github Link: [MyNWKernelRegression.ipynb](https://github.com/statkwon/ML_Study/blob/master/MyNWKernelRegression.ipynb)

```py
import math
import numpy as np

class MyNWKernelRegression:
    def __init__(self, kernel='Epanechnikov', width=10):
        self.kernel = kernel
        self.width = width
        
    def epanechnikov(self, x):
        return np.where(abs(x) <= 1, 0.75*(1-x**2), 0)
    
    def tricube(self, x):
        return np.where(abs(x) <= 1, (1-abs(x)**3)**3, 0)
    
    def gaussian(self, x):
        return 1/np.sqrt(2*math.pi)*np.exp(-0.5*(x**2))
    
    def uniform(self, x):
        return np.where(abs(x) <= 1, 0.5, 0)
    
    def predict(self, X_train, y_train, X_test):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_pred = np.array([])
        for i in range(len(X_test)):
            if self.kernel == 'KNN':
                t = abs(X_train-X_test[i])/abs(X_train-X_test[i])[np.argsort(abs(X_train-X_test[i]))==self.width][0]
                d = self.uniform(t)
            else:
                t = abs(X_train-X_test[i])/self.width
                if self.kernel == 'Epanechnikov':
                    d = self.epanechnikov(t)
                elif self.kernel == 'Tri-Cube':
                    d = self.tricube(t)
                else:
                    d = self.gaussian(t)
            y_pred = np.append(y_pred, np.sum(d*y_train)/np.sum(d))
        return y_pred
```

---

**Reference**

1. Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.
2. [https://en.wikipedia.org/wiki/Kernel_regression](https://en.wikipedia.org/wiki/Kernel_regression)
