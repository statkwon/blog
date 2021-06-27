---
title: "Nadaraya-Watson Kernel Regression"
date: 2021-03-17
weight: 12
---

Kernel smoothing is a meomry-based method to add a flexibility in estimating the regression function by fitting a different but simple model seperately at each query point $x_0$ with only those observations close to the target point $x_0$. This localization is achieved via kernel $K_\lambda(\mathbf{X}_0, \mathbf{X})$, which assigns a weight to $\mathbf{X}$ based on its distance from $\mathbf{X}_0$.

**Kernel Functions**

$K_\lambda(\mathbf{X}_0, \mathbf{X})=D\left(\dfrac{d(\mathbf{X}_0, \mathbf{X})}{h_\lambda(\mathbf{X}_0)}\right)$

Kernel function can be divided by three parts: distance function, width function, and weighting function. A few famous distances such as euclidean distance or mahalanobis distance can be selected for the distance function. There are two types of width, metric window width $h_\lambda(\mathbf{X}_0)=\lambda$, which is just a constant and nearest-neighbor window width $h_k(\mathbf{X}_0)=d(\mathbf{X}_0, \mathbf{X}\_{[k]})$, where $\mathbf{X}_{[k]}$ is the $k$th closest point to $\mathbf{X}_0$. Lastly, there exist several candidates of weighting function $D(t)$. We will only cover about some of them below.

1\) Epanechnikov: $D(t)=\begin{cases} \dfrac{3}{4}(1-t)^2 & \vert t\vert≤1 \\\\ 0 & \text{o.w.} \end{cases}$

2\) Tri-Cube: $D(t)=\begin{cases} (1-\vert t\vert^3)^3 & \vert t\vert≤1 \\\\ 0 & \text{o.w.} \end{cases}$

3\) Gaussian: $D(t)=\dfrac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}t^2}, \quad -\infty<t<\infty$

4\) Uniform: $D(t)=\begin{cases} \dfrac{1}{2} & \vert t\vert≤1 \\\\ 0 & \text{o.w.} \end{cases}$

---

**Nadaraya-Watson Kernel Regression**

Nadaraya-Watson kernel regression estimates $\hat{f}(\mathbf{X}_0)$ with a weighted average of the observations close to $\mathbf{X}_0$. Here the weights are determined by the distances of the observations from $\mathbf{X}_0$.

$\hat{f}(\mathbf{X}\_0)=\dfrac{\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)y_i}{\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)}$

This can be interpreted as using the kernel density estimation for $f(\mathbf{X}_0)$ and $f(\mathbf{X}_0, y_0)$ to estimate $\text{E}[Y_0\vert\mathbf{X}_0]$.

$\begin{aligned}
\text{E}[Y_0\vert\mathbf{X}\_0]&=\int y_0\dfrac{f(\mathbf{X}\_0, y)}{f(\mathbf{X}\_0)}dy_0 \\\\
&=\int\dfrac{y_0\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)K_\lambda(y_0, y_i)}{\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)}dy_0 \\\\
&=\dfrac{\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)\int y_0K_\lambda(y_0, y_i)dy_0}{\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)} \\\\
&=\dfrac{\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)y_i}{\sum_{i=1}^nK_\lambda(\mathbf{X}_0, \mathbf{X}_i)}
\end{aligned}$

The continuity of $\hat{f}(\mathbf{X})$ depends on which kernel to use. If we use the continuous kernel like epanechnikov kernel, the result curve will be smooth, while the result for the discontinuous kernel like nearest-neighbor kernel is rigid.

We need to select an appropriate value for $\lambda$(or $k$) because it has a decisive effect on the bias and variance of the model. As $\lambda$ becomes larger, the variance of the model gets lower, while the bias gets higher. Thus, we call $\lambda$ as a smoothing parameter.

Usually metric window width tends to keep the bias of the estimate constant, but the variance is inversely proportional to the local density. Nearest-neighbor window width exhibits the opposite behavior; the variance stays constant and the absolute biase varies inversely with local density. These are intuitive results because the number of points included in the metric window is proportional to the local density, while the length of nearest-neighbor window is inversely proportional to the local density.

However, both methods suffers from boundary issues. Local density around the boundaries is naturally low and this makes the metric window to contain less points and the nearest-neighbor window to get wider. Also, locally weighted averages can be badly biased on the boundaries of the domain, because of the asymmetry of the kernel in that region.

---

**Reference**

1. Elements of Statistical Learning
2. [https://en.wikipedia.org/wiki/Kernel_regression](https://en.wikipedia.org/wiki/Kernel_regression)