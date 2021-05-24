---
title: "Nadaraya-Watson Kernel Regression"
---

$$K_\lambda(x_0, x)=D\left(\dfrac{\vert x-x_0\vert}{h_\lambda(x_0)}\right)$$

**1\) Width Function** $h_\lambda(x_0)$

Metric Window Width: $\lambda$

Nearest-Neighbor Window Width: $\vert x_0-x_{[k]}\vert$

**2\) Weighting Function** $D(t)$

Epanechnikov: $D(t)=\begin{cases} \dfrac{3}{4}(1-t)^2 & \vert t\vert≤1 \\\\ 0 & \text{o.w.} \end{cases}$

Tri-Cube: $D(t)=\begin{cases} (1-\vert t\vert^3)^3 & \vert t\vert≤1 \\\\ 0 & \text{o.w.} \end{cases}$

Gaussian: $D(t)=\dfrac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}t^2}, \quad -\infty<t<\infty$

Uniform: $D(t)=\begin{cases} \dfrac{1}{2} & \vert t\vert≤1 \\\\ 0 & \text{o.w.} \end{cases}$

**Nadaraya-Watson Kernel Regression**

$\displaystyle\hat{f}(x_0)=\dfrac{\sum_{i=1}^nK_\lambda(x_0, x_i)y_i}{\sum_{i=1}^nK_\lambda(x_0, x_i)}$

---

**Reference**

1. Elements of Statistical Learning