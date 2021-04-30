---
title: "Ridge Regression"
draft: false
---

Subset selection methods can sometimes cause high variance due to its discrete characteristic. As an alternative, shrinkage methods such as ridge regression can be used.

Ridge regression shrinks the regression coefficients by imposing a penalty on their size. The ridge solutions are not equivariant under scaling of the inputs, and so one normally standardizes the inputs.

$\begin{aligned}
\hat{\beta}^{\text{ridge}}&=\underset{\beta}{\text{argmin}}\left\\{\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2+\lambda\sum_{j=1}^p\beta_j^2\right\\} \\\\
&=\underset{\beta}{\text{argmin}}\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2 \\; \text{subject to} \\; \sum_{j=1}^p\beta_j^2≤t
\end{aligned}$

We can solve this problem with a matrix notation and get the ridge coefficient as $\hat{\beta}^{\text{ridge}}=(X^TX+\lambda I)^{-1}X^Ty$.

{{<rawhtml>}}
<details>
<summary>Proof</summary>
$\begin{aligned}
\dfrac{\partial}{\partial\beta}(y-X\beta)^T(y-X\beta)+\lambda\beta^T\beta&=\dfrac{\partial}{\partial\beta}y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta+\lambda\beta^T\beta \\
&=-2X^Ty+2X^TX\beta+2\lambda\beta
\end{aligned}$
</details>
{{</rawhtml>}}

Therefore, we can express our fitted value $\hat{y}$ as $X\hat{\beta}^{\text{ridge}}=X(X^TX+\lambda I)^{-1}X^Ty$. Here, if we use $UDV^T$, the singular value decomposition of $X$ instead of $X$, then we get the result below.

$\begin{aligned}
\hat{y}&=X(X^TX+\lambda I)^{-1}X^Ty \\\\
&=UDV^T(VDU^TUDV^T+\lambda I)^{-1}VDU^Ty \\\\
&=UDV^T(VD^2V^T+\lambda I)^{-1}VDU^Ty \\\\
&=UD\\{V^{-1}(VD^2V^T+\lambda I)(V^T)^{-1}\\}DU^Ty \\\\
&=U(D^2+\lambda I)^{-1}U^Ty \\\\
&=\sum_{i=1}^p\dfrac{d_i^2}{d_i^2+\lambda}u_iu_i^Ty
\end{aligned}$

We can interpret this result geometrically as in the linear regression. $u_iu_i^Ty$ can be viewed as a projection of $y$ onto the subspace spanned by the eigenvectors of $XX^T$, which is the same space as the column space of $X$. Then the shrinkage term $\dfrac{d_i^2}{d_i^2+\lambda}$ makes a shrinkage  to the direction of each $u_i$.

There is another way to see this concept more intuitively. By multiplying $V$ to both side of the equation $X=UDV^T$, we get $XV=UD$ and it means that $Xv_i=d_iu_i$. Thus, the vector $u_i$ can be expressed as $\dfrac{Xv_i}{d_i}$, which is a normalized principal component. Now we can consider the each elements of $u_i$ as the norm of the data projected on the principal component direction. Then we can say that ridge regression makes a shrinkage to these norms. We know that the variance of each principal components is $\dfrac{d_i}{n}$, so the higher the variance of principal component is, the more the shrinkage occurs.

$\begin{aligned}
\text{df}(\lambda)&=\text{tr}[X(X^TX+\lambda I)^{-1}X^T] \\\\
&=\text{tr}(H_\lambda) \\\\
&=\sum_{i=1}^p\dfrac{d_i^2}{d_i^2+\lambda}
\end{aligned}$

Here, $\lambda$ decides the amount of the shrinkage. We call the shrinkage term above 'effective degrees of freedom'.