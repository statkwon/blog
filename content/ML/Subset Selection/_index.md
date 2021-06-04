---
title: "Subset Selection"
date: 2021-02-22
TableOfContents: true
weight: 4
---

As mentioned before, coefficients estimated by least squares method can be modified to exchange a little bias for a larger reduction in variance. Subset selection method not only makes it possible to enhance the prediction accuracy in this sense, but also yields more interpretable model.

**Best-Subset Selection**

{{<figure src="/esl_fig_3.5.png" width="500" height="200">}}

If we have $p$ variables, we can consider $2^p$ models. For each number of variables, we will select a model with smallest residual sum of squares. The red dots in the figure above are the selected ones. Among these $p+1$ models, we have to choose the best model. Here, the best is usually measured by cross-validation error or AIC criterion. It is reasonable to pick a smaller model when the scores are almost similar. This is a quite accurate method as we search through all possible subsets of variables, but is also a time-consuming procedure.

---

**Forward- and Backward-Stepwise Selection**

To deal with the calculation cost of best-subset selection, Forward- and Backward-Stepwise Selection can be suggested.

Forward-Stepwise Selection differs from best-subset selection in that it starts from the model with only the intercept term and add one variable at a time which reduces the residual sum of squares most. Among $p+1$ models from each step, the best model is selected by the same way used in best-subset selection. This process results in fitting $1+p+(p-1)+\cdots+1=1+\dfrac{p(p+1)}{2}$ models in total, so it is certainly a cost-efficient method.

This greedy algorithm can be simply computed by using $QR$-Decomposition.

Let $X_q=\begin{bmatrix} \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_q \end{bmatrix}$ be the data matrix for $q$th step with $q$ variables selected among $p$ variables.

Now let $\mathbf{x}\_{q+i}$ be any variable from the left $p-q$ variables $\mathbf{x}\_{q+1}, \cdots, \mathbf{x}_p$.

We can get a $QR$-decomposition of $X_q$ as $X_q=QR$ and $Q=\begin{bmatrix} \mathbf{u}_1 & \mathbf{u}_2 & \cdots & \mathbf{u}_q \end{bmatrix}$.

If we add the variable $\mathbf{x}\_{q+i}$ into $X_q$, $\mathbf{u}\_{q+i}$ will also be added into $Q$.

$\mathbf{u}\_{q+i}$ is calculated as $\dfrac{\mathbf{z}\_{q+i}}{\Vert\mathbf{z}\_{q+i}\Vert^2}$, where $\mathbf{z}\_{q+i}=\mathbf{x}\_{q+i}-\text{proj}_Q\mathbf{x}\_{q+i}=\mathbf{x}\_{q+i}-\sum\_{j=1}^q(\mathbf{x}\_{q+i}\cdot\mathbf{u}_j)\mathbf{u}_j=\mathbf{x}\_{q+i}-(I-QQ^T)\mathbf{x}\_{q+i}$.

Let $Q^*=\begin{bmatrix} \mathbf{u}_1 & \mathbf{u}_2 & \cdots & \mathbf{u}\_{q+i} \end{bmatrix}$, then

$\begin{aligned}
\hat{\mathbf{y}}^*&=Q^*Q^{*T}\mathbf{y} \\\\
&=\begin{bmatrix} Q & \mathbf{u}\_{q+i} \end{bmatrix}\begin{bmatrix} Q^T \\\\ \mathbf{u}\_{q+i} \end{bmatrix}\mathbf{y} \\\\
&=QQ^T\mathbf{y}+\mathbf{u}\_{q+i}\mathbf{u}\_{q+i}^T\mathbf{y} \\\\
&=\hat{\mathbf{y}}+(\mathbf{u}\_{q+i}^T\mathbf{y})\mathbf{u}\_{q+i}
\end{aligned}$

We want to know how much the residual sum of squares will decrease, so we have to calculate $\mathbf{r}\^{*T}\mathbf{r}^\*$, residual sum of squares after $\mathbf{x}\_{q+i}$ is added.

$\begin{aligned}
\mathbf{r}^*&=\mathbf{y}-\hat{\mathbf{y}}^\* \\\\
&=\mathbf{y}-\hat{\mathbf{y}}-(\mathbf{u}\_{q+i}^T\mathbf{y})\mathbf{u}\_{q+i} \\\\
&=\mathbf{r}-(\mathbf{u}\_{q+i}^T\mathbf{y})\mathbf{u}\_{q+i}
\end{aligned}$

$\begin{aligned}
\mathbf{r}^{*T}\mathbf{r}^\*&=\mathbf{r}^T\mathbf{r}-2(\mathbf{u}\_{q+i}^T\mathbf{y})\mathbf{r}^T\mathbf{u}\_{q+i}+(\mathbf{u}\_{q+i}^T\mathbf{y})^2 \\\\
&=\mathbf{r}^T\mathbf{r}-2(\mathbf{u}\_{q+i}^T\mathbf{y})^2+(\mathbf{u}\_{q+i}^T\mathbf{y})^2 \\\\
&=\mathbf{r}^T\mathbf{r}-(\mathbf{u}\_{q+i}^T\mathbf{y})^2
\end{aligned}$

Thus we have to pick $\mathbf{x}\_{q+i}$, where $i=\underset{i}{\text{argmax}}\vert\mathbf{u}\_{q+i}^T\mathbf{y}\vert$.

Backward-Stpewise Selection is almost same as forward-stepwise method, except that it starts with the full model. It removes one variable at a time which reduces the residual sum of squares most. Unlike the forward-stepwise method, backward-stepwise selection cannot be used when $p>n$. There also exists a hybrid method that consider both forward and backward moves at each step, and select the best of the two.

---

**Forward-Stagewise Regression**

Due to the fact that Forward-Stagewise Regression takes more than $p$ steps to reach the least squares fit, it is regarded as being inefficient. However, this slow-fitting algorithm  takes advantage in high-dimensional situation.

It starts with an intercept and centered predictors with coefficients initially all $0$. At each step the algorithm finds the variable most correlated with the current residual. It then computes the simple linear regression coefficient of the residual on this chosen variable, and then adds it to the current coefficient for that variable. This is continued till none of the variables have correlation with the residuals.

---

**Reference**

1. Elements of Statistical Learning