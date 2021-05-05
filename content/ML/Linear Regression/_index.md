---
title: "Linear Regression"
date: 2021-03-10
draft: false
---

Linear Regression은 Input과 Output 사이의 Linear Relationship을 가정한다.

$$y=X\beta+\epsilon, \quad \epsilon \sim (0, \sigma^2I)$$

$$Y=\begin{bmatrix} y_1 \\\\ y_2 \\\\ \vdots \\\\ y_n \end{bmatrix} \qquad X=\begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\\\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\\\ \vdots & \vdots & \vdots & \ddots & \vdots \\\\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{bmatrix} \qquad \beta=\begin{bmatrix} \beta_0 \\\\ \beta_1 \\\\ \vdots \\\\ \beta_p \end{bmatrix} \qquad \epsilon=\begin{bmatrix} \epsilon_1 \\\\ \epsilon_2 \\\\ \vdots \\\\ \epsilon_n \end{bmatrix}$$

회귀 모델의 Parameter를 추정하는 방법 중 가장 보편적인 것은 Least Square Estimation이다. LSE는 Residual Sum of Squares인 $(y-X\beta)^T(y-X\beta)$를 최소화하는 값을 $\beta$의 추정치로 사용하는 방식이다.

$\begin{aligned}
\dfrac{\partial\text{SSE}}{\partial\beta}&=\dfrac{\partial (y-X\beta)^T(Y-X\beta)}{\partial\beta} \\\\
&=\dfrac{\partial (y^T-\beta^TX^T)(y-X\beta)}{\partial\beta} \\\\
&=\dfrac{\partial (y^Ty-\beta^TX^Ty-y^TX\beta+\beta^TX^TX\beta)}{\partial\beta} \\\\
&=-2X^Ty+2X^TX\beta
\end{aligned}$

따라서 이 식이 $0$이 되게 하는 $\beta$를 찾아보면 아래와 같다.

$$X^TX\beta=X^Ty$$

$X$가 Full Column Rank를 갖는 경우($X^TX$가 Nonsingular한 경우) $\hat{\beta}=(X^TX)^{-1}X^Ty$이 위 식을 만족시키는 유일한 해가 된다. 만약 $X$가 Full Column Rank를 갖지 않는다면 위 식을 만족시키는 해가 존재하기는 하지만, 유일하지 않고 여러 개의 해를 갖게 된다. 회귀 분석에서 $X$ 변수들 사이의 독립을 전제로 하는 것 역시 이와 같은 맥락에서 나온 것으로 생각할 수 있다.

$$\hat{y}=X\hat{\beta}=X(X^TX)^{-1}X^Ty$$

![ESL fig 3.2](/esl_fig_3.2.png)

위와 같은 그림을 참고하여 LSE의 기하적인 의미를 생각해볼 수 있다. $\hat{y}=X(X^TX)^{-1}X^Ty$에서 $X(X^TX)^{-1}X^T$를 $H$라고 하면, $H^2=H$이고 $H^T=H$임을 쉽게 확인할 수 있다. 즉, $H$는 Symmetric하고 Idempotent한 행렬이므로 Orthogonal Projection에 대한 Standard Matrix라고 할 수 있다. 그렇기 때문에 $\hat{y}$은 $y$를 $X$의 Column Space에 Orthogonal Projection한 것과 같다. 앞서 $X$가 Full Column Rank를 갖지 않으면 해가 존재하기는 하지만 여러 개의 해를 갖게 된다고 하였는데, 이는 기하적으로 생각했을 때, 여전히 $y$를 $X$의 Column Space에 Projection할 수는 있지만 그것이 Orthogonal하지는 않은 것이라고 할 수 있다.