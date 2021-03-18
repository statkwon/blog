---
title: "Gauss-Markov Theorem"
date: 2021-03-10
draft: false
weight: 2
TableOfContents: false
---

Gauss-Markov Theorem은 Least Square Estimation이 가장 좋은 추정량임을 증명하는 이론이다. 이를 BLUE(Best Linear Unbiased Estimator)라고 표현하기도 한다.

우선 우리는 $\hat{\beta}$이 Unbiased Estimator인 것을 보일 수 있다.

$\begin{aligned}
\text{E}[\hat{\beta}]&=\text{E}[(X^TX)^{-1}X^Ty] \\\\
&=(X^TX)^{-1}X^T\text{E}[y] \\\\
&=(X^TX)^{-1}X^TX\beta \\\\
&=\beta
\end{aligned}$

Unbiased Estimator라는 것만으로는 좋은 추정량이라고 할 수 없다. LSE가 BLUE인 이유는 $\beta$의 모든 Linear Unbiased Estimator 중 분산이 가장 작기 때문이다.

$\tilde{\beta}=Cy$를 $\beta$의 어떤 한 추정량이라고 하자. 이때 $C=(X^TX)^{-1}X^T+D$이고, $D$는 영행렬이 아니다. 우선 $\tilde{\beta}$ 역시 $\beta$의 Unbiased Estimator이어야 한다.

$\begin{aligned}
\text{E}[\tilde{\beta}]&=\text{E}[\left\((X^TX)^{-1}X^T+D\right\)y] \\\\
&=\text{E}[\left\((X^TX)^{-1}X^T+D\right\)(X\beta+\epsilon)] \\\\
&=\left\((X^TX)^{-1}X^T+D\right\)X\beta+\left\((X^TX)^{-1}X^T+D\right\)\text{E}[\epsilon] \\\\
&=(X^TX)^{-1}X^TX\beta+DX\beta \\\\
&=(I+DX)\beta
\end{aligned}$

따라서 $\tilde{\beta}$가 $\beta$의 Unbiased Estimator가 되기 위해서는 $DX=0$이어야 한다.

$\begin{aligned}
\text{Var}(\tilde{\beta})&=\text{Var}(Cy) \\\\
&=C\text{Var}(y)C^T \\\\
&=\sigma^2CC^T \\\\
&=\sigma^2\left\((X^TX)^{-1}X^T+D\right\)\left\(X(X^TX)^{-1}+D^T\right\) \\\\
&=\sigma^2\left\((X^TX)^{-1}X^TX(X^TX)^{-1}+(X^TX)^{-1}X^TD^T+DX(X^TX)^{-1}+DD^T\right\) \\\\
&=\sigma^2(X^TX)^{-1}+\sigma^2(X^TX)^{-1}(DX)^T+\sigma^2DX(X^TX)^{-1}+\sigma^2DD^T \\\\
&=\sigma^2(X^TX)^{-1}+\sigma^2DD^T \\\\
&=\text{Var}(\hat{\beta})+\sigma^2DD^T
\end{aligned}$

$DX=0$이기 때문에 위와 같은 식이 성립하고, $\text{Var}(\hat{\beta})≤\text{Var}(\tilde{\beta})$임을 확인할 수 있다. 즉, $\hat{\beta}$은 $\beta$의 모든 Linear Unbiased Estimator 중 분산이 가장 작은 추정량이다.

---

그렇지만 우리는 LSE가 사실은 최적의 추정량이 아닐 수도 있다는 점에 대해 생각해볼 필요가 있다. Gauss-Markov Theorem에서는 좋은 추정량의 조건으로 Unbiased Estimator를 전제로 하고 있다. 즉 Bias가 $0$인 후보들만 고려한 것이다.

어떤 추정량 $\tilde{\theta}$의 $\text{MSE}$를 아래와 같이 분해할 수 있다.

$\begin{aligned}
\text{MSE}&=\text{E}[\tilde{\theta}-\theta]^2 \\\\
&=\text{E}\left[\tilde{\theta}-\text{E}[\tilde{\theta}]+\text{E}[\tilde{\theta}]-\theta\right]^2 \\\\
&=\text{E}\left[\tilde{\theta}-\text{E}[\tilde{\theta}]\right]^2+\text{E}\left[\text{E}[\tilde{\theta}]-\theta\right]^2 \\\\
&=\text{Var}(\tilde{\theta})^2+\left\\{\text{E}[\tilde{\theta}]-\theta\right\\}^2
\end{aligned}$

$\text{MSE}$가 Variance와 Bias의 결합으로 이루어져 있는 것을 확인할 수 있다. 따라서 Bias가 $0$이 된다고 무조건 좋은 추정량이라고 하기는 어렵다. Bias가 어느 정도 발생하더라도 Variance를 큰 폭으로 감소시킬 수 있다면 그러한 추정량이 더 좋은 추정량일 수 있다.