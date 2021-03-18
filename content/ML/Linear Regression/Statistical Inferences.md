---
title: "Statistical Inferences"
date: 2021-03-10
draft: true
weight: 1
TableOfContents: true
---

## 1. Distribution Assumption

LSE를 사용하여 추정한 회귀 모델의 Parameter에 대한 통계적 추론은 새로운 가정을 필요로 한다.

$$\epsilon \overset{\text{iid}}{\sim} N(0, \sigma^2I)$$

회귀 모델의 Error Term이 평균이 $0$이고 분산이 $\sigma^2$인 정규 분포를 따른다는 가정이다. 이러한 가정 하에서 $\hat{\beta}$ 역시 다음과 같은 분포를 따르게 된다.

$\begin{aligned}
\text{E}[\hat{\beta}]&=\text{E}[(X^TX)^{-1}X^Ty] \\\\
&=(X^TX)^{-1}X^T\text{E}[y] \\\\
&=(X^TX)^{-1}X^TX\beta \\\\
&=\beta
\end{aligned}$

$\begin{aligned}
\text{Cov}(\hat{\beta})&=\text{Cov}((X^TX)^{-1}X^Ty) \\\\
&=(X^TX)^{-1}X^T\text{Cov}(y)X(X^TX)^{-1} \\\\
&=(X^TX)^{-1}\sigma^2
\end{aligned}$

$\therefore \hat{\beta} \sim N(\beta, (X^TX)^{-1}\sigma^2)$

$H=X(X^TX)^{-1}X^T$라고 하면 $H^2=H$이고 $H^T=H$이므로 $\text{SSE}$를 아래와 같이 표현할 수 있다.

$\begin{aligned}
\text{SSE}&=(Y-X\hat{\beta})^T(Y-X\hat{\beta}) \\\\
&=(Y-HY)^T(Y-HY) \\\\
&=(Y^T-Y^TH^T)(Y-HY) \\\\
&=Y^T(I-H^T)(I-H)Y \\\\
&=Y^T(I-H)Y
\end{aligned}$

## 2. $F$-Test

$$H_0:\beta_R=0 \quad \text{vs} \quad H_1:\beta_R \neq 0$$

$$F=\dfrac{(\text{SSE}_R-\text{SSE}_F)/(p_F-p_R)}{\text{SSE}_F/(n-p_F-1)}$$

## 3. $t$-Test

$$H_0:\beta_i=0 \quad \text{vs} \quad H_1:\beta_i \neq 0$$

$$t=\dfrac{\hat{\beta}_i}{\left[(X^TX)^{-1}\hat{\sigma}^2\right]\_{ii}}$$