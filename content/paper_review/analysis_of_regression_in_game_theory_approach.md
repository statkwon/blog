---
title: "Analysis of Regression in Game Theory Approach"
date: 2022-03-27
categories:
    - "Paper Review"
tags:
    - "XAI"
    - "SHAP"
    - "Net Effect"
    - "Shapley Value"
sidebar: false
---

## Summary
---

Multiple Regression에서 다중공선성이 존재할 경우, 변수들의 상대적 중요도를 파악하기 어렵다. 이 논문에서는 협동 게임 이론의 Shapley Value Imputation을 사용하여 다중공선성이 존재할 때도 일관적인 변수 중요도를 계산할 수 있는 방법을 소개하고 있다.

## 1. Introduction
---

Multicollinearity는 예측에 영향을 미치지는 않지만, 예측에 대한 변수들의 영향력을 판단할 때 방해가 된다.

- 회귀 계수가 불안정하다. (주어진 데이터가 조금만 바뀌어도 회귀 계수가 크게 바뀐다.)
- Simple Regression과 Multiple Regression에서의 회귀 계수가 다른 방향의 부호를 가질 수 있다.
- 이론적으로 중요한 변수의 회귀 계수가 중요하지 않은 값을 가질 수 있다.
- 통계적 검정력을 떨어뜨린다. (추정한 회귀 계수의 C.I.가 넓어진다.)

Multiple Regression의 상대적인 변수 중요도를 파악하기 위한 기존 지표로 Net Effect가 있지만, 역시 다중공선성의 영향을 받는다. 물론 상관성이 높은 변수를 제거하는 것이 해결책이 될 수 있지만, 다른 변수를 완벽하게 대체할 수 있는 변수는 없다는 점에서 모든 변수를 포함한 상태에서 각각의 중요도를 판단하는 것이 더욱 바람직하다. 따라서 Multicollinearity에 영향을 받지 않으면서 변수 중요도를 판단할 방법이 필요하고, Shapley Value Imputation이 그 해결책이 될 수 있다.

## 2. Predictor’s Contribution in Regression
---

2장에서는 Multiple Regression에서 변수들의 중요도를 측정하기 위한 기존 방법론들을 소개하고 있다. 시작하기 앞서, 모든 독립 변수 및 종속 변수가 표준화된 상황을 가정한다.

**1) Net Effects**

$\text{SSE}=(\mathbf{y}-X\boldsymbol{\beta})^T(\mathbf{y}-X\boldsymbol{\beta})=1-2\boldsymbol{\beta}^TX^T\mathbf{y}+\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}$

Let $\displaystyle \hat{\boldsymbol{\beta}}=\underset{\boldsymbol{\beta}}{\text{argmin}}\\;\text{SSE}$, then $X^TX\hat{\boldsymbol{\beta}}=X^T\mathbf{y}$ and thus $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$.

$\begin{aligned}
R^2&=\dfrac{\text{SSR}}{\text{SSTO}} \\\\
&=1-\dfrac{\text{SSE}}{\text{SSTO}} \\\\
&\approx1-\text{SSE} \\\\
&=2\boldsymbol{\beta}^TX^T\mathbf{y}-\boldsymbol{\beta}^TX^TX\boldsymbol{\beta} \\\\
&=\boldsymbol{\beta}^T(2X^T\mathbf{y}-X^TX\boldsymbol{\beta})
\end{aligned}$

Therefore, $\displaystyle \underset{\boldsymbol{\beta}}{\text{argmin}}\\;\text{SSE}=\underset{\boldsymbol{\beta}}{\text{argmax}}\\;R^2$ and $\displaystyle \max_{\boldsymbol{\beta}}R^2=\boldsymbol{\beta}^TX^T\mathbf{y}$.

Let $\mathbf{r}=X^T\mathbf{y}$, then $\displaystyle \max_{\boldsymbol{\beta}}R^2=\boldsymbol{\beta}^T\mathbf{r}=\sum_j\beta_jr_j$, where $r_j=\mathbf{x}_j^T\mathbf{y}$.

이때 $\text{NEF}_j=\beta_jr_j$를 $j$번째 변수에 대한 Net Effect라고 하며, $j$번째 변수의 중요도를 나타내는 데 사용할 수 있다. 여기서 $r_j$는 Simple Regression에서의 회귀 계수이고, $\beta_j$는 Multiple Regression에서의 회귀 계수이다. 다중공선성이 존재하는 경우 $r_j$와 $\beta_j$가 서로 다른 방향의 부호를 가질 수 있는데, 이러한 경우 $\text{NEF}_j<0$이 되어 해석이 어려워진다는 단점을 갖는다.

**2) Square of the Partial Correlation between $Y$ and $X_j$ with Fixed Other $X$'s**

$R_{yj}^2=\dfrac{R^2-R_{-j}^2}{1-R_{-j}^2}$, where $R_{-j}^2$ denotes $R^2$ calculated without $X_j$

**3) Usefulness**

$U_j=R^2-R_{-j}^2$, where $R_{-j}^2$ denotes $R^2$ calculated without $X_j$

변수들 가운데 상관성이 높은 변수들이 존재할 경우, 그 정도에 따라 $X^TX$의 역행렬이 존재하지 않거나, 존재하더라도 Ill-Conditioned Matrix가 되어 회귀 계수와 Net Effect의 신뢰도가 낮아진다.

## 3. Shapley Value Imputation
---

Shapley Value는 협동 게임 이론에서 협동 게임에 참여하는 선수들의 가치를 평가하기 위에 고안된 지표이다. $j$번째 선수에 대한 Shapley Value는 다음과 같다.

$\displaystyle S_j=\sum_{\text{all} \\; M}\gamma_n(M)[v(M\cup\{j\})-v(M)]$, where $\gamma_n(M)=\dfrac{m!(n-m-1)!}{n!}$

이는 $v$가 Value Function일 때, $j$번째 선수가 포함된 조합과 포함되지 않은 조합의 $v$값의 차이를 모든 가능한 조합에 대하여 가중 평균한 것과 같다.

이를 응용하여 회귀 모형에 사용된 변수를 게임에 참여한 선수로 간주하고, Value Function이 $R_M^2$일 때 $R_{\\{1, \ldots, p\\}}^2=R^2$에 대한 각 변수의 Shapley Value를 계산할 수 있다. 이러한 경우 $v(M\cup\{j\})-v(M)$가 곧 Usefulness와 같은 형태임을 어렵지 않게 확인할 수 있다. 즉, $j$번째 변수에 대한 Shapley Value는 $j$번째 변수를 포함하는 모든 가능한 조합에 대한 Usefulness의 가중 평균과 같다.

예를 들어, 독립 변수가 $X_1$, $X_2$, $X_3$ 세 개인 경우, $X_1$에 대한 Shapley Value는 다음과 같이 계산된다.

$\begin{aligned}
S_1&=\gamma_3(\phi)v(X_1)+\gamma_3(X_2)[v(X_1, X_2)-v(X_2)]+\gamma_3(X_3)[v(X_1, X_3)-v(X_3)]+\gamma_3(X_2, X_3)[v(X_1, X_2, X_3)-v(X_2, X_3)] \\\\
&=\gamma_3(\phi)U_1+\gamma_3(X_2)U_{12}+\gamma_3(X_3)U_{13}+\gamma_3(X_2, X_3)U_{123} \\\\
&=\gamma_3(0)U_1+\gamma_3(1)(U_{12}+U_{13})+\gamma_3(2)U_{123}
\end{aligned}$

where $U_1=R_1^2$, $U_{12}=R_{12}^2-R_2^2$, $U_{13}=R_{13}^2-R_3^2$, $U_{123}=R_{123}^2-R_{23}^2$ (2장과 표기가 다름)

여기서 주목해야할 점은 모든 변수에 대한 Shapley Value를 모두 합하면 그 값이 $R^2$와 같아진다는 것이다. 따라서 $j$번째 변수에 대한 Shapley Value를 $j$번째 변수가 $R^2$에서 차지하는 몫으로 해석할 수 있다.

$j$번째 변수에 대한 Shapley Value를 다음과 같은 방식으로도 표현할 수 있다.

$S_j=(R_j^2-\bar{R}_1^2)/(n-1)+(\bar{R}\_{j*}^2-\bar{R}_2^2)/(n-2)+(\bar{R}\_{j**}^2-\bar{R}_3^2)/(n-3)+\cdots+R^2/n$,

where $\bar{R}_{j\underset{p-1}{\underbrace{\*\cdots*}}}^2$ is a mean for all the models with $p$ regressors one of which is $X_1$ and $\bar{R}_p^2$ is a mean for all the models with any $p$ regressors.

## 4. SV Net Effects and Coefficients of Adjusted Regression
---

이 논문에서는 $\text{SV}_j$를 Shapley Value Imputation을 사용해서 추정한 $j$번째 변수의 Net Effect라고 정의한다. SV Net Effect는 모든 가능한 변수 조합에 대해 평균한 값이기 때문에 일반적인 Net Effect와 달리 다중공선성에 취약하지 않다. $\text{SV}_j$는 항상 $0$보다 큰 값을 갖기 때문에, 독립 변수들의 기여도에 대한 해석으로 사용하기에 적절하다. 4장에서는 이러한 SV Net Effect를 사용하여 회귀 계수를 보정하는 방법들에 대해 소개하고 있다.

첫 번째 방법은 $R^2=\sum_j\beta_jr_j$이고 동시에 $R^2=\sum_j\text{SV}_j$라는 점에서, $\text{SV}_j=a_jr_j$를 만족하는 $a_j$, 즉, $a_j=\text{SV}_j/r_j$를 $j$번째 변수에 대한 보정된 회귀 계수로 사용하는 것이다.

하지만 엄밀하게는 $\displaystyle \max\_{\boldsymbol{\beta}}R^2=\sum_j\beta_jr_j$이므로, 보다 일반적인 식인 $\text{SV}_j=a_j(2\mathbf{r}-X^TX\mathbf{a})_j$를 만족하는 $a_j$를 찾는 것이 두 번째 방법이다. 이 논문에서는 주어진 조건을 만족하는 $a_j$를 찾기 위한 방법으로 다음과 같은 식을 최소화하는 $a_j$를 찾는 방식을 제안한다.

$\displaystyle F=\sum_j[\text{SV}_j-a_j(2\mathbf{r}-X^TX\mathbf{a})_j]^2=\sum_j(\text{SV}_j-2a_jr_j+a_j\sum_k r\_{jk}a_k)^2$

이때 첫 번째 방법을 사용하여 얻은 결과를 최적화 알고리즘의 초깃값 $a_j^{(0)}$으로 사용할 수 있다. 이러한 방식을 통해 보정한 회귀 계수는 다중공선성에 취약하지 않으며, 해석이 용이하다는 장점을 갖는다.

## Memo
---

- SV Net Effects와 Adjusted Coefficients는 Linear Model에 한하여 사용할 수 있는 Global Interpretation 방식이다.

## Reference
---

1. Lipovetsky, S., & Conklin, M. (2001). Analysis of regression in game theory approach. Applied Stochastic Models in Business and Industry, 17(4), 319-330.
