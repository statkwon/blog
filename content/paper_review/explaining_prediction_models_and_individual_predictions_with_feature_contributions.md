---
title: "Explaining Prediction Models and Individual Predictions with Feature Contributions"
date: 2022-03-28
categories:
  - "Paper Review"
tags:
  - "XAI"
  - "SHAP"
  - "Shapley Sampling Value"
sidebar: false
---

## Abstract
---

이 논문에서는 어떠한 종류의 Classification 또는 Regression 모형에도 적용할 수 있는 모형 해석 방법에 대해 소개한다. 이 방법은 기존의 방법들과는 달리 가능한 모든 변수 조합을 살핌으로써 변수 간 상호작용을 고려할 수 있다는 장점을 갖는다.

## 1. Introduction
---

변수 간 상호작용이 존재하지 않는 Additive Regression Model에서 모든 변수가 표준화된 경우, 회귀 계수를 통해 변수들의 Global Importance를 파악할 수 있다. 반면 단일 예측값에 대한 변수들의 기여도를 알고 싶은 경우에는 다음과 같은 식의 Situational Importance를 사용할 수 있다.

$\begin{aligned}
\phi_i(\mathbf{x})&=f(\mathbf{x})-\text{E}[f\vert X_j=x_j, \forall j\neq i] \\\\
&=\beta_0+\beta_1x_1+\cdots+\beta_nx_n-\text{E}[\beta_0+\beta_1x_1+\cdots+\beta_iX_i+\cdots+\beta_nx_n] \\\\
&=\beta_0+\beta_1x_1+\cdots+\beta_nx_n-\beta_0+\beta_1x_1+\cdots+\beta_i\text{E}[X_i]+\cdots+\beta_nx_n \\\\
&=\beta_ix_i-\beta_i\text{E}[X_i]
\end{aligned}$

즉, $f(\mathbf{x})$라는 예측값에 대한 $i$번째 변수의 Situational Importance는 $f(\mathbf{x})$와 $i$번째 변수의 값이 주어지지 않았을 때 평균적인 예측값의 차이로 정의된다.

이를 축구 경기에 빗대어 표현하자면, 어떤 경기에서 팀이 5점을 득점했을 때 $i$번째 선수의 기여도를

[해당 경기의 득점(5점) $-$ $i$번째 선수를 제외한 나머지 선수들이 모두 해당 경기와 동일하게 득점한 다른 경기들의 평균 득점]으로 계산하는 것으로 생각할 수 있다.

이러한 지표가 유의미하게 받아들여지기 위해서는 득점에 있어 선수들 사이의 협력은 존재하지 않는다는 가정이 필요하다. 하지만 축구 경기에서 대부분의 득점이 다른 팀원들의 도움을 통해 이루어지듯이, 많은 경우 변수 간 상호작용이 존재한다. 따라서 보다 일반적인 모형에 대해 Situational Importance를 적용하는 것은 바람직하지 않다.

## 2. Computing a Feature's Contribution
---

2장에서는 Situational Importance를 수정하여 보다 일반적인 모형(Not Additive)에 적용할 수 있게 만든 방식에 대해 소개하고 있다.

Assumption: $\mathbf{X}\in\mathcal{X}=[0, 1]^n$, $Y\in\mathbb{R}$, $f:\mathcal{X}\rightarrow\mathbb{R}$

우선 어떠한 모형의 예측값 $f(\mathbf{x})$에 대한 특정 변수들($Q\subseteq S=\\{1, 2, \ldots, n\\}$)의 기여도를 다음과 같이 정의한다.

$\Delta_Q(\mathbf{x})=\text{E}[f\vert X_i=x_i, \forall i\in Q]-\text{E}[f]$

$Q\subseteq S$이므로, 총 $2^n$개의 조합에 대한 기여도를 구할 수 있다.

다음으로 $Q$의 임의의 부분집합 $W$에 대하여 $W$에 속한 변수들 간의 상호작용을 $\mathcal{I}_W(\mathbf{x})$라고 할 때, $\Delta_Q(\mathbf{x})$를 모든 상호작용의 합으로 표현할 수 있다고 가정한다.

이를 식으로 표현하면 $\displaystyle \Delta\_Q(\mathbf{x})=\sum\_{W\subseteq Q}\mathcal{I}\_W(\mathbf{x})$이고, 이 식을 통해 $Q$에 속한 변수들 간의 상호작용을 $\displaystyle \mathcal{I}\_Q(\mathbf{x})=\Delta\_Q(\mathbf{x})-\sum\_{W\subset Q}\mathcal{I}\_W(\mathbf{x})$로 계산할 수 있다.

이후 $i$번째 변수가 포함된 모든 상호작용의 가중평균, $\displaystyle \phi_i(\mathbf{x})=\sum_{Q\subseteq S\backslash\\{i\\}}\dfrac{\mathcal{I}_{W\cup\\{i\\}}(\mathbf{x})}{\vert W\vert+1}$를 예측값 $f(\mathbf{x})$에 대한 $i$번째 변수의 기여도로 사용한다.

이 논문에서는 이러한 과정을 통해 구한 $\phi_i(\mathbf{x})$가 Value Function이 $\Delta_Q(\mathbf{x})$일 때 $\Delta_S(\mathbf{x})=f(\mathbf{x})-\text{E}[f]$에 대한 $i$번째 변수의 Shapley Value와 같음을 밝히고 있다.

$\begin{aligned}
\phi\_i(\mathbf{x})&=\sum\_{Q\subseteq S\backslash\\{i\\}}\dfrac{\vert Q\vert!(\vert S\vert-\vert Q\vert-1)!}{\vert S\vert!}(\Delta\_{Q\cup\\{i\\}}(\mathbf{x})-\Delta\_Q(\mathbf{x})) \\\\
&=\sum\_{Q\subseteq S\backslash\\{i\\}}\dfrac{\vert Q\vert!(\vert S\vert-\vert Q\vert-1)!}{\vert S\vert!}(\text{E}[f\vert X\_j=x\_j, \forall j\in Q\cup\\{i\\}]-\text{E}[f\vert X\_j=x\_j, \forall j\in Q])
\end{aligned}$

**Properties of $\phi_i(\mathbf{x})$**

- $\sum_{i=1}^n\phi_i(\mathbf{x})=\Delta_S(\mathbf{x})$ - 모든 변수에 대한 Shapley Value의 합은 $\Delta_S(\mathbf{x})$, 즉, $f(\mathbf{x})-\text{E}[f]$와 같다.
- $\forall W\subseteq S\backslash\\{i\\}:\Delta_W=\Delta_{W\cup\\{j\\}}\Rightarrow\phi_i(\mathbf{x})=0$ - 어떤 변수가 예측값에 영향을 미치지 않을 경우, 해당 변수의 기여도는 $0$으로 책정된다.
- $\forall W\subseteq S\backslash\\{i, j\\}:\Delta_{W\cup\\{i\\}}=\Delta_{W\cup\\{j\\}}\Rightarrow\phi_i(\mathbf{x})=\phi_j(\mathbf{x})$ - 서로 다른 두 변수가 예측값에 미치는 영향이 동일할 경우, 두 변수의 기여도는 동일하게 책정된다.
- $\forall \mathbf{x}, \mathbf{y}\in\mathcal{X}:\phi(\mathbf{x}+\mathbf{y})=\phi(\mathbf{x})+\phi(\mathbf{y})$, where $\Delta_Q(\mathbf{x}+\mathbf{y})=\Delta_Q(\mathbf{x})+\Delta_Q(\mathbf{y})$ for all $Q\subseteq S$ - Additivity across Instances

## 3. Approximation Algorithm
---

$\phi_i(\mathbf{x})$의 정확한 값을 계산하는 것의 시간 복잡도는 $O(2^n)$으로, 변수의 개수가 많을 수록 실질적인 계산이 불가능하다. 3장에서는 그 값을 근사하기 위한 알고리즘에 대해 소개하고 있다.

**Theoretical Background for Algorithm**

Shapley Value의 Alternative Formula를 사용하여 $\phi_i(\mathbf{x})$를 다음과 같은 형태로 표현할 수 있다.

$\begin{aligned}
\phi_i(\mathbf{x})&=\sum_{Q\subseteq S\backslash\\{i\\}}\dfrac{\vert Q\vert!(\vert S\vert-\vert Q\vert-1)!}{\vert S\vert!}(\Delta_{Q\cup\\{i\\}}(\mathbf{x})-\Delta_Q(\mathbf{x})) \\\\
&=\dfrac{1}{n!}\sum_{\mathcal{O}\in\pi(n)}(\Delta_{\text{Pre}^i(\mathcal{O})\cup\\{i\\}}-\Delta_{\text{Pre}^i(\mathcal{O})})
\end{aligned}$

where $\pi(n)$ is the set of all ordered permutations of the feature indices $\\{1, 2, \ldots, n\\}$, $\text{Pre}^i(\mathcal{O})$ is the set of all indices that precede $i$ in permutation $\mathcal{O}\in\pi(n)$

(ex. $\pi(3)=\\{\\{1, 2, 3\\}, \\{1, 3, 2\\}, \\{2, 1, 3\\}, \\{2, 3, 1\\}, \\{3, 1, 2\\}, \\{3, 2, 1\\}\\}$, If $\mathcal{O}=\\{2, 1, 3\\}$, then $\text{Pre}^1(\mathcal{O})=\\{2\\}$.)

$\Delta$-terms의 계산 비용이 작을 경우 $\Delta\_{\text{Pre}^i(\mathcal{O})\cup\\{i\\}}-\Delta\_{\text{Pre}^i(\mathcal{O})}$에 대한 Monte-Carlo Sampling을 사용하여 $\phi_i(\mathbf{x})$를 추정할 수 있다.
하지만 $\Delta$-terms의 계산 비용은 $O(2^n)$이므로, Sampling Algorithm을 사용하기 위해서는 추가적인 가정이 필요하다.

이 논문에서는 변수들 사이의 독립을 가정한다. 해당 가정 하에서 $\Delta_Q(\mathbf{x})$를 다음과 같은 간단한 형태로 나타낼 수 있다.

$\begin{aligned}
\Delta_Q(\mathbf{x})&=\text{E}[f\vert X_i=x_i, \forall i\in Q]-\text{E}[f] \\\\
&=\text{E}\_{\mathbf{X}\_{Q^c}\vert\mathbf{X}\_Q}[f]-\text{E}[f] \\\\
&=\text{E}\_{\mathbf{X}\_{Q^c}}[f]-\text{E}[f] \qquad (\because\text{Feature Independece}) \\\\
&=\sum_{\mathbf{w}\in\mathcal{X}}p(\mathbf{w})(f(\mathbf{w}_{[w_i=x_i, i\in Q]})-f(\mathbf{w}))
\end{aligned}$

where $\mathbf{w}_{[w_i=x_i, i\in S]}$ denotes instance $\mathbf{w}$ with the value of feature $i$ replaced with that feature's value in instance $\mathbf{x}$, for each $i\in S$

(ex. $\mathbf{w}=\left<2, 4, 6\right>$, $\mathbf{x}=\left<3, 5, 7\right>$, $\mathbf{w}_{[w_i=x_i, i\in\\{1, 3\\}]}=\left<3, 4, 7\right>$)

이를 $\phi_i(\mathbf{x})$의 식에 대입하면 $\displaystyle \phi\_i(\mathbf{x})=\dfrac{1}{n!}\sum\_{\mathcal{O}\in\pi(n)}\sum\_{\mathbf{w}\in\mathcal{X}}p(\mathbf{w})\cdot\left(f(\mathbf{w}\_{[w\_j=x\_j, j\in\text{Pre}^i(\mathcal{O})\cup\\{i\\}]})-f(\mathbf{w}\_{[w\_j=x\_j, j\in\text{Pre}^i(\mathcal{O})]})\right)$와 같은 식을 얻을 수 있다.

이후 $V\_{\mathcal{O}, \mathbf{w}\in\mathcal{X}}=\left(f(\mathbf{w}\_{[w\_j=x\_j, j\in\text{Pre}^i(\mathcal{O})\cup\\{i\\}]})-f(\mathbf{w}\_{[w\_j=x\_j, j\in\text{Pre}^i(\mathcal{O})]})\right)$로부터 $p(\mathbf{w})$의 확률로 추출한 $m$개의 Sample $V_j$를 사용하여 $\phi_i(\mathbf{x})$를 추정할 수 있다.

$\displaystyle \hat{\phi}\_i(\mathbf{x})=\dfrac{1}{m}\sum_{j=1}^mV_j$

이때 $\hat{\phi}_i(\mathbf{x})$는 근사적으로 평균이 $\phi_i(\mathbf{x})$이고 분산이 $\frac{\sigma_i^2}{m}$인 정규분포를 따른다. ($\sigma_i^2$ is the population variance.) 즉, $\hat{\phi}_i(\mathbf{x})$는 $\phi_i(\mathbf{x})$에 대한 Consistent Estimator이다.

**Algorithm** - Approximating the $i$th feature's contribution for model $f$, instance $\mathbf{x}\in\mathcal{X}$ and distribution $p$

{{<figure src="/paper_review/shapley_sampling_values1.png" width="700">}}

## Memo
---

- Shapley Sampling Values는 Feature Independence를 가정한다.

## Reference
---

1. Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and information systems, 41(3), 647-665.
2. Štrumbelj, E., & Kononenko, I. (2011, April). A general method for visualizing and explaining black-box regression models. In International Conference on Adaptive and Natural Computing Algorithms (pp. 21-30). Springer, Berlin, Heidelberg.
3. Strumbelj, E., & Kononenko, I. (2010). An efficient explanation of individual classifications using game theory. The Journal of Machine Learning Research, 11, 1-18.
4. [https://en.wikipedia.org/wiki/Shapley_value](https://en.wikipedia.org/wiki/Shapley_value)
