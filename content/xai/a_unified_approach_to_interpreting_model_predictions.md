---
title: "A Unified Approach to Interpreting Model Predictions"
date: 2022-04-02
categories:
  - "XAI"
tags:
  - "SHAP"
  - "Shapley Value"
sidebar: false
---

## Abstract

최근 복잡한 머신러닝 모형의 예측 결과를 해석하기 위한 많은 방법들이 제안되었지만, 서로 어떠한 관련이 있는지, 어떠한 방법이 더 우세한지 등이 불분명한 상황이다. 이 논문에서는 이러한 문제를 해결하기 위해 SHAP(SHapley Additive exPlanations)이라는 통합 프레임워크를 제시한다. SHAP은 기존의 여러 방법들이 충족하지 못하던 몇 가지 바람직한 특성을 모두 만족한다는 장점을 갖는다.

## 2. Additive Feature Attribution Methods

간단한 모형(ex. Linear Regression)에 대한 최고의 해석은 모형 그 자체이다. 하지만 복잡한 모형의 경우 모형을 곧바로 해석하는 것이 어려우므로, 기존 모형을 해석 가능한 모형으로 근사한 Explanation Model을 통해 해석하는 것이 일반적이다. 이 논문에서는 임의의 $\mathbf{x}$에 대한 예측 결과인 $f(\mathbf{x})$를 해석하는 경우에 초점을 맞추고 있다.

Explanation Model은 종종 Original Input $\mathbf{x}$ 대신 임의의 Mapping Function $h_\mathbf{x}$를 통해 $\mathbf{x}$로 매핑되는 Simplified Input $\mathbf{x}'$을 사용한다. 2장에서는 이러한 Explanation Model의 한 형태로 다음과 같은 식의 Additive Feature Attribution Method(이하 AFAM)를 소개하고 있다.

$\displaystyle g(\mathbf{z}')=\phi_0+\sum_{i=1}^M\phi_iz_i'$, where $\mathbf{z}'\in\\{0, 1\\}^M$, $M$ is the number of simplified input features, and $\phi_i\in\mathbb{R}$

AFAM은 $0$ 또는 $1$의 값만을 갖는 Simplified Input $z_1', \ldots, z_M'$의 선형 결합으로 $f(\mathbf{z})$를 근사한다. ($\mathbf{z}'\approx\mathbf{x}'$)

여기서 주목할 점은 [LIME](/xai/why_should_i_trust_you_explaining_the_predictions_of_any_classifier), DeepLIFT, Layer-Wise Relevance Propagation, [Shapley Regression Values](/xai/analysis_of_regression_in_game_theory_approach), [Shapley Sampling Values](/xai/explaining_prediction_models_and_individual_predictions_with_feature_contributions), Quantitative Input Influence 등의 기존 방법론들이 모두 이와 같은 형태를 갖는다는 것이다.

## 3. Simple Properties Uniquely Determine Additive Feature Attributions

3장에서는 AFAM이 갖추면 좋을 세 가지 특성에 대해 소개하고 있다.

**Local Accuracy** - "Explanation Model이 Original Model을 정확하게 근사한다."

$\displaystyle f(\mathbf{x})=g(\mathbf{x}')=\phi_0+\sum_{i=1}^M\phi_ix_i'$, where $\mathbf{x}=h_\mathbf{x}(\mathbf{x}')$

**Missingness**

$x_i'=0 \quad\Rightarrow\quad \phi_i=0$

**Consistency**

Let $f_\mathbf{x}(\mathbf{z}')=f(h_\mathbf{x}(\mathbf{z}'))$ and $\mathbf{z}'\backslash i$ denote setting $z_i'=0$.

For any two models $f$ and $f'$, if $f_\mathbf{x}'(\mathbf{z}')-f_\mathbf{x}'(\mathbf{z}'\backslash i)≥f_\mathbf{x}(\mathbf{z}')-f_\mathbf{x}(\mathbf{z}'\backslash i)$ for all inputs $\mathbf{z}'\in\\{0, 1\\}^M$, then $\phi_i(f', \mathbf{x})≥\phi_i(f, \mathbf{x})$.

하지만 모든 AFAM이 항상 이 세 가지 특성을 만족하는 것은 아니다.

**Theorem**

Only one possible explanation model $g$ follows the form of AFAM and satisfies Local Accuracy, Missingness, and Consistency:

$\displaystyle \phi_i(f, \mathbf{x})=\sum_{\mathbf{z}'\subseteq\mathbf{x}'}\dfrac{\vert\mathbf{z}'\vert!(M-\vert\mathbf{z}'\vert-1)!}{M!}[f_\mathbf{x}(\mathbf{z}')-f_\mathbf{x}(\mathbf{z}'\backslash i)]$,

where $\vert\mathbf{z}'\vert$ is the number of non-zero entries in $\mathbf{z}'$, and $\mathbf{z}'\subseteq\mathbf{x}'$ represents all $\mathbf{z}'$ vectors where the non-zero entries are a subset of the non-zero entries in $\mathbf{x}'$.

AFAM이 Local Accuracy, Missingness, 그리고 Consistency를 만족시키기 위해서는 반드시 위와 같은 형태의 $\phi_i$를 사용해야 한다. 이때 $\phi_i(f, \mathbf{x})$는 Value Function이 $f_\mathbf{x}(\mathbf{z}')$일 때 $f_\mathbf{x}(\mathbf{x}')$에 대한 $i$번째 변수의 Shapley Value를 의미한다. LIME이나 DeepLIFT와 같은 기존 방법들은 AFAM에 속하지만 Shapley Value에 기반하지 않으므로 Local Accuracy 또는 Consistency를 만족하지 못한다.

## 4. SHAP (SHapley Additive exPlanation) Values

{{<figure src="/xai/shap1.png" width="700">}}

이 논문에서는 Value Function으로 $f\_\mathbf{x}(\mathbf{z}')=\text{E}[f(\mathbf{z})\vert\mathbf{z}\_S]$를 사용함으로써 $\phi\_i(f, \mathbf{x})$를 $f\_\mathbf{x}(\mathbf{x}')=f(\mathbf{x})$에 대한 $i$번째 변수의 Shapley Value로 사용하는 방식을 SHAP이라는 이름의 통합 프레임워크로 제시하고 있다. ($S$ is the set of non-zero indices in $\mathbf{z}'$)

SHAP Value를 정확하게 계산하는 것은 많은 비용을 필요로 한다. 따라서 4장에서는 비교적 적은 비용으로 SHAP Value를 추정하기 위한 다양한 방법들에 대해 소개하고 있다.

### 4.1. Model-Agnostic Approximations

변수들 간의 독립을 가정함으로써 계산 비용을 줄일 수 있다.

$\begin{aligned}
f\_\mathbf{x}(\mathbf{z}')&=\text{E}[f(\mathbf{z})\vert\mathbf{z}\_S] \\\\
&=\text{E}\_{\mathbf{z}\_\bar{S}\vert\mathbf{z}\_S}[f(\mathbf{z})] \qquad (\bar{S} \\; \text{is the set of features not in} \\; S)\\\\
&\approx\text{E}_{\mathbf{z}\_\bar{S}}[f(\mathbf{z})] \qquad (\text{Feature Independence})
\end{aligned}$

[Shapley Sampling Values](/xai/explaining_prediction_models_and_individual_predictions_with_feature_contributions)나 Quantitative Input Influence와 같은 방식으로 SHAP Value를 추정할 수 있다.

**Kernel SHAP**

$\Omega(g)=0$

$\pi\_{\mathbf{x}'}(\mathbf{z}')=\dfrac{(M-1)}{\_MC\_{\vert\mathbf{z}'\vert}\vert\mathbf{z}'\vert(M-\vert\mathbf{z}'\vert)}$, where $\vert\mathbf{z}'\vert$ is the number of non-zero elements in $\mathbf{z}'$

$\displaystyle L(f, g, \pi\_{\mathbf{x}'})=\sum\_{\mathbf{z}'\in Z}[f(h\_\mathbf{x}(\mathbf{z}'))-g(\mathbf{z}')]^2\pi\_{\mathbf{x}'}(\mathbf{z}')$

## Memo

- Shapley Regression Values의 경우 Global Interpretation을 위한 방식이므로 Additive Feature Attribution Method에 포함시키는 것이 다소 부적절하다고 생각한다.
- Shapley Value를 사용하여 변수들의 기여도를 책정하겠다는 시도는 SHAP이 발표되기 이전부터 존재하였다.
- 이 논문에서 소개하고 있는 모든 SHAP Value 추정 방식은 Feature Independence를 가정한다.

---

**Reference**

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.
2. Lundberg, S., & Lee, S. I. (2016). An unexpected unity among methods for interpreting model predictions. arXiv preprint arXiv:1611.07478.
