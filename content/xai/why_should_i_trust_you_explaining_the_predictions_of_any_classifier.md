---
title: "\"Why Should I Trust You?\" Explaining the Predictions of Any Classifier"
date: 2022-02-06
categories:
  - "XAI"
tags:
  - "LIME"
sidebar: false
---

## Abstract

이 논문에서는 예측한 데이터 주변에서 해석 가능한 모형을 적합함으로써 임의의 Classifier의 예측 결과를 해석하는 방법(LIME)과 Submodular Optimization을 통해 대표적인 개별 데이터를 제시하는 방법을 제안한다.

## 1. Introduction

“사용자가 모형 자체나 개별 예측 결과를 믿지 않는다면, 그것을 사용하지 않을 것이다”

모형을 의사 결정에 활용하는 경우(ex. 의료 진단, 테러 적발 등에 ML 모형을 사용하는 경우) 개별 예측 결과에 대한 신뢰 여부를 결정하는 것은 굉장히 중요한 문제이다. 즉, 개별 예측 결과에 대한 해석을 제공하는 것이 중요하다.

어떤 모형의 사용 여부를 결정할 때, Validation Set에 대한 정확도 뿐만 아니라, 사용자가 집중해서 확인해야 할 개별 데이터를  선별하는 능력을 척도로 삼을 수 있다.

**Main Contributions**

- LIME: 국부적으로 해석 가능한 모형을 추정하여 임의의 Classifier 또는 Regressor의 예측 결과를 해석할 수 있는 알고리즘
- SP-LIME: Submodular Optimizatoin을 통해 주목해야 할 데이터를 선택하는 방법

## 2. The Case for Explanations

{{<figure src="/xai/lime1.png" width="600">}}

예측 결과를 해석한다는 것은 모형의 예측 결과와 변수 사이의 관계를 이해할 수 있는 자료를 제시하는 것

사용자들은 일반적으로 특정 도메인에 대한 사전 지식을 가지고 있기 때문에, 예측에 대한 이유를 알 수 있을 경우, 그것을 바탕으로 예측 결과에 대한 신뢰 여부를 결정할 수 있다.

모형의 예측 결과를 해석하는 것은 Data Leakage나 Data Shift 등의 문제를 해결하는데 도움이 된다.

모형을 비교함에 있어 항상 모형의 Accuracy가 제1기준이 되는 것은 아니다. 때로는 개별 예측 결과에 대한 해석이 더 중요한 역할을 할 수 있다.

**Desired Characteristics for Explainers**

- Interpretable: 반응변수와 독립변수 사이의 관계에 대한 정보를 누구나 쉽게 이해할 수 있는 형태로 제공할 수 있어야 함
- Local Fidelty: Local Fidelty를 만족한다고 해서 Global Fidelty까지 만족하는 것은 아니지만, 의미있는 해석이라면 적어도 Local Fidelty는 만족시켜야 함 (cf. Global Fidelty를 만족하면 Local Fidelty도 만족함)
- Model-Agnostic
- Global Perspective: 사용자가 개별 데이터에 대한 해석 뿐만 아니라, 모형 자체를 신뢰하기 위해서는 거시적인 관점을 제시하는 것 역시 중요함

## 3. Local Interpretable Model-Agnostic Explanations

### 3.1. Interpretable Data Representations

Original Feature($\mathbf{x}\in\mathbb{R}^d$)를 해석 가능한 표현($\mathbf{x}'\in\{0, 1\}^{d'}$)으로 바꾼다. $\mathbf{x}'$은 각 Feature의 존재(또는 작용) 여부를 나타내는 Binary Vector이다.

### 3.2. Fidelity-Interpretability Trade-off

LIME에 의한 해석 $\xi(\mathbf{x})$는 다음과 같은 식으로 표현된다.

$\displaystyle \xi(\mathbf{x})=\underset{g\in G}{\text{argmin}}\mathcal{L}(f, g, \pi\_\mathbf{x})+\Omega(g)$

$f:\mathbb{R}^d\mapsto\mathbb{R}$ is a model begin explained.

$g\in G:\{0, 1\}^{d'}\mapsto\mathbb{R}$ is an explanation, where $G$ is a class of potentially interpretable models (ex. Linear Models, Decision Trees)

$\Omega(g)$ is a measure of complexity of $g$. (ex. depth of tree, number of nonzero weights)

$\pi_\mathbf{x}(\mathbf{z})$ is a proximity measure between $\mathbf{z}$ to $\mathbf{x}$.

### 3.3. Sampling for Local Exploration

우리의 목표는 $f$에 대한 아무런 가정 없이(Model-Agnostic) $f$와 $g$의 국부적인 차이를 최소화하는 것이다. 따라서 Random Sampling을 통해 $\mathcal{L}(f, g, \pi_\mathbf{x})$를 추정한다.

1. 예측 결과를 해석하고자 하는 (해석 가능한 형태로 변형된) 데이터 $\mathbf{x}'$을 기준으로, $\mathbf{x}'$의 $0$이 아닌 원소를 Random Sampling하여 $\mathbf{x}'$ 주변의 데이터 $\mathbf{z}'\in\{0, 1\}^{d'}$을 생성한다.
2. $\mathbf{z}'$을 Original Feature와 같은 형태($\mathbf{z}\in\mathbb{R}^d$)로 되돌린다.
3. $\mathcal{Z}=\{\mathbf{z}\_1, \ldots, \mathbf{z}\_N\}$를 사용하여 $\xi(\mathbf{x})$를 구하기 위한 식의 최적해를 구한다. 이때 $\pi_\mathbf{x}$를 사용하여 $\mathbf{z}$와 $\mathbf{x}$의 거리를 기준으로 각각의 데이터에 가중치를 부여한다.

### 3.4. Sparse Linear Explanations

$G$ is a class of linear models, s.t. $g(\mathbf{z}')=\mathbf{w}_g\cdot\mathbf{z}'$.

$\mathcal{L}$ is a locally weighted square loss.

$\pi_\mathbf{x}(\mathbf{z})=\exp(-D(\mathbf{x}, \mathbf{z})^2/\sigma^2)$ (exponential kernel), where $D$ is cosine distance for text or $L_2$ distance for images

$\displaystyle \mathcal{L}(f, g, \pi_\mathbf{x})=\sum_{\mathbf{z}, \mathbf{z}'\in\mathcal{Z}}\pi_\mathbf{x}(\mathbf{z})(f(\mathbf{z})-g(\mathbf{z}'))^2$, $\Omega(g)=\infty\mathbb{1}[\Vert\mathbf{w}_g\Vert_0>K]$

Text 데이터의 경우 해석 가능한 표현($\mathbf{x}'$)으로 Bag of Words를, 모형 복잡도의 척도($\Omega$)로 단어의 개수를 사용하고, Image 데이터의 경우에도 이와 유사하게 Super Pixel과 그 개수를 사용한다. 따라서 Lasso 회귀를 통해 $K$개의 Feature를 선택하고, Sparse Linear Regression을 통해 각 Feature에 대한 가중치를 결정한다.

## 4. Submodular Pick for Explaining Models

사용자들이 많은 양의 해석을 전부 확인하기에는 시간이 부족하기 때문에, 사용자가 집중해야 할 개별 데이터를 신중하게 선별하여 제시함으로써 모형 자체에 대한 신뢰도를 높일 수 있다. 본 논문에서는 Text 데이터에 적용할 수 있는 알고리즘에 대해 소개하고 있다.

{{<figure src="/xai/lime2.png" width="200">}}

1. 주어진 데이터 $X$에 대해 Explanation Matrix $\mathcal{W}_{n\times d'}$을 구한다. Interpretable Model($g$)로 Linear Model($g(\mathbf{z}')=\mathbf{w}_g\cdot\mathbf{z}'$)을 사용한 경우 $\mathcal{W}\_{ij}=\vert w\_{g\_{ij}}\vert$가 된다.
2. 각 Feature의 중요도 $I_j=\sqrt{\sum_{i=1}^n\mathcal{W}_{ij}}$를 계산한다.
3. 사람들이 모형을 이해하기 위해 기꺼이 확인하는 데이터의 최대 개수 $B$에 도달할 때까지 중요도가 높은 Feature를 최대한 많이 포괄할 수 있는 개별 데이터를 선택한다.
$\displaystyle \text{Pick}(\mathcal{W}, I)=\underset{{V, \vert V\vert≤B}}{\text{argmax}}\\;c(V, \mathcal{W}, I)$, where $\displaystyle c(V, W, I)=\sum_{j=1}^{d'}\mathbb{1}_{[\exists i\in V:\mathcal{W}_{ij}>0]}I_j$
4. Marginal Coverage $c(V\cup\{i\}, \mathcal{W}, I)-c(V, \mathcal{W}, I)$를 가장 크게 만드는 데이터를 ($\vert V\vert≤B$일 때까지) $V$에 반복적으로 추가함으로써 최적화 문제의 근사해를 구한다.

## 8. Conclusion and Future Work

- 본 논문에서는 Interpretable Model($g$)로 Sparse Linear Model을 사용하는 경우에 대해서만 다루고 있지만, Decision Tree와 같은 더 다양한 종류의 모형을 사용하는 방향으로의 연구가 필요하다.
- Image 데이터에 적용할 수 있는 Submodular Pick 알고리즘에 대한 연구가 필요하다.
- Speech, Video, Medical Domain, Recommender System 등 다양한 분야에 적용하기 위한 연구가 필요하다.
- Theoretical Property(적절한 표본의 수 등)와 Compuatational Optimization(Parallelization, GPU Processing 등)에 대한 연구가 필요하다.

## Memo

- 이 논문은 Tabular Data에 대한 내용을 담고 있지 않다. Tabular Data에 대한 알고리즘이 궁금할 경우 [lime/lime_tabular.py](https://github.com/marcotcr/lime/blob/ce2db6f20f47c3330beb107bb17fd25840ca4606/lime/lime_tabular.py)를 참조해야한다.

---

**Reference**

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining* (pp. 1135-1144).
