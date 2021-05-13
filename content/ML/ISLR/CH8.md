---
title: "8. Tree-Based Methods"
date: 2021-02-10
TableOfContents: true
weight: 7
---

8장에서는 Tree를 기반으로 하는 방법들에 대해 다루고 있다. 각각의 방법들은 Regression과 Classification의 문제에 모두 적용 가능하다.

## 8.1. The Basics of Decision Trees

가장 기본적인 형태의 Tree-Method는 독립변수 공간을 여러 개의 영역으로 구분하고, 각 영역에 속한 데이터에 대해서 해당 영역의 평균이나 중앙값 등을 사용하여 반응변수를 예측하는 방식이다. 나무 형태로 결과를 나타내기 때문에 Decision Tree라고 부른다.

### 8.1.1. Regression Trees

Regression Tree는 다음과 같은 두 가지 단계의 과정을 거쳐 만들어진다.

1. 독립변수 $X_1, X_2, \ldots, X_p$에 대해, Predictor Space를 서로 겹치지 않는 $J$개의 영역, $R_1, R_2, \ldots, R_J$로 구분한다.
2. $R_j$에 속한 모든 데이터에 대해서는 동일한 방식으로 반응변수를 예측한다. 일반적으로 $R_j$에 속한 데이터들의 평균을 사용한다.

![FIGURE 8.2](/ISLR/FIGURE_8.2.png)

Predictor Space를 $R_1, R_2, \ldots, R_J$로 나누기 위해서는 Recursive Binary Splitting이라는 Top-Down 방식을 사용한다. 모든 데이터가 한 영역에 들어있는 상태를 시작으로 해당 영역을 $\\{X|X_j<s\\}$와 $\\{X|X_j≥s\\}$로 나누었을 때 아래와 같은 식의 $\text{RSS}$를 최소화하는 $X_j$와 $s$를 결정한다. 예를 들어, 위 그림에서는 Years라는 변수와 $4.5$라는 기준값을 가지고 전체 영역을 $R_1$과 $R_2+R_3$로 구분한 것이다.

$$\sum_{j=1}^J\sum_{i \in R_j}{(y_i-\hat{y}_{R_j})^2}$$

이후 나누어진 각 영역들에 대해서 동일한 과정을 반복 시행하며 특정한 정지 기준에 도달할 때까지 분할한다. 위 그림에서는 Years와 $4.5$를 기준으로 첫 번째 분할을 한 후, Hits와 $117.5$를 기준으로 $R_2+R_3$를 $R_2$와 $R_3$로 나누었다.

하지만 Tree를 계속해서 분할하기만 하는 것은 예측 모형을 만드는데 있어 결코 도움이 되지 않는다. 계속된 분할로 만들어진 복잡한 Tree는 Overfittting 될 확률이 높기 때문에 Test 데이터에 적용이 어렵다. 이러한 문제에 대한 대안으로 Predictor Space를 분할할 때 발생하는 $\text{RSS}$의 감소량이 일정한 Threshold를 넘기는 경우에만 분할이 이루어지도록 하는 방법을 생각해볼 수 있다. 이러한 방법을 사용하면 Tree의 크기를 확실히 줄일 수 있다. 하지만 동시에 지나치게 근시안적인 방법이라는 단점을 가지고 있다. 현 단계에서는 의미없어 보이는 분할이 이후 단계에서는 유효할 수도 있는데, 앞선 방법으로는 이를 반영할 수가 없다. 따라서 우리는 Tree Pruning이라는 방법을 해결책으로 사용한다.

Tree Pruning은 우선 Tree를 최대한으로 성장시킨 후, 가지를 치며 가장 작은 Test Error를 갖는 Subtree를 얻어내는 방식이다. 이때 각 Subtree의 Test Error를 계산하기 위해 Cross-Validation이나 Validation Set Approach를 사용할 수 있다. 하지만 가능한 모든 경우의 Subtree에 대해 Cross-Validation Error를 계산하는 것은 상당히 오랜 시간을 필요로 하기 때문에, 일반적으로 Cost Complexity Pruning이라는 방식을 사용한다. Cost Complexity Pruning은 가능한 모든 경우의 Subtree가 아닌, 아래와 같은 식을 작게 하는 Subtree만을 고려하는 방식이다. 이때 $|T|$는 Tree $T$의 Terminal Node의 갯수를 의미한다.

$$\sum_{m=1}^{|T|}\sum_{x_i \in R_m}{(y_i-\hat{y}_{R_m})^2}+\alpha|T|$$

Tuning Parameter라고 할 수 있는 $\alpha$는 Subtree의 복잡도와 Training 데이터에 대한 Fit 사이의 Trade-Off를 조절한다.