---
title: "Subset Selection"
date: 2021-03-10
draft: true
TableOfContents: true
---

Subset Selection은 Model Selection의 일환으로, 전체 $X$ 변수 중 회귀 모형을 적합하는데 사용할 변수를 선택하는 과정을 의미한다.

## 1. Best-Subset Selection

가장 단순한 방법은 Best-Subset Selection이다. 이 방법은 가능한 모든 경우의 수를 고려하는 방식이다. $X$ 변수의 개수가 총 $p$개인 상황을 가정하면, 발생 가능한 변수들의 조합은 $2^p$가지이다. 이 $2^p$가지 모형의 $\text{SSE}$를 계산하여 아래와 같은 그래프로 나타낼 수 있다.

![ESL fig 3.5](/ml/fig/esl_fig3.5.png)

최적의 모형은 Bias와 Variance를 모두 고려하여 $\text{SSE}$가 낮은 모형들 가운데 모형의 크기가 가장 작은 모형을 선택한다. 모형 비교의 기준으로 $\text{SSE}$ 외에도 $\text{AIC}$ 등의 값을 사용할 수 있다.

## 2. Forward(Backward)-Stepwise Selection

Best-Subset Selection 방법의 경우 $2^p$개의 모형을 모두 적합해야 하기 때문에 수행에 상당히 많은 시간을 필요로 한다. Forward(Backward)-Stepwise Selection은 이에 비해 수행 속도에 있어 이점을 갖는다.

Clever updating algorithms can exploit the $QR$ decomposition for the current fit to rapidly establish the next candidate(Exercise 3.9)

Backward - the candidate for dropping is the variable with the smallest $Z$-score (Exercise 3.10)

Backward selection can only be used when $N>p$, while forward stepwise can always be used.

## 3. Forward-Stagewise Regression

Starts with an intercept equal to $\bar{y}$, and centered predictors with coefficients initially all $0$.

Identifies the variable most corrleated with the current residual at each step.

Computes the simple linear regression coefficient of the residual on this chosen variable, and then adds it to the current coefficient for that variable.

Continues till none of the variables have correlation with the residuals.