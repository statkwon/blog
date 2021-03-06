---
title: "7. Moving Beyond Linearity"
date: 2021-02-10
TableOfContents: true
weight: 6
---

지금까지는 선형 모형에 초점을 맞추어 왔다. 6장에서 Ridge, Lasso, PCA 등 선형 모형의 복잡성을 줄이는 방법들을 다루긴 하였지만, 이 역시 결국 선형 모형을 사용하는 것이었다. 선형 모형의 경우 해석이 용이하다는 장점을 가지고 있지만, 모형의 선형성에 대한 가정이 전제되기 때문에 예측 정확도는 다소 떨어지는 것이 사실이다. 이러한 단점을 보완하기 위해 7장에서는 선형성 가정을 완화하면서도 해석력을 유지할 수 있는 방법들에 대해서 다루고 있다.

## 7.3. Basis Functions

$$y_i=\beta_0+\beta_1b_1(x_i)+\beta_2b_2(x_i)+\beta_3b_3(x_i)+...+\beta_Kb_K(x_i)+\epsilon_i$$

Basis Functions를 먼저 다루는 것은 아래 나오는 Polynomial Regression과 Step Functions가 Basis Function Approach의 특수한 경우에 해당하기 때문이다. Basis Function Approach는 독립변수 대신 독립변수의 함수 또는 변형에 대해 Least Square Estimate을 사용하여 선형 모형을 적합하는 방식이다. 아래에서 다루는 두 가지 방법(Polynomial Regression, Step Functions) 이외에도 Wavelets이나 Fourier Series 등을 Basis Function으로 사용한다.

## 7.1. Polynomial Regression

$$y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+\beta_3x_i^3+...+\beta_dx_i^d+\epsilon_i$$

Polynomial Regression은 $b_j(x_i)=x_i^j$를 Basis Function으로 사용하는 모형이다. 일반 선형 회귀 모형과 마찬가지로 Least Square Estimate를 사용하여 회귀계수를 추정하기 때문에 그저 독립변수가 $x_i, x_i^2, x_i^3, ..., x_i^d$인 선형 회귀 모형이라고 생각해도 무방하다. 일반적으로 독립변수의 최고 차수는 3차 또는 4차 이하를 사용한다. 모형을 해석할 때, 2차 이상 독립변수들의 회귀계수의 해석이 어렵기 때문에 회귀계수보다 Polynomial Curve의 형태에 집중하는 것이 바람직하다.

$$P(y_i|x_i)=\frac{\text{exp}(\beta_0+\beta_1x_i+\beta_2x_i^2+...+\beta_dx_i^d)}{1+\text{exp}(\beta_0+\beta_1x_i+\beta_2x_i^2+...+\beta_dx_i^d)}$$

위와 같이 로지스틱 회귀 모형에 다항식 구조를 적용하는 것도 가능하다.

## 7.2. Step Functions(Piecewise Constant Regression)

$$y_i=\beta_0+\beta_1C_1(x_i)+\beta_2C_2(x_i)+...+\beta_KC_K(x_i)+\epsilon_i$$

$$C_0(X)+C+1(X)+...+C_K(X)=1$$

Piecewise Constant Regression은 $C_j(x_i)=I(c_j≤x_i<c_{j+1})$를 Basis Function으로 사용하는 모형이다. 독립변수의 범위에 따라 구간을 나누고 각 구간마다 특정 상수값으로 모형을 적합한다. 연속형 독립변수를 범주형으로 변환하는 것이라고 할 수 있다. 위 모형식에서 $\beta_0$ 대신 $C_0(x_i)$를 사용해도 상관없다. Piecewise Polynomial Regression의 경우 데이터가 적은 구간의 예측 정확도가 상대적으로 낮기 때문에 각 구간에서 다항식 대신 상수를 사용하여 모형을 적합하는 것이 Piecewise Constant Regression이다. 하지만 독립변수에 Natural Breakpoints가 없는 경우 각 구간의 특성을 반영하지 못할 수 있다는 단점을 가지고 있다.

{{<figure src="/islr_fig_7.2.png" width="400" height="200">}}

## 7.4. Regression Splines

### 7.4.1. Piecewise Polynomials

$$y_i=\begin{cases} \beta_{01}+\beta_{11}x_i+\beta_{21}x_i^2+\beta_{31}x_i^3+\epsilon_i & \text{if } x_i<c \\ \beta_{02}+\beta_{12}x_i+\beta_{22}x_i^2+\beta_{32}x_i^3+\epsilon_i & \text{if } x_i≥c \end{cases}$$

Piecewise Polynomial Regression은 독립변수의 범위에 따라 구간을 나누고 각 구간마다 저차원의 다항식으로 모형을 적합하는 방식이다(Least Square Estimate 사용). 이때 각 구간이 나뉘는 점들을 Knots라고 부른다. 따라서 $K$개의 Knots를 사용하는 경우, $K+1$개의 다항식을 적합하게 된다. 다항식의 차수가 0인 경우 Piecewise Constant Regression과 동일한 결과를 갖는다. 위 모형식은 Knot가 한 개인 경우의 Cubic Spline에 해당한다.

### 7.4.2. Constraints and Splines

{{<figure src="/islr_fig_7.3.png" width="600" height="400">}}

아무런 제약 없이 Piecewise Polynomial Regression을 적합하는 경우 위 그림과 같이 불연속한 점이 발생할 수 있다. 이러한 경우 도함수와 이계도함수에 제약을 가함으로써 Curve가 연속임과 동시에 smooth하게 만들어줄 수 있다.

### 7.4.4. Choosing the Number and Locations of the Knots

Spline을 적합할 경우 우리는 Knots의 위치와 개수를 결정해야 한다. Knots의 위치를 결정하는 방법에는 두 가지가 있다. 첫번째는 Curve가 급격하게 변화하는 곳에 많은 Knots를, 상대적으로 안정적인 곳에 적은 Knots를 배치하는 방법이다. 하지만 실제로는 이러한 방식보다는 균등한 방식을 사용하여 knots를 배치하는 것이 일반적이다. 컴퓨터 소프트웨어를 사용하여 원하는 자유도에 상응하는 수의 Knots를 일정한 간격으로 배치하는 것이 가능하다. Knots의 개수를 결정하는 방법도 두 가지가 있다. 첫번째는 다양한 개수의 Knots를 배치해보며 최적의 Curve를 형성하는 값을 사용하는 것이다. 다른 방법으로는 CV를 사용하여 결정하는 방법이 있다.

### 7.4.5. Comparison to Polynomial Regression

Polynomial Regression의 경우 유연한 모형을 적합하기 위해서는 높은 차수의 다항식을 사용해야 한다. 반면 Regression Spline은 자유도를 고정한 채로 Knots의 개수를 늘려가며 유연성을 조절하기 때문에 더욱 안정적인 추정이 가능하다.

## 7.6. Local Regression

$$\sum_{i=1}^nK_{i0}(y_i-\beta_0-\beta_1x_i)^2$$

Local Regression은 각각의 Data Point에 대하여 그 근처에 있는 데이터만 가지고 모형을 적합하는 방식이다. 우선 몇 개의 (근처에 있는) 데이터를 사용할 것인지 정해야 한다. 다음으로 사용하고자 하는 데이터들에 대해서 거리에 따라 Weight($K_{i0}=K(x_i, x_0)$)를 부과하고, 위의 식을 최소화하는 계수를 구한다. 매우 간략하게 설명하였지만, 이러한 과정 중에는 Weighitng Function $K$로 어떤 함수를 사용할 것인지, 어떤 종류의 회귀 모형을 적합할 것인지 등 많은 것들을 결정해야 한다. 그 중 가장 중요한 것은 앞서 언급했던 '몇 개의 근처 데이터를 사용할 것인지'이다. 이것은 Smoothing Splines의 Tuning Parameter $\lambda$와 같이 모형의 유연성을 조절하는 기능을 한다. 이론적으로 2차원 이상에도 적용 가능하지만, 3 또는 4차원이 넘어가게 되면 Training Data인 $x_0$에 가까운 데이터가 많이 없기 때문에 모형의 성능이 급격히 떨어진다.

## 7.7. Generalized Additive Models

Generalized Additive Models (GAMs)는 각 변수에 비선형 함수를 적용함으로써 일반 선형 모형을 확장한 형태의 모형이다. 일반 선형 모형과 마찬가지로 반응변수의 형태(수치형, 범주형)에 상관없이 사용이 가능하다.

### 7.7.1. GAMs for Regression Problems

일반적인 Multiple Linear Regression은 다음과 같은 형태이다.

$$y_i=\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+...+\beta_px_{ip}+\epsilon_i$$

이때 독립변수와 반응변수 사이에 비선형 관계를 적용하고 싶은 경우, 위 식의 $\beta_jx_{ij}$를 $f_j(x_{ij})$로 바꾸어주면 된다.

$$y_i=\beta_0+\sum_{j=1}^pf_j(x_{ij})+\epsilon_i=\beta_0+f_1(x_{i1})+f_2(x_{i2})+...+f_p(x_{ip})+\epsilon_i$$

이는 각 독립변수마다 서로 다른 모형을 적용한 후 합한 형태가 된다.

### 7.7.2. GAMs for Classification Problems

$$\text{log}(\frac{\text{P}(X)}{1-\text{P}(X)})=\beta_0+f_1(X_1)+f_2(X_2)+...+f_p(X_p)$$

GAMs는 반응변수가 범주형인 경우에도 사용할 수 있다. 위 식과 같이 Logistic Regression Model에서 $\beta_jX_j$를 $f_j(X_j)$로 바꾸어주면 된다.