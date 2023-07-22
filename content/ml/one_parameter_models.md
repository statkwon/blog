---
title: "One Parameter Models"
date: 2021-02-16
categories:
  - "ML"
tags:
  - "Bayesian"
sidebar: false
---

## The Binomial Model

### Inference for Exchangable Binary Data

**Posterior Inference Under a Unifrom Prior Distribution**

$Y_i$가 평균이 $\theta$인 i.i.d. Binary Variable인 경우 Sampling Model은 $p(y_1, \ldots, y_n|\theta)=\theta^{\sum y_i}(1-\theta)^{n-\sum y_i}$와 같다.

이때 임의의 $\theta_a$와 $\theta_b$에 대한 Relative Probability를 계산하면

$ \begin{align}
\dfrac{p(\theta_a | y_1, \ldots, y_n)}{p(\theta_b | y_1, \ldots, y_n)}&=\dfrac{\theta_a^{\sum y_i}(1-\theta_a)^{n-\sum y_i} \times p(\theta_a)/p(y_1, \ldots, y_n)}{\theta_b^{\sum y_i}(1-\theta_b)^{n-\sum y_i} \times p(\theta_b)/p(y_1, \ldots, y_n)}
\\\\
&=\left\(\dfrac{\theta_a}{\theta_b}\right\)^{\sum y_i}\left\(\dfrac{1-\theta_a}{1-\theta_b}\right\)^{n-\sum y_i}\dfrac{p(\theta_a)}{p(\theta_b)}
\end{align}$

가 된다. 여기서 $\theta_b$에 대한 $\theta_a$의 확률이 $\sum_{i=1}^n y_i$에 의해 $y_1, \ldots, y_n$에만 의존한다는 것을 알 수 있다.

$\text{Pr}(\theta \in A|Y_1=y_1, \ldots, Y_n=y_n)=\text{Pr}(\theta \in A|\sum_{i=1}^n Y_i=\sum_{i=1}^n y_i)$

이는 곧 $\sum_{i=1}^n Y_i$가 주어진 데이터로부터 $\theta$에 대해 알 수 있는 모든 정보를 포함하고 있다는 것, 즉, $\sum_{i=1}^n Y_i$가 $\theta$와 $p(y_1, \ldots, y_n|\theta)$에 대한 Sufficient Statistic이라는 의미를 갖는다. 따라서 $p(y_1, \ldots, y_n|\theta)$ 대신 $p(y|\theta)$를 사용한다. 이때 $Y=\sum_{i=1}^n Y_i$는 $B(n, \theta)$의 Bionmial Distribution을 따른다.

이렇게 Sampling Model이 Binomial Distribution을 따르는 경우에 Prior Distribution으로 Uniform한 분포를 사용하면, 다음과 같은 Posterior Distribution을 구할 수 있다.

$\begin{align}
p(\theta|y)&=\dfrac{p(y|\theta)p(\theta)}{p(y)}
\\\\
&=\dfrac{{n \choose y} \theta^y(1-\theta)^{n-y}p(\theta)}{p(y)}
\\\\
&=c(y)\theta^y(1-\theta)^{n-y}p(\theta)
\end{align}$

이때 $p(\theta)=1$이므로 $p(\theta|y)$가 Distribution Function이라는 것을 이용하여 Normalizing Constant $c(y)$의 정확한 값을 계산할 수 있다.

$\begin{align}
1&=\displaystyle \int_0^1c(y)\theta^y(1-\theta)^{n-y}d\theta
\\\\
&=c(y)\displaystyle \int_0^1\theta^y(1-\theta)^{n-y}d\theta
\\\\
&=c(y)\dfrac{\Gamma(y+1)\Gamma(n-y+1)}{\Gamma(n+2)}
\end{align}$

$\therefore c(y)=\dfrac{\Gamma(n+2)}{\Gamma(y+1)\Gamma(n-y+1)}$

이를 위의 Posterior Distribution을 구하는 식에 대입하면,

$\begin{align}
p(\theta|y)&=\dfrac{\Gamma(n+2)}{\Gamma(y+1)\Gamma(n-y+1)}\theta^y(1-\theta)^{n-y}
\\\\
&=\dfrac{\Gamma(n+2)}{\Gamma(y+1)\Gamma(n-y+1)}\theta^{(y+1)-1}(1-\theta)^{(n-y+1)-1}
\\\\
&=\text{beta}(y+1, n-y+1)
\end{align}$

$p(\theta|y)$가 Beta Distribution을 따르는 것을 알 수 있다.

**Posterior Distributions Under Beta Prior Distributions**

여기까지의 과정은 Prior Distribution이 Uniform하다는 것을 전제로 한다. 그런데 Unifrom Distribution $p(\theta)=1$, $\theta \in [0, 1]$은 사실 $(1, 1)$을 Parameter로 갖는 Beta Distribution으로 볼 수 있다.

$p(\theta)=\dfrac{\Gamma(2)}{\Gamma(1)\Gamma(1)}\theta^{1-1}(1-\theta)^{1-1}=\dfrac{1}{1 \times 1}1 \times 1=1$

이를 정리하면 $\theta \sim \text{beta}(1, 1)$이고 $Y|\theta \sim B(n, \theta)$인 경우, $\theta|Y \sim \text{beta}(1+y, 1+n-y)$가 된다. 즉, Posterior Distribution의 두 Parameter는 Prior Distribution의 두 Parameter에 각각 $y$와 $n-y$를 더한 것과 같다. 이는 Prior Distribution이 임의의 Parameter $(a, b)$를 갖는 Beta Distribution을 따르는 경우에도 성립한다.

$\theta \sim \text{beta}(a, b)$이고 $Y|\theta \sim B(n, \theta)$인 상황을 가정하면,

$\begin{align}
p(\theta|y)&=\dfrac{p(\theta)p(y|\theta)}{p(y)}
\\\\
&=\dfrac{1}{p(y)}\times\dfrac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\theta^{a-1}(1-\theta)^{b-1} \times {n \choose y}\theta^y(1-\theta)^{n-y}
\\\\
&=c(n, y, a, b) \times \theta^{a+y-1}(1-\theta)^{b+n-y-1}
\end{align}$

가 되고, 마찬가지로 $c(\cdot)$의 값을 구하여 대입하면 $p(\theta|y)$가 $\text{beta}(a+y, b+n-y)$를 따름을 확인할 수 있다. 이는 Prior Distribution의 두 Parameter $(a, b)$에 각각 $y$와 $n-y$를 더한 것과 같다.

**Conjugacy**

위와 같이 Prior Distribution $p(\theta)$와 Posterior Distribution $p(\theta|y)$가 동일한 분포를 따르게 하는 분포 $\mathcal{P}$를 Sampling Model $p(y|\theta)$에 대한 Conjugate이라고 한다. 이러한 Conjugate Prior를 사용하는 경우 Prior Information을 정확하게 나타내지는 못하지만, Posterior Distribution을 계산하는 과정을 보다 쉽게 만들 수 있다.

**Combining Information**

Posterior Distribution이 $\text{beta}(a+y, b+n-y)$를 따를 때 평균은 다음과 같다.

$E[\theta|y]=\dfrac{a+y}{a+b+n}$

위 식을 변형하면 Posterior Expectation이 Prior Expectation과 Sample Average의 가중 평균인 것을 확인할 수 있다.

$\begin{align}
E[\theta|y]&=\dfrac{a+y}{a+b+n}
\\\\
&=\dfrac{a+b}{a+b+n}\dfrac{a}{a+b}+\dfrac{n}{a+b+n}\dfrac{y}{n}
\\\\
&=\dfrac{a+b}{a+b+n}\times\text{posterior expectation}+\dfrac{n}{a+b+n}\times\text{data average}
\end{align}$

이는 Sample Size $n$에 따라 데이터가 Posterior Information에 주는 영향력이 달라지는 직관적 구조를 따른다.

**Prediction**

Bayesian Inference의 중요한 기능 중 하나는 새로운 데이터에 대한 Predictive Distribution을 구할 수 있다는 점이다. 주어진 데이터 $y_1, \ldots, y_n$과 동일한 모집단에 속하는 새로운 데이터 $\tilde{Y} \in \{0, 1\}$에 대한 Predictive Distribution은 $\{Y_1=y_1, \ldots, Y_n=y_n\}$이 주어졌을 때 $\tilde{Y}$의 조건부 분포를 따른다.

$\begin{align}
\text{Pr}(\tilde{Y}=1|y_1, \ldots, y_n)&=\displaystyle \int \text{Pr}(\tilde{Y}=1, \theta|y_1, \ldots, y_n)d\theta
\\\\
&=\int \text{Pr}(\tilde{Y}=1|\theta, y_1, \ldots, y_n)p(\theta|y_1, \ldots, y_n)d\theta
\\\\
&=\int \theta p(\theta|y_1, \ldots, y_n)d\theta
\\\\
&=E[\theta|y_1, \ldots, y_n]=\dfrac{a+\sum_{i=1}^n y_i}{a+b+n}
\\\\
\text{Pr}(\tilde{Y}=0|y_1, \ldots, y_n)&=1-E[\theta|y_1, \ldots, y_n]=\dfrac{b+\sum_{i=1}^n (1-y_i)}{a+b+n}
\end{align}$

위 식을 통해 Predictive Distribution은 Unknown Quantities의 영향을 받지 않고, Observed Data의 영향을 받는다는 사실을 확인 수 있다.

### Confidence Regions

빈도론적 관점에서 모수에 대한 신뢰 구간을 구하는 것처럼, Bayesian Methods에도 Bayesian Coverage라는 것이 존재한다.

Confidence Interval: $\text{Pr}(l(Y)<\theta<u(Y)|\theta)=0.95$

Bayesian Coverage: $\text{Pr}(l(y)<\theta<u(y)|Y=y)=0.95$

빈도론적 관점에서의 신뢰 구간이 데이터를 관찰하기 전에 $\theta$를 포함할 확률이 $95$%인 구간을 의미한다면, Bayesian Coverage는 주어진 데이터에 기반하여 $\theta$의 참값의 위치에 대한 정보를 나타내는 것으로 생각할 수 있다.

데이터 $Y=y$를 관찰한 후, 이를 신뢰 구간에 반영하면

$\text{Pr}(l(y)<\theta<u(y)|\theta)=\begin{cases} 0 & \text{if } \theta \not\in [l(y), u(y)] \\\\ 1 & \text{if } \theta \in [l(y), u(y)] \end{cases}$

와 같은 결과를 얻게 된다. 신뢰 구간은 이렇게 주어진 데이터에 대한 정보를 반영하지 못한다는 한계를 갖는다.

**Highest Posterior Density (HPD) Region**

Quantile-Based Interval과 같은 신뢰 구간을 사용하는 경우 구간 밖에 있는 $\theta$ 값들에 대한 확률이 구간 안보다 높은 경우가 발생한다. 이러한 점을 보완하기 위해 좀 더 엄격한 구간을 정의한 것이 HPD이다.

> **Definition** *A $100 \times (1-\alpha)$% HPD region consists of a subset of the parameter space, $s(y) \subset \Theta$ such that*
> 
> 1. $\text{Pr}(\theta \in s(y)|Y=y)=1-\alpha$
> 2. *If $\theta_a \in s(y)$, and $\theta_b \not \in s(y)$, then $p(\theta_a|Y=y)>p(\theta_b|Y=y)$.*

HPD의 정의는 위와 같다. 하지만 이러한 정의보다 아래의 그래프를 통해 보다 직관적인 이해가 가능하다.

## The Poisson Model

### Posterior Inference

이번에는 $Y_1, \ldots, Y_n$이 평균이 $\theta$인 Poisson Distribution을 따르는 i.i.d. Random Variable이라고 하자. 이러한 경우 Sampling Model은 다음과 같다.

$\begin{align}
\text{Pr}(Y_1=y_1, \ldots, Y_n=y_n|\theta)&=\prod_{i=1}^n p(y_i|\theta)
\\\\
&=\prod_{i=1}^n \dfrac{1}{y_i!}\theta^{y_i}e^{-\theta}
\\\\
&=c(y_1, \ldots, y_n)\theta^{\sum y_i}e^{-n\theta}
\end{align}$

3.1에서와 마찬가지로 임의의 $\theta_a$와 $\theta_b$에 대한 Relative Probability를 계산하면

$\begin{align}
\dfrac{p(\theta_a|y_1, \ldots, y_n)}{p(\theta_b|y_1, \ldots, y_n)}&=\dfrac{c(y_1, \ldots, y_n)}{c(y_1, \ldots, y_n)}\dfrac{e^{-n\theta_a}}{e^{-n\theta_b}}\dfrac{\theta_a^{\sum y_i}}{\theta_b^{\sum y_i}}\dfrac{p(\theta_a)}{p(\theta_b)}
\\\\
&=\dfrac{e^{-n\theta_a}}{e^{-n\theta_b}}\dfrac{\theta_a^{\sum y_i}}{\theta_b^{\sum y_i}}\dfrac{p(\theta_a)}{p(\theta_b)}
\end{align}$

가 된다. 여기서도 $\sum_{i=1}^n Y_i$가 Sufficient Statistic임을 확인할 수 있다. 이때 $\sum_{i=1}^n Y_i|\theta$는 $\text{Poisson}(n\theta)$를 따른다.

이제 앞서 다룬 Conjugacy 개념을 활용하여 Poisson Sampling Model에 대한 Conjugate Prior를 구해보고자 한다.

$\begin{align}
p(\theta|y_1, \ldots, y_n) &\propto p(\theta) \times p(y_1, \ldots, y_n|\theta)
\\\\
&\propto p(\theta) \times \theta^{\sum y_i}e^{-n\theta}
\end{align}$

$\theta^{c_1}e^{-c_2\theta}$와 같은 구조를 갖는 분포로 Gamma Distribution이 있다. 따라서 Poisson Sampling Model에 대한 Conjugate Prior는 Gamma Distribution이 된다.

따라서 우리는 $\theta \sim \text{Gamma}(a, b)$이고 $Y|\theta \sim \text{Poisson}(\theta)$에 대한 Posterior Distribution을 구해야 한다.

$\begin{align}
p(\theta|y_1, \ldots, y_n)&=p(\theta) \times p(y_1, \ldots, y_n|\theta)/p(y_1, \ldots, y_n)
\\\\
&=\\{\theta^{a-1}e^{-b\theta}\\} \times \\{\theta^{\sum y_i}e^{-n\theta}\\} \times c(y_1, \ldots, y_n, a, b)
\\\\
&=\\{\theta^{a+\sum y_i-1}e^{-(b+n)\theta}\\} \times c(y_1, \ldots, y_n, a, b)
\end{align}$

$p(\theta|y)$가 Distribution Function이라는 것을 이용하여 Normalizing Constant를 계산하고 위 식에 대입하면 $p(\theta|y)$가 $\displaystyle \text{Gamma}(a+\sum_{i=1}^n Y_i, b+n)$를 따른다는 것을 알 수 있다.

Posterior Expectation 역시 Binary Sampling Model의 경우와 마찬가지로 Prior Expectation과 Sample Average의 가중 평균의 형태를 갖는다.

$\begin{align}
E[\theta|y_1, \ldots, y_n]&=\dfrac{a+\sum y_i}{b+n}
\\\\
&=\dfrac{b}{b+n}\dfrac{a}{b}+\dfrac{n}{b+n}\dfrac{\sum y_i}{n}
\end{align}$

$\tilde{Y}$에 대한 Predictive Distribution을 구해보면

$\begin{align}
p(\tilde{y}|y_1, \ldots, y_n)&=\displaystyle \int_0^\infty p(\tilde{y}|\theta, y_1, \ldots, y_n)p(\theta|y_1, \ldots, y_n)d\theta
\\\\
&=\int p(\tilde{y}|\theta)p(\theta|y_1, \ldots, y_n)d\theta
\\\\
&=\int \left\\{\dfrac{1}{\tilde{y}!}e^{\tilde{y}}e^{-\theta}\right\\}\left\\{\dfrac{(b+n)^{a+\sum y_i}}{\Gamma(a+\sum y_i)}\theta^{a+\sum y_i-1}e^{-(b+n)\theta}\right\\}d\theta
\\\\
&=\dfrac{(b+n)^{a+\sum y_i}}{\Gamma(\tilde{y}+1)\Gamma(a+\sum y_i)}\int_0^\infty \theta^{a+\sum y_i+\tilde{y}-1}e^{-(b+n+1)\theta}d\theta
\\\\
&=\dfrac{\Gamma(a+\sum y_i+\tilde{y})}{\Gamma(\tilde{y}+1)\Gamma(a+\sum y_i)}\left\(\dfrac{b+n}{b+n+1}\right\)^{a+\sum y_i}\left\(\dfrac{1}{b+n+1}\right\)^{\tilde{y}}
\end{align}$

가 된다. 이는 $(a+\sum y_i, b+n)$을 Parameter로 갖는 Negative Binomial Distribution에 해당한다. 세부적인 연산 과정은 Details를 눌러 확인할 수 있다.

{{<rawhtml>}}
<details>
<summary>Details</summary>
$1=\displaystyle \int_0^\infty \dfrac{b^a}{\Gamma(a)}\theta^{a-1}e^{-b\theta}d\theta, \quad \forall a, b>0$ <br>
$\displaystyle \int_0^\infty \theta^{a-1}e^{-b\theta}d\theta=\dfrac{\Gamma(a)}{b^a}, \quad \forall a, b>0$ <br>
$\displaystyle \int_0^\infty \theta^{a+\sum y_i+\tilde{y}}e^{-(b+n+1)\theta}d\theta=\dfrac{\Gamma(a+\sum y_i+\tilde{y})}{(b+n+1)^{a+\sum y_i+\tilde{y}}}$
</details>
{{</rawhtml>}}

## Exponential Families and Conjugate Priors

Binomial Distribution과 Poisson Distribution은 모두 One-Parameter Exponential Family에 속한다. One-Parameter Exponential Family란 분포 함수가 $p(y|\phi)=h(y)c(\phi)e^{\phi t(y)}$의 형태로 표현될 수 있는 분포를 의미한다. 이때 $\phi$는 Unknown Parameter이고 $t(y)$는 Sufficient Statistic이다.

---

**Reference**

1. Hoff, P. D. (2009). A first course in Bayesian statistical methods (Vol. 580). New York: Springer.
