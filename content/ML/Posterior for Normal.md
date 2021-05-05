---
title: "Posterior for Normal"
draft: false
---

$p(\theta, \sigma^2|y_1, \ldots, y_n)=p(\theta|\sigma^2, y_1, \ldots, y_n)p(\sigma^2|y_1, \ldots, y_n)$

To calculate the posterior distribution, we should calculate $p(\theta|\sigma^2, y_1, \ldots, y_n)$ and $p(\sigma^2|y_1, \ldots, y_n)$ first.

We can consider $p(\theta|\sigma^2, y_1, \ldots, y_n)$ as the posterior distribution when $\sigma^2$ is known. Then $p(\theta|\sigma^2)$ will be the prior distribution and we will assume that this follows a normal distribution with mean $\mu_0$ and variacne $\tau_0^2$.

$\begin{aligned}
p(\theta|y_1, \ldots, y_n, \sigma^2) &\propto p(y_1, \ldots, y_n|\theta, \sigma^2) \times p(\theta|\sigma^2) \\\\
&\propto \exp{\left\\{-\dfrac{\sum(y_i-\theta)^2}{2\sigma^2}\right\\}}\times\exp{\left\\{-\dfrac{(\theta-\mu_0)^2}{2\tau_0^2}\right\\}} \\\\
&\propto \exp{\left[-\dfrac{1}{2}\left\\{\left(\dfrac{1}{\tau_0^2}+\dfrac{n}{\sigma^2}\right)\theta^2+2\left(\dfrac{\mu_0}{\tau_0^2}+\dfrac{\sum y_i}{\sigma^2}\right)\theta\right\\}\right]} \\\\
&\propto \exp{\left\\{-\dfrac{1}{2}(a\theta^2+2b\theta)\right\\}} \\\\
&=\exp{\left\\{-\dfrac{1}{2}a\left(\theta^2-\dfrac{2b}{a}\theta+\dfrac{b^2}{a^2}\right)+\dfrac{b^2}{2a}\right\\}} \\\\
&\propto \exp{\left\\{-\dfrac{1}{2}a\left(\theta-\dfrac{b}{a}\right)^2\right\\}} \\\\
&=\exp{\left\\{-\dfrac{1}{2}\left(\dfrac{\theta-b/a}{1/\sqrt{a}}\right)^2\right\\}} \sim N\left(\dfrac{b}{a}, \dfrac{1}{a}\right)=N(\mu_n, \tau_n^2)
\end{aligned}$

Now we can check the parameter update.

$\mu_0\rightarrow\mu_n=\dfrac{\frac{1}{\tau_0^2}\times\mu_0+\frac{n}{\sigma^2}\times\bar{y}}{\frac{1}{\tau_0^2}+\frac{n}{\sigma^2}}=\dfrac{\kappa_0\mu_0+n\bar{y}}{\kappa_0+n}=\dfrac{\kappa_0\mu_0+n\bar{y}}{\kappa_n}$

The posterior mean $\mu_n$ is an weighted average between the prior mean and the sample mean. Here the weight of each term is the prior precision $\dfrac{1}{\tau_0^2}$ and the sampling precision $\dfrac{n}{\sigma^2}$. 여기서 Prior Information이 $\kappa_0$개의 Prior Observations에 기반한다고 생각한다면, $\tau_0^2=\dfrac{\sigma^2}{\kappa_0}$라고 나타낼 수 있다. 이를 이용하여 위 식을 변형해주면, $\mu_n$이 각각의 Sample Size를 가중치로 갖는 Prior Mean과 Sample Mean의 가중 평균이라는 보다 직관적인 표현으로 바꾸어줄 수 있다.

$\tau_0^2\rightarrow\tau_n^2=\dfrac{1}{\frac{1}{\tau_0^2}+\frac{n}{\sigma^2}}=\dfrac{\sigma^2}{\kappa_0+n}=\dfrac{\sigma^2}{\kappa_n}$

Posterior precision is the sum of prior precision and the sampling precision.

Now we will think about $p(\sigma^2|y_1, \ldots, y_n)$. We can consider this as a posterior distribution when $p(\sigma^2)$ is a prior which follows a inverse-gamma distribution with parameters $\dfrac{\nu_0}{2}$ and $\dfrac{\nu_0\sigma_0^2}{2}$.

$\begin{aligned}
p(\sigma^2|y_1, \ldots, y_n)&\propto p(\sigma^2)p(y_1, \ldots, y_n|\sigma^2) \\\\
&=p(\sigma^2)\int p(y_1, \ldots, y_n|\theta, \sigma^2)p(\theta|\sigma^2)d\theta
\end{aligned}$

If we solve the integral above, we will get the result that $p(\sigma^2|y_1, \ldots, y_n)$ follows a inverse-gamma distribution with paramters $\dfrac{\nu_n}{2}$ and $\dfrac{\nu_n\sigma_n^2}{2}$. Let's check the paramter update now.

$\nu_0\rightarrow\nu_n=\nu_0+n$

Posterior Sample Size $\nu_n$은 Prior Sample Size와 Data Sample Size의 합과 같다. Why?

$\sigma_0^2\rightarrow\sigma_n^2=\dfrac{\nu_0\sigma_0^2+\sum(y_i-\theta)^2}{\nu_0+n}$

Posterior Variance $\sigma_n^2$는 Prior Sum of Squares와 Data Sum of Squares의 합을 Prior Sample Size와 Data Sample Size의 합으로 나눈 것과 같다.

**Summary**

$\sigma_0^2\rightarrow\sigma_n^2=\dfrac{\nu_0\sigma_0^2+\sum(y_i-\theta)^2}{\nu_0+n}$

$\nu_0\rightarrow\nu_n=\nu_0+n$

$\mu_0\rightarrow\mu_n=\dfrac{\kappa_0\mu_0+n\bar{y}}{\kappa_n}$

$\kappa_0\rightarrow\kappa_n=\kappa_0+n$