---
title: "Curse of Dimensionality"
date: 2021-02-15
weight: 2
---

**As dimension goes higher, the range for a neighborhood gets larger**

{{<figure src="/esl_fig_2.6.png" width="300" height="200">}}

Suppose that our data are uniformly distributed in a $p$-dimensional unit hypercube. Among them, we will define our neighborhood as a fraction $r$ of the observations. Then the expected edge length of our neighborhood will be $r^{1/p}$. Because our input space was a unit hypercube, the edge length of our input space should be $1$. So it is evident that the amount of the data in our neighborhood is $\left(\dfrac{r^{1/p}}{1} \times100\right)\\%=r^{1/p}\\%$ of the total data. This result explains that as dimension goes higher, we should cover larger space to create the neighborhood. For example, if we want to make a neighborhood that captures $10\%$ of the observations in a $10$-dimensional input space, the expected edge length of the neighborhood will be $(0.1)^{1/10}\approx0.8$ which means $80\%$ of the total space. We cannot say anymore that this neighborhood is a local one. One could say that we can solve this problem by reducing $r$ dramatically, but it is not helpful because the fewer observations we average, the higher is the variance of our fit.

{{<figure src="/fig3.png" width="400" height="400">}}

**As dimension goes higher, data gets closer to the boundary of the input space**

{{<figure src="/fig4.png" width="400" height="400">}}

Again, we will consider the situation that our data are uniformly distributed in a $p$-dimensional space, but this time in a $p$-dimensional unit ball. The volume of this unit ball is $\dfrac{\pi^{\frac{N}{2}}}{\Gamma\left(\frac{N}{2}+1\right)}$. Now let's think about the probability that a point will be within distance $x$ from the origin. It is same as the probability that a point is in a ball with radius $x$. So we can say that $P(X≤x)=\dfrac{x^p\pi^{\frac{N}{2}}/\Gamma\left(\frac{N}{2}+1\right)}{\pi^{\frac{N}{2}}/\Gamma\left(\frac{N}{2}+1\right)}=x^p$, where $0≤x≤1$. If we differentiate both sides of the equation with $x$, we can get the pdf of our data point $f(x)=px^{p-1}$. Now we want to know the median distance from the origin to the closest data point. Ordered Statistic will be helpful for solving this problem. Because we are interested in the closest data point from the origin, we will get the cdf of the first ordered statistic $X_{(1)}$.

$f_{X_{(1)}}(x)=N(1-x^p)^{N-1}px^{p-1} \quad (0≤x≤1)$

$F_{X_{(1)}}(x)=1-(1-x^p)^N \quad (0≤x≤1)$

Our criteria is a median distance, so let $F_{X_{(1)}}(x)=\dfrac{1}{2}$ and solve this equation. Finally, it gives us the formula $d(p, N)=\left(1-\dfrac{1}{2}^{1/N}\right)^{1/p}$. Now it is obvious with this formula that as dimension goes higher, our data go closer to an edge of input space. The reason that this represents a problem is that prediction is much more difficult near the edges of the training sample.

{{<figure src="/fig5.png" width="400" height="400">}}

**As dimension goes higher, the sampling density becomes sparser**

{{<figure src="/fig6.jpeg" width="400" height="400">}}

As the picture($9^{1/2}=3$, $8^{1/3}=2$) above, it is evident that the sampling density is proportional to $N^{1/p}$. For a fixed value of $N$, this value becomses smaller as $p$ gets larger, which means that the data is more sparsly distributed in the higher dimension.

{{<figure src="/fig7.png" width="400" height="400">}}

---

**Reference**

1. Elements of Statistical Learning