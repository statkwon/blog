---
title: "Piecewise Polynomial"
draft: true
---

To move beyond linearity, we will transform $X$ and use linear models in this new space of derived input features.

$$f(X)=\sum_{m=1}^M\beta_mh_m(X)$$

Divide the domain of $X$ into contiguous intervals and fit polynomial regression models in each interval.

**Piecewise Constant**

**Piecewise Linear**

**Piecewise Cubic**

The number of paramters to fit piecewise polynomial is $(M+1)\times(K+1)$, where $M$ is the order and $K$ is the number of knots.

Piecewise polynomials are erratic at each knots.