---
title: "Regression Spline"
draft: false
---

We can obtain regression splines by adding a continuity constraint to piecewise polynomials.

Regression splines are often called the $M$th order spline, a piecewise polynomial of degree $M$, that is continuous and has continuous derivatives of orders $1, \ldots, M-1$ at its know points.

**$M$th order spline with $K$ knots**

$h_j(X)=X^{j-1} \quad (j=1, 2, \ldots, M+1)$

$h_{M+1+l}(X)=(X-\xi_l)_+^M \quad (l=1, 2, \ldots, K)$

$f(X)=\beta_1+\beta_2X\cdots+\beta_{M+1}X^M+\beta_{M+2}(X-\xi_1)_+^M+\cdots+\beta_{M+K+1}(X-\xi_K)_+^M$

---

We will make a proof for the case when $M=3$ and $K=2$.

$f(X)=\beta_1+\beta_2X+\beta_3X^2+\beta_4X^3+\beta_5(X-\xi_1)_+^3+\beta_6(X-\xi_2)_+^3$

We will show the continuity of $f(X)$, $f'(X)$, $f''(X)$ at the knots $\xi_1$, $\xi_2$ ($\xi_1<\xi_2$).

1\) Continuity of $f(X)$

$\begin{aligned}
f(\xi_1-h)&=\beta_1+\beta_2(\xi_1-h)+\beta_3(\xi_1-h)^2+\beta_4(\xi_1-h)^3+\beta_5(\xi_1-h-\xi_1)_+^3+\beta_6(\xi_1-h-\xi_2)_+^3 \\\\
&=\beta_1+\beta_2(\xi_1-h)+\beta_3(\xi_1-h)^2+\beta_4(\xi_1-h)^3
\end{aligned}$

$f(\xi_1+h)=\beta_1+\beta_2(\xi_1+h)+\beta_3(\xi_1+h)^2+\beta_4(\xi_1+h)^3+\beta_5(\xi_1+h-\xi_1)_+^3+\beta_6(\xi_1+h-\xi_2)_+^3$

$\displaystyle\lim_{h\rightarrow0}f(\xi_1-h)=\lim_{h\rightarrow0}f(\xi_1+h)=\beta_1+\beta_2\xi_1+\beta_3\xi_1^2+\beta_4\xi_1^3$

$\displaystyle\therefore\lim_{x\rightarrow\xi_1}f(x)=\beta_1+\beta_2\xi_1+\beta_3\xi_1^2+\beta_4\xi_1^3=f(\xi_1)$

$f(X)$ is continuous at $\xi_1$ and we can show that $f(X)$ is continuous at $\xi_2$ by a similar way.

2\) Continuity of $f'(X)$ at $\xi_1$, $\xi_2$

$\begin{aligned}
f'(\xi_1^-)&=\lim_{h\rightarrow0}\dfrac{f(\xi_1)-f(\xi_1-h)}{h} \\\\
&=\lim_{h\rightarrow0}\dfrac{\beta_2h+2\beta_3\xi_1h+3\beta_4\xi_1^2h+O(h^2)}{h} \\\\
&=\beta_2+2\beta_3\xi_1+3\beta_4\xi_1^2
\end{aligned}$

$\begin{aligned}
f'(\xi_1^+)&=\lim_{h\rightarrow0}\dfrac{f(\xi_1+h)-f(\xi_1)}{h} \\\\
&=\lim_{h\rightarrow0}\dfrac{\beta_2h+2\beta_3\xi_1h+3\beta_4\xi_1^2h+\beta_5(\xi_1+h-\xi_1)_+^3+\beta_6(\xi_1+h-\xi_2)_+^3+O(h^2)}{h} \\\\
&=\beta_2+2\beta_3\xi_1+3\beta_4\xi_1^2
\end{aligned}$

$\therefore\lim_{x\rightarrow\xi_1}f'(x)=\beta_2+2\beta_3\xi_1+3\beta_4\xi_1^2=f'(\xi_1)$

$f'(X)$ is continuous at $\xi_1$ and we can show that $f'(X)$ is continuous at $\xi_2$ by a similar way.

3\) Continuity of $f''(X)$ at $\xi_1$, $\xi_2$

Similarly, $\lim_{x\rightarrow\xi_1}f''(x)=6\beta_4\xi_1^2=f''(\xi_1)$ and $\lim_{x\rightarrow\xi_2}f''(x)=6\beta_4\xi_2^2=f''(\xi_2)$.

We showed that $f(X)$, $f'(X)$ and $f''(X)$ are all continuous at $\xi_1$, $\xi_2$. Thus, $f(X)$ represents a cubic spline with two knots.

---

The number of parameters to fit regression spline is $(M+1)\times(K+1)-M\times K=M+K+1$.

However, regression spline still has a problem that adding a continuity constraint cannot fix the irregularity beyond the boundaries.

---

**Reference**

1. Elements of Statistical Learning