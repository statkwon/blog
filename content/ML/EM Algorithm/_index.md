---
title: "EM Algorithm"
draft: true
---

**EM Algorithm**

We will prove that $\theta^{(j+1)}$ which maximizes $Q(\theta^{(t+1)}\vert \theta^{(t)})$ always satisfies $l(\theta^{(t+1)}\vert\mathbf{Z})≥l(\theta^{(t)}\vert\mathbf{Z})$.

Let $\mathbf{Z}=Z_1, \ldots, Z_n$ be the observed data and $\mathbf{Z}^m=Z_1^m, \ldots, Z_n^m$ be the missing data. We want to find a MLE for $(\mathbf{Z}, \mathbf{Z}^m)$, but it is a clumsy task due to the missing data.

$f(\mathbf{Z}, \mathbf{Z}^m\vert\theta^{(t+1)})=f(\mathbf{Z}\vert\theta^{(t+1)})f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t+1)})$

$\Leftrightarrow l(\theta^{(t+1)}\vert\mathbf{Z})=l(\theta^{(t+1)}\vert\mathbf{Z}, \mathbf{Z}^m)-\log{f(\mathbf{Z}^m\vert \mathbf{Z}, \theta^{(t+1)})}$

Now we will take a conditional expecation with respect to $f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t)})$ to both side of the equation. Then,

$\begin{aligned}
\Leftrightarrow\text{E}\left[l(\theta^{(t+1)}\vert\mathbf{Z})\vert\mathbf{Z}, \theta^{(t)}\right]&=\text{E}\left[l(\theta^{(t+1)}\vert\mathbf{Z}, \mathbf{Z}^m)\vert\mathbf{Z}, \theta^{(t)}\right]-\text{E}\left[\log{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t+1)})}\vert\mathbf{Z}, \theta^{(t)}\right] \\\\
&=Q(\theta^{(t+1)}\vert\theta^{(t)})-R(\theta^{(t+1)}\vert\theta^{(t)})
\end{aligned}$

$\Leftrightarrow l(\theta^{(t+1)}\vert\mathbf{Z})=Q(\theta^{(t+1)}\vert\theta^{(t)})-R(\theta^{(t+1)}\vert\theta^{(t)})$

We have to show that $l(\theta^{(t+1)}\vert\mathbf{Z})≥l(\theta^{(t)}\vert\mathbf{Z})$.

$l(\theta^{(t+1)}\vert\mathbf{Z})-l(\theta^{(t)}\vert\mathbf{Z})=\left[Q(\theta^{(t+1)}\vert\theta^{(t)})-Q(\theta^{(t)}\vert\theta^{(t)})\right]-\left[R(\theta^{(t+1)}\vert\theta^{(t)})-R(\theta^{(t)}\vert\theta^{(t)})\right]$

$Q(\theta^{(t+1)}\vert\theta^{(t)})-Q(\theta^{(t)}\vert\theta^{(t)})≥0$ is trivial, because we assumed that $\theta^{(t+1)}$ maximizes $Q(\theta^{(t+1)}\vert \theta^{(t)})$. Thus, we just need to show that the second part of LHS equals to or less than $0$.

$\begin{aligned}
R(\theta^{(t+1)}\vert\theta^{(t)})-R(\theta^{(t)}\vert\theta^{(t)})&=\text{E}\left[\log{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t+1)})}\vert\mathbf{Z}, \theta^{(t)}\right]-\text{E}\left[\log{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t)})}\vert\mathbf{Z}, \theta^{(t)}\right] \\\\
&=\text{E}\left[\log{\dfrac{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t+1)})}{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t)})}}\Big\vert\mathbf{Z}, \theta^{(t)}\right] \\\\
&≤\log{\text{E}\left[\dfrac{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t+1)})}{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t)})}\Big\vert\mathbf{Z}, \theta^{(t)}\right]} \quad (\text{by Jensen's Inequality}) \\\\
&=\log{\left\\{\int\dfrac{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t+1)})}{f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t)})}f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t)})d\mathbf{Z}^m\right\\}} \\\\
&=\log\int f(\mathbf{Z}^m\vert\mathbf{Z}, \theta^{(t+1)})d\mathbf{Z}^m=0
\end{aligned}$

---

**EM for GMM**

---

**Reference**

1. Elements of Statistical Learning