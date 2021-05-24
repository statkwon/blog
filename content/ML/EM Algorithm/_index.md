---
title: "EM Algorithm"
draft: true
---

$P(\mathbf{Z}^m\vert \mathbf{Z}, \theta^{(t+1)})=\dfrac{P(\mathbf{T}, \theta^{(t+1)})}{P(\mathbf{Z}, \theta^{(t+1)})}=\dfrac{P(\mathbf{T}, \theta^{(t+1)})}{P(\theta^{(t+1)})}\times\dfrac{P(\theta^{(t+1)})}{P(\mathbf{Z}, \theta^{(t+1)})}=\dfrac{P(\mathbf{T}\vert\theta^{(t+1)})}{P(\mathbf{Z}\vert\theta^{(t+1)})}$

$\Leftrightarrow P(\mathbf{Z}\vert\theta^{(t+1)})=\dfrac{P(\mathbf{T}\vert\theta^{(t+1)})}{P(\mathbf{Z}^m\vert \mathbf{Z}, \theta^{(t+1)})}$

$\Leftrightarrow\log{P(\mathbf{Z}\vert\theta^{(t+1)})}=\log{P(\mathbf{T}\vert\theta^{(t+1)})}-\log{P(\mathbf{Z}^m\vert \mathbf{Z}, \theta^{(t+1)})}$

$\Leftrightarrow l(\theta^{(t+1)};\mathbf{Z})=l_0(\theta^{(t+1)};\mathbf{T})-l_1(\theta^{(t+1)};\mathbf{Z}^m\vert \mathbf{Z})$

$\begin{aligned}
\Leftrightarrow l(\theta^{(t+1)};\mathbf{Z})&=\text{E}\left[l_0(\theta^{(t+1)};\mathbf{T})\vert\mathbf{Z}, \theta^{(t)}\right]-\text{E}\left[l_1(\theta^{(t+1)};\mathbf{Z}^m\vert\mathbf{Z})\vert\mathbf{Z}, \theta^{(t)}\right] \\\\
&\equiv Q(\theta^{(t+1)}, \theta^{(t)})-R(\theta^{(t+1)}, \theta^{(t)})
\end{aligned}$

We will find $\hat{\theta}^{(t+1)}$ which maximizes $Q(\theta^{(t+1)}, \theta^{(t)})$, rather than $l(\theta^{(t+1)};\mathbf{Z})$. Actually it does not matter because both it eventually result in maximizing $l(\theta^{(t+1)};\mathbf{Z})$. Below is the proof for this.

$\begin{aligned}
l(\theta^{(t+1)};\mathbf{Z})-l(\theta^{(t)};\mathbf{Z})&=\left[Q(\theta^{(t+1)}, \theta^{(t)})-Q(\theta^{(t)}, \theta^{(t)})\right]-\left[R(\theta^{(t+1)}, \theta^{(t)})-R(\theta^{(t)}, \theta^{(t)})\right] \\\\
&=
\end{aligned}$

**EM for GMM**

---

**Reference**

1. Elements of Statistical Learning