$$
\begin{aligned}
\widehat{D} &\equiv  \widehat{D}(q, q_d(s))\\
s^* &= \underset{s}{\arg\min}\widehat{D}\\
D &= \widehat{D}(q, q_d(s^*))\\
\quad q\in\mathbb{R}^n,\,& W\in\mathbb{R}^{n\times m},\,\xi\in\mathbb{R}^m,\,
\frac{\partial \widehat{D}}{\partial q}\in\mathbb{R}^{1\times n}\\
\dot{q} &= W(q(t))\xi\\
\dot{D} &= \xi_N^T\xi
\end{aligned}
$$

### Normal

$$
\dot{D} = \frac{\partial \widehat{D}}{\partial q}\dot{q}+\underbrace{\left.\frac{\partial \widehat{D}}{\partial q_d}\frac{\partial q_d}{\partial s}\right|_{s^*}}_0\dot{s^*}
=\frac{\partial \widehat{D}}{\partial q}W\xi\\
\implies \xi_N = W^T(q(t))\frac{\partial \widehat{D}}{\partial q}^T
$$

### Tangente

$$
\frac{\partial q_d}{\partial s} = W(q_d(s^*))\xi_T\\
\implies \xi_T = W^\dagger(q_d(s^*))\frac{\partial q_d}{\partial s}
$$

---

### Ortogonalidade

$$
\begin{aligned}
\xi_N^T\xi_T &= 0\\
\frac{\partial \widehat{D}}{\partial q}W(q(t))W^\dagger(q_d(s^*))\frac{\partial q_d}{\partial s} &=0 \iff \frac{\partial \widehat{D}}{\partial q}W(q(t))W^\dagger(q_d(s^*)) = \alpha\frac{\partial \widehat{D}}{\partial q_d}\\
&\implies (W^\dagger(q_d(s^*)))^TW^T(q(t))\frac{\partial \widehat{D}}{\partial q}^T-\alpha\frac{\partial \widehat{D}}{\partial q_d}^T=0\\
&= \begin{bmatrix}\underbrace{(W^\dagger(q_d(s^*)))^TW^T(q(t))}_{\Omega(q,q_d)} & -\alpha I\end{bmatrix}\begin{bmatrix}\frac{\partial \widehat{D}}{\partial q}^T\\\frac{\partial \widehat{D}}{\partial q_d}^T\end{bmatrix}\\
&= \begin{bmatrix}\Omega& -\alpha I\end{bmatrix}\nabla\widehat{D}
\end{aligned}
$$

Então $\nabla\widehat{D}$ pertence a $\text{ker}\left(\begin{bmatrix}\Omega& -\alpha I\end{bmatrix}\right)$

---

## Considerando natureza matricial

Seja $q=\begin{bmatrix}h_1^T & h_2^T & \dots & h_m^T\end{bmatrix}^T\in\mathbb{R}^{m^2}, H=\begin{bmatrix}h_1 & h_2 & \dots & h_m\end{bmatrix} \in \text{GL}(m, \mathbb{R})$, $\xi= \xi_1 e_1 + \xi_2 e_2 + \dots +\xi_r e_r\in\mathbb{R}^r$, $\begin{bmatrix}e_1 & e_2 & \dots & e_r\end{bmatrix} = I_{r\times r}$

$$
\begin{aligned}
\dot{H} &= S(\xi(t))H(t) \\
\implies \dot{h}_i = S(\xi)h_i &= S(\xi_1 e_1 + \xi_2 e_2 + \dots +\xi_r e_r)h_i\\
&= S(e_1)h_i\xi_1  + S(e_2)h_i\xi_2  + \dots+ S(e_r)h_i\xi_r\\
&= \begin{bmatrix} S(e_1)h_i & S(e_2)h_i & \dots & S(e_r)h_i \end{bmatrix}\xi\\
\therefore \dot{q} = \begin{bmatrix}\dot{h}_1 \\ \dot{h}_2 \\ \vdots \\ \dot{h}_m\end{bmatrix} &= \underbrace{
    \begin{bmatrix}
    S(e_1)h_1 & S(e_2)h_1 & \dots & S(e_r)h_1\\
    S(e_1)h_2 & S(e_2)h_2 & \dots & S(e_r)h_2\\
    \vdots & \vdots & \ddots & \vdots\\
    S(e_1)h_m & S(e_2)h_m & \dots & S(e_r)h_m\\
    \end{bmatrix}
}_{W(q)}\xi
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial D}{\partial q} W(q) &= \begin{bmatrix}\frac{\partial D}{\partial q_1} & \frac{\partial D}{\partial q_2} & \dots & \frac{\partial D}{\partial q_{m^2}}\end{bmatrix}W(q) \\&= \begin{bmatrix}\frac{\partial D}{\partial h_1} & \frac{\partial D}{\partial h_2} & \dots & \frac{\partial D}{\partial h_{m}}\end{bmatrix}\begin{bmatrix}
    S(e_1)h_1 & S(e_2)h_1 & \dots & S(e_r)h_1\\
    S(e_1)h_2 & S(e_2)h_2 & \dots & S(e_r)h_2\\
    \vdots & \vdots & \ddots & \vdots\\
    S(e_1)h_m & S(e_2)h_m & \dots & S(e_r)h_m\\
    \end{bmatrix}\\
&= \begin{bmatrix}\sum_i\frac{\partial D}{\partial h_i}S(e_1)h_i & \sum_i\frac{\partial D}{\partial h_i}S(e_2)h_i & \dots & \sum_i\frac{\partial D}{\partial h_i}S(e_r)h_i\end{bmatrix}\\
&= \sum_j\sum_i\left(\frac{\partial D}{\partial h_i}S(e_j)h_i\right)e_j^T
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial D}{\partial q} W(q) &= -\frac{\partial D}{\partial q_d} W(q_d)\\
\implies &\sum_i\frac{\partial D}{\partial h_i}S(e_j)h_i + \frac{\partial D}{\partial h_{di}}S(e_j)h_{di} = 0\ \forall\, j\\
\end{aligned}
$$

Mas, $S(e_j)h_i = \dot{h}_i$ para o twist $\xi=e_j$

$$
\therefore \frac{d}{dt}D(H, H_d)=0,\ \xi=e_j\ \forall \,j
$$

Suponha $\xi= \alpha_1 e_1 + \alpha_2 e_2 + \dots + \alpha_r e_r$,
$$
\begin{aligned}
&\sum_i\frac{\partial D}{\partial h_i}S(\xi)h_i + \frac{\partial D}{\partial h_{di}}S(\xi)h_{di}\\
&= \sum_i\frac{\partial D}{\partial h_i}S(\alpha_1 e_1 + \alpha_2 e_2 + \dots + \alpha_r e_r)h_i + \frac{\partial D}{\partial h_{di}}S(\alpha_1 e_1 + \alpha_2 e_2 + \dots + \alpha_r e_r)h_{di}\\
&= \sum_i\frac{\partial D}{\partial h_i}\left(\alpha_1S( e_1) + \alpha_2S( e_2) + \dots + \alpha_rS( e_r)\right)h_i + \frac{\partial D}{\partial h_{di}}\left(\alpha_1S( e_1) + \alpha_2S( e_2) + \dots + \alpha_rS( e_r)\right)h_{di}\\
&= \alpha_1\sum_i\left(\frac{\partial D}{\partial h_i}S( e_1)h_i + \frac{\partial D}{\partial h_{di}}S(e_1)h_{di}\right) + \alpha_2\sum_i\left(\frac{\partial D}{\partial h_i}S( e_2)h_i + \frac{\partial D}{\partial h_{di}}S(e_2)h_{di}\right) + \dots\\
&=0
\end{aligned}
$$

Portanto, os vetores normal e tangente serão ortogonais se
$$
\frac{d}{dt}D(H, H_d)=0,\ \forall \ \xi
$$

Se $D(H,H_d) = f(H_d^{-1}H)$, então a propriedade é atendida:

Note que
$$
\begin{aligned}
\frac{\partial A^{-1}(t)A(t)}{\partial t} &= \frac{\partial A^{-1}(t)}{\partial t}A(t) + A^{-1}(t)\frac{\partial A(t)}{\partial t} = 0\\
\implies& \frac{\partial A^{-1}(t)}{\partial t} = -A^{-1}(t)\frac{\partial A(t)}{\partial t}A^{-1}(t)
\end{aligned}
$$

$$
\begin{aligned}
\frac{d}{dt}f(H_d^{-1}H) &= \frac{\partial f}{\partial (H_d^{-1}H)}\frac{\partial(H_d^{-1}H)}{\partial t}
= \frac{\partial f}{\partial (H_d^{-1}H)}\left(-H_d^{-1}\frac{\partial H_d}{\partial t}H_d^{-1}H + H_d^{-1}\frac{\partial H}{\partial t}\right)\\
&=\frac{\partial f}{\partial (H_d^{-1}H)}\left(-H_d^{-1}S(\xi)H_dH_d^{-1}H + H_d^{-1}S(\xi)H\right)\\
&= 
\frac{\partial f}{\partial (H_d^{-1}H)}\left(-H_d^{-1}S(\xi)H + H_d^{-1}S(\xi)H\right)\\
&=0
\end{aligned}
$$

### Interpretação em $\mathbb{R}^3$:

Seja $p=\begin{bmatrix}p_1 & p_2 & p_3 \end{bmatrix}^T$, $p_d=\begin{bmatrix}p_{d1} & p_{d2} & p_{d3} \end{bmatrix}^T$ as posições atual e desejada. E seja $D(p, p_d) = \|p-p_d\|$

---
<!-- \begin{bmatrix} \frac{\partial D}{\partial h_1} & \frac{\partial D}{\partial h_2} & \dots & \frac{\partial D}{\partial h_1} & \frac{\partial D}{\partial h_m}\frac{\partial D}{\partial h_{d1}} & 
\dots & \frac{\partial D}{\partial h_{dm}}\end{bmatrix}&
\begin{bmatrix}S(e_j) & 0  & 0\\ 0  &\ddots & 0\\  0 & \dots &S(e_j)\\\end{bmatrix}
\begin{bmatrix}h_1 \\ h_2 \\ \vdots \\ h_m \\ h_{d1} \\ \vdots \\ h_{dm}\end{bmatrix}=0
\\ -->
<!-- \begin{bmatrix}\frac{\partial D}{\partial h_1}S(e_j) & 0 & \dots & 0 & 0 & \dots & 0\\
0 & \frac{\partial D}{\partial h_2}S(e_j) & \dots & 0& 0 & \dots & 0\\
\vdots & \vdots & \ddots & \vdots & \vdots & \dots & 0\\
0 & 0 & \dots & \frac{\partial D}{\partial h_{m}}S(e_j) & 0 & \dots & 0\\
0 & 0 & \dots & 0 & \frac{\partial D}{\partial h_{d1}}S(e_j) & \dots & 0\\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots\\
0 & 0 & \dots & 0 & 0 & \dots &\frac{\partial D}{\partial h_{dm}}S(e_j)\\
\end{bmatrix}
\end{aligned} -->

### Teste em $S^1$

$$
\begin{aligned}
q &= \begin{bmatrix} x\\y\end{bmatrix},\,\dot{q}=\begin{bmatrix} y\\-x\end{bmatrix}\xi=W\xi\\
W^\dagger(q_d(s^*)) &= \begin{bmatrix}\frac{y_{d}}{x_{d}^{2} + y_{d}^{2}} & - \frac{x_{d}}{x_{d}^{2} + y_{d}^{2}}\end{bmatrix}\\
\Omega&=(W^\dagger(q_d(s^*)))^TW^T(q(t)) = \begin{bmatrix}\frac{y y_{d}}{x_{d}^{2} + y_{d}^{2}} & - \frac{x y_{d}}{x_{d}^{2} + y_{d}^{2}}\\- \frac{x_{d} y}{x_{d}^{2} + y_{d}^{2}} & \frac{x x_{d}}{x_{d}^{2} + y_{d}^{2}}\end{bmatrix}\\
\begin{bmatrix}\Omega& -\alpha I\end{bmatrix}&=\begin{bmatrix}\frac{y y_{d}}{x_{d}^{2} + y_{d}^{2}} & - \frac{x y_{d}}{x_{d}^{2} + y_{d}^{2}} & -\alpha & 0\\- \frac{x_{d} y}{x_{d}^{2} + y_{d}^{2}} & \frac{x x_{d}}{x_{d}^{2} + y_{d}^{2}} & 0 & -\alpha\end{bmatrix}
\end{aligned}
$$

$$
\text{ker}\left(\begin{bmatrix}\Omega& -\alpha I\end{bmatrix}\right) = 
\left\{\begin{bmatrix}x\\y\\0\\0\end{bmatrix},
\begin{bmatrix}\alpha \left(- x_{d}^{2} - y_{d}^{2}\right)\\0\\- y y_{d}\\x_{d} y\end{bmatrix}
\right\}
\\\implies \nabla\widehat{D} = \beta\begin{bmatrix}x\\y\\0\\0\end{bmatrix} + \gamma\begin{bmatrix}-\alpha \left(x_{d}^{2} +y_{d}^{2}\right)\\0\\- y y_{d}\\x_{d} y\end{bmatrix}
$$

<!-- #### Se $\widehat{D}=\|q - q_d\|^2$:
$$
\nabla\widehat{D} = \begin{bmatrix}x - x_{d}\\y - y_{d}\\- x + x_{d}\\- y + y_{d}\end{bmatrix}\notin \mathcal{N}
$$
#### Se $\widehat{D}=1-q_d^Tq$:
$$
\nabla\widehat{D} = \begin{bmatrix}- x_{d}\\- y_{d}\\- x\\- y\end{bmatrix}\notin \mathcal{N}
$$ -->