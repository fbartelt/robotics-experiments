$$
\xi = \begin{bmatrix}x & R\end{bmatrix},\ \dot{\xi} = \begin{bmatrix}\dot{x} & \omega\end{bmatrix},\ \ddot{\xi} = \begin{bmatrix}\ddot{x} & \dot{\omega}\end{bmatrix}\\
\tilde{x} = x - x_d,\ \tilde{\omega} = \omega - \omega_d,\ \tilde{R} = R_d^TR\\
\dot{R}_d = S(\omega_d)R_d,\ R^TR=I\\
S(Ra) = RS(a)R^T\\
$$

$$
\begin{align}
\begin{split}
\dot{\tilde{R}} &= \dot{R}_d^TR + R_d^T\dot{R} = R_d^T\left(S(\omega_d)^T + S(\omega) \right)R = 
R_d^T\left(-S(\omega_d) + S(\omega) \right)R = R_d^TS(\omega- \omega_d)R\\
&=R_d^TR_dR_d^TS(\omega- \omega_d)R_dR_d^TR = S(R_d^T(\omega- \omega_d))R_d^TR 
= S(R_d^T\tilde{\omega})\tilde{R}\\
\end{split}
\end{align}
$$


$$
s = \begin{bmatrix}
s_l \\ \sigma
\end{bmatrix} = 
\begin{bmatrix}
\dot{\tilde{x}} + \lambda \tilde{x} \\ \tilde{\omega} + \lambda R_d S^{-1}\left(\mathbb{P}_a(\tilde{R})\right)
\end{bmatrix}\\
s = \dot{\xi} -\dot{\xi}_r, \text{ se }  \dot{\xi}_r = \begin{bmatrix}
\dot{x}_d -\lambda \tilde{x} \\ \omega_d - \lambda R_d S^{-1}\left(\mathbb{P}_a(\tilde{R})\right)
\end{bmatrix}
$$
Em $s=0$:
$$
\begin{align*}
s_l &= 0 = \dot{\tilde{x}} + \lambda \tilde{x} \implies\dot{\tilde{x}} =- \lambda \tilde{x}\\
\sigma &=0 = \tilde{\omega} + \lambda R_d S^{-1}\left(\mathbb{P}_a(\tilde{R})\right)\\
&\implies \tilde{\omega} = -\lambda R_d S^{-1}\left(\mathbb{P}_a(\tilde{R})\right)\\
&\text{substituindo em (1)}\\
&\dot{\tilde{R}} = S\left(R_d^T\left(-\lambda R_d S^{-1}\left(\mathbb{P}_a(\tilde{R})\right)\right)\right)\tilde{R} = -\lambda S\left(S^{-1}\left(\mathbb{P}_a(\tilde{R})\right)\right)\tilde{R} 
= -\lambda \mathbb{P}_a(\tilde{R})\tilde{R}
\end{align*}
$$

$$
\begin{align*}
V(t) &= \frac{1}{2}\left( s^TMs + \sum_{i=1}^N \tilde{o}_i^T\Gamma_o^{-1}\tilde{o}_i + 
\tilde{r}_i^T\Gamma_r^{-1}\tilde{r}_i \right)\\
\dot{V} &=  s^T M \dot{s} + \frac{1}{2}s^T\dot{M}s + \sum_{i=1}^N \tilde{o}_i^T\Gamma_o^{-1}\dot{\tilde{o}}_i + \tilde{r}_i^T\Gamma_r^{-1}\dot{\tilde{r}}_i
\end{align*}
$$

$$
\begin{align*}
M\dot{s} &= M\ddot{\xi} - M\ddot{\xi}_r = \sum_{i=1}^N \mathcal{M}_i(\xi)\tau_i - C(\xi, \dot{\xi})\dot{\xi} - g(\xi) - M\ddot{\xi}_r= \sum_{i=1}^N \mathcal{M}_i(\xi)\tau_i - C(\xi, \dot{\xi})(s+\dot{\xi}_r) - g(\xi) - M\ddot{\xi}_r \\
&\alpha_i\left(M(\xi)\ddot{\xi}_r + C(\xi,\dot{\xi})\dot{\xi}_r+g\right) = Y_o(\xi, \dot{\xi}, \dot{\xi}_r, \ddot{\xi}_r)o_i, \quad \sum \alpha_i = 1\\
&\sum_{i=1}^N Y_oo_i = M(\xi)\ddot{\xi}_r + C(\xi,\dot{\xi})\dot{\xi}_r+g\\
&\implies M\dot{s} = \sum_{i=1}^N (\mathcal{M}_i(\xi)\tau_i - Y_oo_i)- C(\xi, \dot{\xi})s
\end{align*} 
$$

$$
\begin{align*}
\dot{V} &=  \sum_{i=1}^N s^T(\mathcal{M}_i(\xi)\tau_i - Y_oo_i) + \frac{1}{2}s^T\left(\dot{M} - 2C\right)s
+ \tilde{o}_i^T\Gamma_o^{-1}\dot{\tilde{o}}_i + \tilde{r}_i^T\Gamma_r^{-1}\dot{\tilde{r}}_i\\
\tau_i &= \hat{\mathcal{M}}_i^{-1}F_i\\
F_i &= Y_o\hat{o}_i - K_Ds\\
\end{align*}
$$