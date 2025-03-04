$$
M(q)\ddot{q} + C\dot{q} + g = \tau\\
\tau = J^Tf\\
v = J\dot{q}\\
\implies \dot{v} = \dot{J}\dot{q} + J\ddot{q}
$$

$$
M(q)J^{-1}(\dot{v} - \dot{J}\dot{q}) + CJ^{-1}v + g = J^Tf\\
M(q)J^{-1}(\dot{v} - \dot{J}J^{-1}v) + CJ^{-1}v + g = J^Tf\\
MJ^{-1}\dot{v} + (CJ^{-1}- MJ^{-1}\dot{J}J^{-1})v + g = J^Tf\\
\implies J^{-T}MJ^{-1}\dot{v} + J^{-T}(CJ^{-1}- MJ^{-1}\dot{J}J^{-1})v + J^{-T}g = f\\
\therefore \Lambda\dot{v} + \Gamma v + \eta = f,\\
\Lambda = J^{-T}MJ^{-1},\\ 
\Gamma=J^{-T}(CJ^{-1}- MJ^{-1}\dot{J}J^{-1}) = J^{-T}CJ^{-1} - \Lambda\dot{J}J^{-1},
\\ \eta=J^{-T}g\\
$$

$$
\ddot{q} = M^{-1}(-C\dot{q}-g + \tau)\\
\tau = \hat{C}\dot{q}-\hat{g} + \hat{M}a\\
a = a_d + K_d(\dot{q} - \dot{q}_d) + K_p(q - q_d)\\
a = a_d + K_d(\dot{q} - J^{-1}v_d) + K_p(q - q_d)\\
\implies \ddot{q} = M^{-1}(-\tilde{C}\dot{q}-\tilde{g} + \hat{M}a)
$$



---
### $\dot{J}$

$$
\dot{x} = J\dot{q} \implies \ddot{x} = \dot{J}\dot{q} + J\ddot{q}\\
\ddot{q} = J^{-1}(\ddot{x} - \dot{J}\dot{q})
$$

$$
J(q)\in\R^{6\times n}, q(t)\in\R^{n}\\
\dot{J} = \frac{dJ}{dt} = \frac{\partial J}{\partial q}\dot{q}
$$
---
### Cruz-Ancona
$$
x=q-q_d\\
\tilde{M} = M + \Delta = (I + \Delta M^{-1})M \\
\tilde{M}\ddot{q} + \bar{C} + g = \tau + \delta;\ \tau=\bar{C} + g + Ma;\ a=\ddot{q}_d + u\\
\tilde{M}\ddot{q} + \bar{C} + g = \tau + \delta\\
\tilde{M}\ddot{q} + \bar{C} + g -\tilde{M}\ddot{q}_d = \tau + \delta -\tilde{M}\ddot{q}_d\\
\dot{x}_2 = \tilde{M}^{-1}(-\bar{C} - g + \delta - \tilde{M}\ddot{q}_d + \tau) = \tilde{M}^{-1}(\bar{h} + \tau)\\
\dot{x}_2 = (\tilde{M}^{-1} + M^{-1} - M^{-1})(\bar{h} + \tau) = M^{-1}(I + \Delta_G)(\bar{h} + \tau),\\
\Delta_G = M\tilde{M}^{-1} - I\\
\implies \dot{x} = Ax + B(G(I + \Delta_G)\tau + h),\\
A = \begin{bmatrix}\mathbf{0} & I_{n\times n}\\\mathbf{0} & \mathbf{0}\end{bmatrix},\ B= \begin{bmatrix}\mathbf{0} \\ I_{n\times n}\end{bmatrix},\ G=M^{-1},\ h=G(I+\Delta_G)\bar{h}
$$

$$
\tau = \psi + v\\
\psi = -\varrho B^TPx,\\
PA + A^TP -\varrho PBB^TP + 2I=0
$$

#### Modification: change $\psi$
$$
\psi = M(\ddot{q}_d + K_d(\dot{q}_d - \dot{q} )) + C\dot{q} + g\\
V_0 = x^TPx,\ \text{t.q.  }\ A^TP + PA = -Q\\
w^T = \frac{\partial V_0}{\partial x}B= x^TPB\\
$$
$$
\begin{bmatrix}\dot{q}-\dot{q}_d\\\ddot{q}-\ddot{q}_d\end{bmatrix} = \begin{bmatrix}\mathbf{0} & I_{n\times n}\\\mathbf{0} & \mathbf{0}\end{bmatrix}\begin{bmatrix}q-{q}_d\\\dot{q}-\dot{q}_d\end{bmatrix}
+ \begin{bmatrix}\mathbf{0} \\ G(I+\Delta_G)\end{bmatrix}\tau + \begin{bmatrix}\mathbf{0}\\h\end{bmatrix}\\
G(I+\Delta_G)\tau = (M^{-1}+\tilde{M}^{-1}-M^{-1})\tau = \tilde{M}^{-1}M(\ddot{q}_d - K_d(\dot{q}-\dot{q}_d)) + \tilde{M}^{-1}(C\dot{q} + g) + \tilde{M}^{-1}v \\
= \ddot{q}_d - K_d(\dot{q}-\dot{q}_d) - \tilde{M}^{-1}\Delta(\ddot{q}_d - K_d(\dot{q}-\dot{q}_d)) + \tilde{M}^{-1}(C\dot{q} + g)+ \tilde{M}^{-1}v\\
h = \tilde{M}^{-1}(-C\dot{q}-g + \delta -\tilde{M}\ddot{q}_d)=\tilde{M}^{-1}(-C\dot{q}-g + \delta) - \ddot{q}_d\\
\implies\ddot{q}-\ddot{q}_d= -K_d(\dot{q}-\dot{q}_d) - \tilde{M}^{-1}\Delta(\ddot{q}_d - K_d(\dot{q}-\dot{q}_d)) + \tilde{M}^{-1}\delta + \tilde{M}^{-1}v
$$


$$
\bar{B}= \frac{\partial w}{\partial x}B = B^TPB\ \implies \bar{w} = \bar{B}w=B^TPBB^TPx\\
\dot{V} = x^TP\dot{x} + \dot{x}^TPx = 2x^TP\dot{x} = 2x^TPAx + 2x^TPBG\tau + 2x^TPBG\Delta_G\tau 
+ 2x^TPBh\\
\bullet 2x^TPBG\tau = 2x^TPBM^{-1}(M(\ddot{q}_d + K_d(\dot{q}_d - \dot{q} )) + C\dot{q} + g)\\
= - 2x^TPB(K_d(\dot{q} - \dot{q}_d)) + 2x^TPB\ddot{q}_d + 2x^TPBM(C\dot{q} + g)\\
\bullet 2x^TPBG\Delta_G\tau=
$$

#### Mod
$$
M\ddot{q} + C\dot{q} + g = \tau + \delta\\
\tau = \hat{M}a + C\dot{q} + g\\
a = \ddot{q}_d - K_p(q - q_d) - K_d(\dot{q} - \dot{q}_d) + v\\
\tilde{M} = \hat{M} - M
$$

$$
M\ddot{q} + C\dot{q} + g = \hat{M}\left(\ddot{q}_d - K_p(q - q_d) - K_d(\dot{q} - \dot{q}_d) + v\right) + C\dot{q} + g + \delta\\
\ddot{q} = M^{-1}\hat{M}\left(\ddot{q}_d - K_p(q - q_d) - K_d(\dot{q} - \dot{q}_d) + v\right) + M^{-1}\delta\\
\ddot{q} = \ddot{q}_d - K_p(q - q_d) - K_d(\dot{q} - \dot{q}_d) + v + M^{-1}(\tilde{M}a + \delta)\\
\text{Seja } x = \begin{bmatrix}q - q_d & \dot{q} - \dot{q}_d\end{bmatrix}^T:\\
\dot{x} = \underbrace{\begin{bmatrix}\mathbf{0} & I_{n\times n} \\ -K_p & -K_d\end{bmatrix}}_{\bar{A}}x + \underbrace{\begin{bmatrix}\mathbf{0}\\I_{n\times n }\end{bmatrix}}_{B}(v + \eta),\\
\eta = M^{-1}(\tilde{M}a + \delta)\\
v = \begin{cases}-\bar{w} - \Gamma\frac{\bar{w}}{\|\bar{w}\|} - \rho\frac{\bar{w}}{\|\bar{w}\|^2}, & \|w\| > \frac{\epsilon}{2}\\
-\mathcal{B}(w)\frac{\bar{w}}{\|\bar{w}\|}, & \|w\| \ge \frac{\epsilon}{2}

\end{cases}\\

\Gamma = \kappa^T b\\ \kappa=\begin{bmatrix}1 & \|x\| & \|x\|^2\end{bmatrix}^T\\
\mathcal{B}(z) = \frac{\|z\|}{\epsilon - \|z\|}\\

\dot{b} = L (\kappa \|\bar{w}\| - \Xi b)\\
\dot{\rho} = l - \rho
$$
---
$$
\text{Seja } u = -Kx \text{ tal que } a = \ddot{q}_d +u + v\ \text{então}\\
\bar{A} = A - BK = A-B \begin{bmatrix}K_p&K_d\end{bmatrix}
$$

$$
\text{Let } V_0 = x^TPx, \text{where } P \text{ is the solution of the algebraic Ricatti equation}\\
PA + A^TP - PBR^{-1}B^TP = -Q,\ R=R^T>0, Q=Q^T\ge0\\
\dot{V}_0 = x^TP\dot{x} + \dot{x}^TPx=x^TP(Ax + B(v+u+\eta)) + x^TA^TPx + (v+u+\eta)^TB^TPx\\
= x^T(PA + A^TP)x + 2x^TPB(v+u+\eta)\\
= x^T(PA + A^TP)x + 2x^TPB(v+\eta) - 2x^TPBKx\\
\text{ Let } K = R^{-1}B^TP\\
\dot{V}_0= x^T(PA + A^TP - 2PBR^{-1}B^TP)x + 2x^TPB(v+\eta) = -x^T(Q + PBR^{-1}B^TP)x + 2x^TPB(v+\eta)\\
\text{If the system is observable then } PBR^{-1}B^TP > 0. \text{ Considering no uncertainties and disturbances:}\\
\dot{V}_0 \le -x^T(Q + PBR^{-1}B^TP)x < 0. 
\\
\text{Then } \lambda_{\text{min}}(P)\|x\|^2 \le V_0 \le \lambda_{\text{max}}(P)\|x\|^2, \dot{V}_0 \le \lambda_{\text{min}}(Q + PBR^{-1}B^TP)\|x\|^2.\\
\lambda_{\text{min}}(P)\|x\|^2=: \alpha_1(x) \in \mathcal{K}_\infty\\
\lambda_{\text{max}}(P)\|x\|^2=: \alpha_2(x) \in \mathcal{K}_\infty\\
\lambda_{\text{min}}(Q + PBR^{-1}B^TP)\|x\|^2=: \alpha_3(x) \in \mathcal{K}

$$
---
### Frank Lewis Adaptive Robust Control
$$
e = q_d - q \implies \dot{e} = \dot{q}_d - \dot{q} \implies \ddot{e} = \ddot{q}_d - \ddot{q}\\
\vec{e} = \begin{bmatrix}e \\ \dot{e}\end{bmatrix}\\
r = e + \dot{e}\\
\rho = \delta_0 + \delta_1\|\vec{e}\| + \delta_2\|\vec{e}\|^2 = S\cdot\theta\\
S = \begin{bmatrix}1&\|\vec{e}\|&\|\vec{e}\|^2\end{bmatrix}\\
\hat{\rho} =  S\cdot\hat{\theta}\\
\hat{\theta} = \begin{bmatrix}\hat{\delta}_0 & \hat{\delta}_1 & \hat{\delta}_2\end{bmatrix}^T\\
v_R = \frac{r\hat{\rho}^2}{\hat{\rho}\|r\|+\varepsilon}\\
$$

$$
\dot{\varepsilon} = - k_\varepsilon \varepsilon,\ \varepsilon(0)>0\\
\dot{\hat{\theta}} = \gamma S^T \|r\|\\
\dot{\tilde{\theta}} = -\gamma S^T \|r\|,\ \tilde{\theta} = \theta - \hat{\theta}
$$


$$
M\ddot{q} + C\dot{q} + g + T_d = \tau = K_vr + v_R\\
M(\ddot{q}_d - \ddot{e}) + C(\dot{q}_d - \dot{e}) + g + T_d - K_vr - v_R = 0\\
= M(\ddot{q}_d - \ddot{e}) + M\dot{e} - M\dot{e} + C(\dot{q}_d - \dot{e}) +Ce-Ce + g + T_d - K_vr - v_R\\
= - M(\ddot{e} + \dot{e}) + M(\ddot{q}_d + \dot{e}) -C(\dot{e} + e) + C(\dot{q}_d + {e}) + g + T_d - K_vr - v_R\\
= - M(\dot{r}) + M(\ddot{q}_d + \dot{e}) -C(r) + C(\dot{q}_d + e) + g + T_d - K_vr - v_R\\
\implies M\dot{r} = -Cr -K_vr -v_R + w,\\
w = M(\ddot{q}_d + \dot{e})  + C(\dot{q}_d + e) + g + T_d
$$