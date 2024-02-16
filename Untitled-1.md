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




$$
\dot{x} = J\dot{q} \implies \ddot{x} = \dot{J}\dot{q} + J\ddot{q}\\
\ddot{q} = J^{-1}(\ddot{x} - \dot{J}\dot{q})
$$

$$
J(q)\in\R^{6\times n}, q(t)\in\R^{n}\\
\dot{J} = \frac{dJ}{dt} = \frac{\partial J}{\partial q}\dot{q}
$$