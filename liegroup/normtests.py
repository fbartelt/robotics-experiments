#%%
import sympy as sp
from IPython.display import display

psid, psi = sp.symbols('psi_d psi')
hyperb_rot_d = sp.Matrix([[sp.cosh(psid), sp.sinh(psid)], [sp.sinh(psid), sp.cosh(psid)]])
hyperb_rot = sp.Matrix([[sp.cosh(psi), sp.sinh(psi)], [sp.sinh(psi), sp.cosh(psi)]])
inv_hyperb_rot_d = sp.simplify(hyperb_rot_d.inv())
display(inv_hyperb_rot_d)
frob_arg = sp.simplify(sp.eye(2) + inv_hyperb_rot_d @ hyperb_rot)
metric = sp.simplify(1/2*sp.trace(frob_arg.T @ frob_arg))
display(metric)
first_derivative = sp.simplify(sp.diff(metric, psi))
display(first_derivative)
second_derivative = sp.simplify(sp.diff(first_derivative, psi))
display(second_derivative)
# %%
import sympy as sp
from IPython.display import display
x, xd, y, yd, alpha = sp.symbols('x x_d y y_d alpha', real=True)
theta, thetad, r, rd = sp.symbols('theta theta_d r r_d', real=True)
q = sp.Matrix([[x], [y]])
qd = sp.Matrix([[xd], [yd]])
W = sp.Matrix([[y], [-x]])
Wd = sp.Matrix([[yd], [-xd]])
# Distance || q - qd||^2
grad_distance = sp.Matrix([[x - xd], [y - yd], [xd - x], [yd - y]])
# Distance = 1 - qd^T q
grad_distance2 = sp.Matrix([[-xd], [-yd], [-x], [-y]])
display(Wd.pinv())
Omega = Wd.pinv().T @ W.T
display(Omega)
A = sp.Matrix(sp.BlockMatrix([[Omega, -alpha*sp.eye(2)]]))
display(A)
display(sp.simplify(A @ grad_distance2))
display(sp.solve(sp.Eq(A @ grad_distance2, 0), [x, y, xd, yd], set=True))
# eq = sp.simplify(A.nullspace()[1] * xd * y)
display(A.nullspace())
# sol = sp.solve(D.diff(sp.Matrix([x, y, xd, yd])), eq)
#%%
## Quaternion
x, y, z, w = sp.symbols('x y z w', real=True)
xd, yd, zd, wd = sp.symbols('xd yd zd wd', real=True)
W = sp.Matrix([[w, z, x], [-z, w, x], [y, -x, w], [-x, -y, -z]])
Wd = sp.Matrix([[wd, zd, xd], [-zd, wd, xd], [yd, -xd, wd], [-xd, -yd, -zd]])
Omega = sp.simplify(Wd.pinv().T @ W.T)
A = sp.simplify(sp.Matrix(sp.BlockMatrix([[Omega, -alpha*sp.eye(4)]])))
display(A.nullspace())

 
# %%
