import numpy as np

def dot_J(robot, qdot, q=None):
    """Compute the end effetctor jacobian matrix time derivative.
    robot.jac_jac_geo returns a list of the jacobians of the i-th jacobian column
    by the configurations q. The jacobian of the jacobian (jac_jac_geo) is a 
    nx6xn tensor. Since \dot{J} = \frac{\partial J}{\partial q}\dot{q}, we can 
    compute the jacobian time derivative as jac_jac_geo @ qdot.
    """
    if q is None:
        q = robot.q
    jj_geo, *_ = robot.jac_jac_geo(q=q, axis='eef')
    dotJ = np.array(jj_geo) @ np.array(qdot).reshape(-1, 1)
    dotJ = dotJ[:, :, 0].T
    
    return dotJ