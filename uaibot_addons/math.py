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

def skew(q):
    """Maps a vector to a skew-symmetric matrix"""
    q = np.array(q).ravel()
    return np.array([[0, -q[2], q[1]],
                     [q[2], 0, -q[0]],
                     [-q[1], q[0], 0]])

def vee(R):
    """Maps a skew-symmetric matrix to a vector"""
    return np.array([[R[2,1]], [R[0,2]], [R[1,0]]])