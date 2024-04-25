#%%
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from itertools import product
from scipy.optimize import minimize_scalar, brute
from scipy.spatial import KDTree
from scipy.linalg import logm, block_diag, expm
import sys
sys.path.append('/home/fbartelt/Documents/Projetos/robotics-experiments/')
from uaibot_addons.math import skew, vee

_INVHALFPI = 0.63660

class VectorField:
    """Vector Field class. Uses extension to the vector field presented in:    
    A. M. C. Rezende, V. M. Goncalves and L. C. A. Pimenta, "Constructive Time-
    Varying Vector Fields for Robot Navigation," in IEEE Transactions on 
    Robotics, vol. 38, no. 2, pp. 852-867, April 2022, 
    doi: 10.1109/TRO.2021.3093674.

    Parameters
    ----------
    parametric_equation : function
        A function that represents the parametric equation of the curve. It must
        return a tuple whose first component is a NxM array of coordinates of 
        the curve, and the second component is a Nx3x3 tensor corresponding to
        the N orientations frames. For the first component, each one of the n rows 
        should contain a m-dimensional float vector that is the n-th 
        m-dimensional sampled point of the curve.
    time_dependent : bool
        Whether the curve (consequently the vector field) is time dependent or 
        not.
    alpha : float, optional
        Controls the vector field behaviour. Greater alpha's imply more 
        robustness to the vector field, but increased velocity and acceleration
        behaviours. Used in G(u) = (2/pi)*atan(alpha*u). The default is 1.
    const_vel : float, optional
        The constant velocity of the vector field. The signal of this number 
        controls the direction of rotation. The default is 1.
    dt : float, optional
        The time step used to compute the time derivative of the vector field.
        The default is 1e-3.

    Methods
    -------
    __call__(position, time=0)
        Returns the vector field value at the given position and time. It is the
        same as calling the psi method.
    __repr__()
        Returns the string representation of the vector field.
    psi(position, time=0, store_points=True)
        Returns the vector field value at the given position and time. If
        store_points is True, the nearest points of the curve are stored in the
        nearest_points attribute.
    acceleration(position, velocity, time=0)
        Returns the acceleration of the vector field at the given position,
        velocity and time.
    """
    def __init__(
        self, parametric_equation, time_dependent, kf=1, vr=1, wr=1, beta=1, dt=1e-3
    ):
        self.parametric_equation = parametric_equation
        self.kf = kf
        self.vr = vr
        self.wr = wr
        self.beta = beta
        self.time_dependent = time_dependent
        self.dt = dt
        self.nearest_points = []

    def __call__(self, position, orientation, time=0, store_points=True):
        return self.psi(position, orientation, time, store_points)

    def __repr__(self):
        return f"Time-{('In'*(not self.time_dependent)+'variant').capitalize()} Vector Field.\n Alpha: {self.alpha},\n Constant Velocity: {self.const_vel},\n dt: {self.dt},\n Parametric Equation: {self.parametric_equation.__name__}"
    
    @staticmethod
    def _distance_fun(p, R, pd, Rd, beta=1):
        """ Computes the distance between the current point and the desired point
        and the current orientation and the desired orientation. The distance is
        described by $D=\|p-p_d\|^2 + \frac{1}{2}\|I - Rd^TR\|^2_F$
        """
        p = np.array(p).reshape(-1, 1)
        pd = np.array(pd).reshape(-1, 1)
        lin_dist = np.linalg.norm(p - pd)**2
        rot_dist = 0.5 * np.linalg.norm(Rd.T @ R, 'fro')**2 
        return lin_dist + (beta)**2 * rot_dist # change term to beta**2
    
    def _divide_conquer(self, curve_segment, p, R):
        curve_points = np.array(curve_segment[0])
        curve_frames = curve_segment[1]
        npoints = curve_segment[0].shape[0]

        if npoints == 1:
            return self._distance_fun(p, R, curve_points[0, :], curve_frames[0, :, :], self.beta), 0
        
        mid_index = npoints // 2
        left_segment = (curve_segment[0][:mid_index, :], curve_segment[1][:mid_index, :, :])
        right_segment = (curve_segment[0][mid_index:, :], curve_segment[1][mid_index:, :, :])

        left_dist, left_index = self._divide_conquer(left_segment, p, R)
        right_dist, right_index = self._divide_conquer(right_segment, p, R)

        if left_dist < right_dist:
            return left_dist, left_index
        else:
            return right_dist, right_index + mid_index

    def _add_nearest_point(self, point):
        self.nearest_points.append(point)

    def psi(self, position, orientation, time=0, store_points=True):
        """Computes the normalized vector field value at the given position and
        time. It is the same as calling the __call__ method. If store_points is
        True, the nearest points of the curve are stored in the nearest_points
        attribute.

        Parameters
        ----------
        position : list or np.array
            The position where the vector field will be computed.
        time : float, optional
            The time at which the vector field will be computed. The default is 0.
        store_points : bool, optional
            Whether to store the nearest points of the curve. The default is True.
        """
        psi_s = self._psi_s(position, orientation, time, store_points=store_points)
        # if self.time_dependent:
        #     psi_s = psi_s / abs(self.const_vel)
        #     psi_t = self._psi_t(position, time, store_points=False)
        #     eta = (
        #         -psi_s.T @ psi_t
        #         + np.sqrt((psi_s.T @ psi_t) ** 2 + self.const_vel**2 - psi_t.T @ psi_t)
        #     )[0, 0]
        #     return eta * psi_s + psi_t
        # else:
        #     return psi_s
        return psi_s

    def _psi_s(self, position, orientation, time=0, store_points=True):
        p = np.array(position).reshape(-1, 1)
        curve = self.parametric_equation(time=time)
        return self._vector_field_vel(
            p,
            orientation,
            curve,
            self.kf,
            self.vr,
            self.wr,
            store_points=store_points,
        )

    def _vector_field_vel(
        self, p, R, curve, kf, vr, wr, store_points=True
    ):
        vec_n, vec_t, min_dist = self._compute_ntd(curve, p, R, store_points=store_points)
        fun_g = _INVHALFPI * np.arctan(kf * min_dist)
        fun_h = np.sqrt(max(1 - fun_g**2, 0))
        sgn = 1
        Lambda = block_diag(np.eye(3) * vr, np.eye(3) * wr)

        return Lambda @ (-fun_g * vec_n + sgn * fun_h * vec_t)

    # def _compute_ntd_nonoptimized(self, curve, p, store_points=True):
    def _compute_ntd(self, curve, p, R, store_points=True):
        #TODO: testing -> normalize velocities separetely
        min_dist = float("inf")
        ind_min = -1

        pr = np.array(p).ravel() # row vector to simplify computations
        min_dist, ind_min = self._divide_conquer(curve, pr, R)

        vec_n_p = pr - curve[0][ind_min, :]
        Rd = curve[1][ind_min, :, :]
        sigma = skew(Rd[:, 0]) @ R[:, 0].reshape(-1, 1) + skew(Rd[:, 1]) @ R[:, 1].reshape(-1, 1) + skew(Rd[:, 2]) @ R[:, 2].reshape(-1, 1)
        # vec_n = np.vstack((vec_n_p.reshape(-1, 1), sigma))
        # vec_n = (vec_n / (np.linalg.norm(vec_n, 2) + 0.0001)).reshape(-1, 1)
        vec_n_p = (vec_n_p / (np.linalg.norm(vec_n_p, 2) + 0.0001)).reshape(-1, 1)
        sigma = (sigma / (np.linalg.norm(sigma, 2) + 0.0001)).reshape(-1, 1)
        vec_n = np.vstack((vec_n_p.reshape(-1, 1), sigma))

        if ind_min == np.shape(curve[0])[0] - 1:
            vec_t_p = curve[0][1, :] - curve[0][ind_min, :]
        else:
            vec_t_p = curve[0][ind_min + 1, :] - curve[0][ind_min, :]

        Rd = curve[1][ind_min, :, :]
        vec_t_rot = vee(logm(Rd @ R.T) / self.dt)
        # vec_t = np.vstack((vec_t_p.reshape(-1, 1), vec_t_rot))
        # vec_t = (vec_t / (np.linalg.norm(vec_t, 2) + 0.0001)).reshape(-1, 1)
        vec_t_p = (vec_t_p / (np.linalg.norm(vec_t_p, 2) + 0.0001)).reshape(-1, 1)
        vec_t_rot = (vec_t_rot / (np.linalg.norm(vec_t_rot, 2) + 0.0001)).reshape(-1, 1)
        vec_t = np.vstack((vec_t_p.reshape(-1, 1), vec_t_rot))
        # print(Rd.shape, vec_n_p.shape)
        # print(vec_n.shape)
        # print(np.shape(curve[0])[0])
        # print(ind_min)
        # print(vec_t.shape)
        if store_points:
            self._add_nearest_point((curve[0][ind_min, :], curve[1][ind_min, :, :]))

        if vec_n.shape != (6, 1) or vec_t.shape != (6, 1):
            print(f"Error in vec_n or vec_t: {vec_n.shape}, {vec_t.shape}")
        return vec_n, vec_t, min_dist
    
    def acceleration(self, position, orientation, v, w, time=0):
        """ Returns the acceleration of the vector field at the given position,
        orientation, linear velocity, angular velocity and time.

        It computes the nummerical approximation for the vector field time derivative
        \dot{VF} = \partial{VF}/\partial{t} + \partial{VF}/\partial{x}\dot{x}.

        Parameters
        ----------
        position : list or np.array
            The position where the acceleration will be computed.
        orientation : np.array
            The orientation at the given position, represented by a 3x3 rotation matrix.
        v : list or np.array
            The linear velocity at the given position.
        w : list or np.array
            The angular velocity at the given position.
        time : float, optional
            The time at which the acceleration will be computed. The default is 0.
        """
        position = np.array(position).reshape(-1, 1)
        l_velocity = np.array(v).reshape(-1, 1)
        a_velocity = np.array(w).reshape(-1, 1)
        current_vf = self.psi(position, orientation, time, store_points=False)
       
        # \partial{\phi_p}/\partial{p} \dot{p}
        dphipdp = (
            np.array(
                [
                    self.psi(
                        position + np.array([self.dt, 0, 0]).reshape(-1, 1),
                        orientation,
                        time,
                        store_points=False,
                    )[:3, :]
                    - current_vf[:3, :],
                    self.psi(
                        position + np.array([0, self.dt, 0]).reshape(-1, 1),
                        orientation,
                        time,
                        store_points=False,
                    )[:3, :]
                    - current_vf[:3, :],
                    self.psi(
                        position + np.array([0, 0, self.dt]).reshape(-1, 1),
                        orientation,
                        time,
                        store_points=False,
                    )[:3, :]
                    - current_vf[:3, :],
                ]
            )
            .reshape(3, 3)
            .T
            / self.dt
        )
        l_acceleration = dphipdp @ l_velocity
        # \partial{\phi_p}/\partial{R} \dot{R}
        dphiRdR = (np.array(
            [
                self.psi(
                    position,
                    expm(self.dt * skew(np.array([0, 0, self.dt]).reshape(-1, 1))) @ orientation,
                    time,
                    store_points=False,
                )[3:, :]
                - current_vf[3:, :],
                self.psi(
                    position,
                    expm(self.dt * skew(np.array([self.dt, 0, 0]).reshape(-1, 1))) @ orientation,
                    time,
                    store_points=False,
                )[3:, :]
                - current_vf[3:, :],
                self.psi(
                    position,
                    expm(self.dt * skew(np.array([0, self.dt, 0]).reshape(-1, 1))) @ orientation,
                    time,
                    store_points=False,
                )[3:, :]
                - current_vf[3:, :],
            ]
        )
        .reshape(3, 3)
        .T
        / self.dt)
        a_acceleration = dphiRdR @ a_velocity
        a = np.vstack((l_acceleration, a_acceleration))

        return a
# %%
from scipy.spatial.transform import Rotation
import numpy as np
# Parametric equation definition
maxtheta = 500

def parametric_eq_factory(w1, w2, c1, c2, c3, h0, maxtheta, T, dt, timedependent=True):
    theta = np.linspace(0, 2 * np.pi, num=maxtheta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    precomputed = ()
    cw1t = np.cos(0)
    sw1t = np.sin(0)
    cw2t = np.cos(0)
    rotz = np.matrix([[cw1t, -sw1t, 0], [sw1t, cw1t, 0], [0, 0, 1]])

    curve = np.empty((3, len(theta)))
    for i, _ in enumerate(theta):
        curve[:, i] = rotz @ np.array([
            c1 * cos_theta[i],
            c2 * sin_theta[i],
            h0 + c3 * cw2t * cos_theta[i] ** 2
        ])
    orientations = np.empty((len(theta), 3, 3))
    for i, ang in enumerate(theta):
        orientations[i, :, :] = Rotation.from_euler('x', ang).as_matrix() @ rotz
    
    precomputed = (curve.T, orientations)
    
    def parametric_eq(time):
        return precomputed


    return parametric_eq

eq = parametric_eq_factory(w1=0, w2=0, c1=0.5, c2=0.5, c3=0.1, h0=0.3, maxtheta=maxtheta, T=2, dt=1e-2, timedependent=False)
vf = VectorField(eq, False, kf=1, vr=1, wr=1, dt=1e-3)
# %%
curve = vf.parametric_equation(time=0)
vf._compute_ntd(curve, np.array([0,0,0]), Rotation.from_euler('x', 0).as_matrix(), False)
vf.psi(np.array([0,0,0]), Rotation.from_euler('x', 0).as_matrix(), 0)
vf.acceleration(np.array([0,0,0]), Rotation.from_euler('x', 0).as_matrix(), np.array([1,0,0]), np.array([0,1,0]), 0)
# %%
