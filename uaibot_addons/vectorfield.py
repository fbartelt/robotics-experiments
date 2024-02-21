import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from itertools import product

_INVHALFPI = 0.63660


class VectorField:
    """Vector Field class. Uses the vector field presented in:    
    A. M. C. Rezende, V. M. Goncalves and L. C. A. Pimenta, "Constructive Time-
    Varying Vector Fields for Robot Navigation," in IEEE Transactions on 
    Robotics, vol. 38, no. 2, pp. 852-867, April 2022, 
    doi: 10.1109/TRO.2021.3093674.

    Parameters
    ----------
    parametric_equation : function
        A function that represents the parametric equation of the curve. It must
        return a NxM array of coordinates of the curve. Each one of the n rows 
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
        self, parametric_equation, time_dependent, alpha=1, const_vel=1, dt=1e-3
    ):
        self.parametric_equation = parametric_equation
        self.alpha = alpha
        self.const_vel = const_vel
        self.time_dependent = time_dependent
        self.dt = dt
        self.nearest_points = []

    def __call__(self, position, time=0):
        return self.psi(position, time)

    def __repr__(self):
        return f"Time-{('In'*(not self.time_dependent)+'variant').capitalize()} Vector Field.\n Alpha: {self.alpha},\n Constant Velocity: {self.const_vel},\n dt: {self.dt},\n Parametric Equation: {self.parametric_equation.__name__}"

    def _add_nearest_point(self, point):
        self.nearest_points.append(point)

    def psi(self, position, time=0, store_points=True):
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
        psi_s = self._psi_s(position, time, store_points=store_points)
        if self.time_dependent:
            psi_s = psi_s / abs(self.const_vel)
            psi_t = self._psi_t(position, time, store_points=False)
            eta = (
                -psi_s.T @ psi_t
                + np.sqrt((psi_s.T @ psi_t) ** 2 + self.const_vel**2 - psi_t.T @ psi_t)
            )[0, 0]
            return eta * psi_s + psi_t
        else:
            return psi_s

    def _psi_s(self, position, time=0, store_points=True):
        # TODO change _add_nearest_point to apply here instead of psi t.
        # this implies copying the uaibot vector field code into here.
        p = np.array(position).reshape(-1, 1)
        curve = np.matrix(self.parametric_equation(time=time))
        # return ub.Robot.vector_field(curve, self.alpha, self.const_vel)(p)
        return self._vector_field_vel(
            p,
            curve,
            self.alpha,
            self.const_vel,
            np.shape(curve)[0],
            store_points=store_points,
        )

    def _psi_t(self, position, time, store_points=True):
        # TODO implement time derivative computation of Distance vector. The component PsiT,
        # which is the null space projection of the time derivative.
        p = np.array(position).reshape(-1, 1)
        curve = np.matrix(self.parametric_equation(time=time))
        next_curve = np.matrix(self.parametric_equation(time=time + self.dt))
        min_dist = float("inf")
        min_dist_next = float("inf")
        ind_min = -1
        ind_min_next = -1

        pr = np.matrix(p).reshape((3, 1))

        for i in range(np.shape(curve)[1]):
            dist_temp = np.linalg.norm(pr - curve[:, i])
            if dist_temp < min_dist:
                min_dist = dist_temp
                ind_min = i

            dist_temp = np.linalg.norm(pr - next_curve[:, i])
            if dist_temp < min_dist_next:
                min_dist_next = dist_temp
                ind_min_next = i

        if ind_min == np.shape(curve)[1] - 1:
            vec_t = (next_curve[:, 1] - curve[:, ind_min]) / self.dt
        else:
            vec_t = (next_curve[:, ind_min_next] - curve[:, ind_min]) / self.dt

        if store_points:
            self._add_nearest_point(curve[:, ind_min])
        vec_t = vec_t
        Tstar = self._compute_ntd(curve, p, False)[1]
        nullspace = np.eye(p.shape[0]) - Tstar @ Tstar.T
        vec_t = -nullspace @ vec_t

        return vec_t

    def acceleration(self, position, velocity, time=0):
        """ Returns the acceleration of the vector field at the given position,
        velocity and time.

        It computes the nummerical approximation for the vector field time derivative
        \dot{VF} = \partial{VF}/\partial{t} + \partial{VF}/\partial{x}\dot{x}.

        Parameters
        ----------
        position : list or np.array
            The position where the acceleration will be computed.
        velocity : list or np.array
            The velocity at the given position.
        time : float, optional
            The time at which the acceleration will be computed. The default is 0.
        """
        position = np.array(position).reshape(-1, 1)
        velocity = np.array(velocity).reshape(-1, 1)
        current_vf = self.psi(position, time, store_points=False)
        # \partial{vf}/\partial{t}
        dvfdt = (
            self.psi(position, time + self.dt, store_points=False) - current_vf
        ) / self.dt
        # \partial{vf}/\partial{x} \dot{x}
        dvfdx = (
            np.array(
                [
                    self.psi(
                        position + np.array([self.dt, 0, 0]).reshape(-1, 1),
                        time,
                        store_points=False,
                    )
                    - current_vf,
                    self.psi(
                        position + np.array([0, self.dt, 0]).reshape(-1, 1),
                        time,
                        store_points=False,
                    )
                    - current_vf,
                    self.psi(
                        position + np.array([0, 0, self.dt]).reshape(-1, 1),
                        time,
                        store_points=False,
                    )
                    - current_vf,
                ]
            )
            .reshape(3, 3)
            .T
            / self.dt
        )
        a = dvfdx @ velocity + dvfdt
        return a

    def _vector_field_vel(
        self, p, curve, alpha, const_vel, vector_size, store_points=True
    ):
        vec_n, vec_t, min_dist = self._compute_ntd(curve, p, store_points=store_points)
        fun_g = _INVHALFPI * np.arctan(alpha * min_dist)
        fun_h = np.sqrt(max(1 - fun_g**2, 0))
        abs_const_vel = abs(const_vel)
        sgn = const_vel / (abs_const_vel + 0.00001)

        return abs_const_vel * (fun_g * vec_n + sgn * fun_h * vec_t)

    def _compute_ntd(self, curve, p, store_points=True):
        min_dist = float("inf")
        ind_min = -1

        pr = np.matrix(p).reshape((3, 1))

        for i in range(np.shape(curve)[1]):
            dist_temp = np.linalg.norm(pr - curve[:, i])
            if dist_temp < min_dist:
                min_dist = dist_temp
                ind_min = i

        vec_n = curve[:, ind_min] - pr
        vec_n = vec_n / (np.linalg.norm(vec_n) + 0.0001)

        if ind_min == np.shape(curve)[1] - 1:
            vec_t = curve[:, 1] - curve[:, ind_min]
        else:
            vec_t = curve[:, ind_min + 1] - curve[:, ind_min]

        vec_t = vec_t / (np.linalg.norm(vec_t) + 0.0001)
        if store_points:
            self._add_nearest_point(curve[:, ind_min])

        return vec_n, vec_t, min_dist


def vector_field_plot(coordinates, field_values, add_lineplot=False, **kwargs):
    """Plot a vector field in 3D. The vectors are represented as cones and the
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function.

    Parameters
    ----------
    coordinates : list or np.array
        3xM array of coordinates of the vectors. Each row corresponds to x,y,z
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        3xM array of field values of the vectors. Each row corresponds to u,v,w
        respectively, i.e. the velocity of the field in each direction.
        The column entries are the respective values.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
    """
    coordinates = np.array(coordinates)
    field_values = np.array(field_values)
    fig = go.Figure(
        go.Cone(
            x=coordinates[0, :],
            y=coordinates[1, :],
            z=coordinates[2, :],
            u=field_values[0, :],
            v=field_values[1, :],
            w=field_values[2, :],
            **kwargs,
        )
    )
    #  sizemode=sizemode, sizeref=2.5, anchor='tail'))
    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )

    return fig


def vector_field_animation(coordinates, field_values, add_lineplot=False, **kwargs):
    # TODO improvements
    coordinates = np.array(coordinates)
    fig = vector_field_plot(coordinates, field_values, add_lineplot, **kwargs)
    frames = []
    for i, p in enumerate(coordinates.T):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=[p[0]],
                        y=[p[1]],
                        z=[p[2]],
                        mode="markers",
                        marker=dict(color="red", size=5),
                    )
                ]
            )
        )
    fig.add_trace(fig.data[0])
    fig.frames = frames
    fig.update_layout(
        title="Slices in volumetric data",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 0,
                                    "easing": "quadratic-in-out",
                                },
                            },
                        ],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "&#9616;&#9616;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 0},
                "type": "buttons",
                "x": 0.07,
                "y": -0.1,
            }
        ],
        margin=dict(l=10, r=10, b=10, t=50),
        #  uirevision=True
    )
    return fig


def curve_animation(parametric_equation, time_range, dt, **kwargs):
    t0, tf = time_range
    time = np.arange(t0, tf, dt)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[], y=[], z=[], mode="markers", marker=dict(color="red", size=10)
            )
        ]
    )
    frames = []
    for t in time:
        p = parametric_equation(time=t)
        frames.append(
            go.Frame(
                data=[go.Scatter3d(x=p[0, :], y=p[1, :], z=p[2, :], mode="markers")]
            )
        )

    fig.update_layout(
        title="Slices in volumetric data",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 0,
                                    "easing": "quadratic-in-out",
                                },
                            },
                        ],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "&#9616;&#9616;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 0},
                "type": "buttons",
                "x": 0.07,
                "y": -0.1,
            }
        ],
        margin=dict(l=10, r=10, b=10, t=50),
        #  uirevision=True
    )

    fig.update(frames=frames)
    return fig


def quiver_plot3d(vector_field, x_range, y_range, z_range, N):
    # TODO doesnt work
    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    z = np.linspace(z_range[0], z_range[1], N)
    # fig = make_subplots(rows=1, cols=3)

    vf, xs, ys = [], [], []
    for xy in product(x, y):
        xs.append(xy[0])
        ys.append(xy[1])
        coords = np.array([xy[0], xy[1], 0])
        vf.append(vector_field(coords).T)
    vf = np.array(vf).reshape(-1, 3)
    fig1 = ff.create_quiver(xs, ys, vf[:, 0], vf[:, 1])

    vf, xs, zs = [], [], []
    for xz in product(x, z):
        xs.append(xz[0])
        zs.append(xz[1])
        coords = np.array([xz[0], 0, xz[1]])
        vf.append(vector_field(coords).T)
    vf = np.array(vf).reshape(-1, 3)
    fig2 = ff.create_quiver(xs, zs, vf[:, 0], vf[:, 2])

    vf, ys, zs = [], [], []
    for yz in product(y, z):
        ys.append(yz[0])
        zs.append(yz[1])
        coords = np.array([0, yz[0], yz[1]])
        vf.append(vector_field(coords).T)
    vf = np.array(vf).reshape(-1, 3)
    fig3 = ff.create_quiver(ys, zs, vf[:, 1], vf[:, 2])

    subplots = make_subplots(rows=1, cols=3)
    # fig1, fig2, fig3 = fig

    for d in fig1.data:
        subplots.add_trace(go.Scatter(x=d["x"], y=d["y"]), row=1, col=1)

    for d in fig2.data:
        subplots.add_trace(go.Scatter(x=d["x"], y=d["y"]), row=1, col=2)

    for d in fig3.data:
        subplots.add_trace(go.Scatter(x=d["x"], y=d["y"]), row=1, col=3)

    return subplots
