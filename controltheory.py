# %%
from abc import ABC, abstractmethod
import warnings
import sys
import numpy as np
from cvxopt import matrix, solvers, spmatrix

sys.path.insert(1, "/home/fbartelt/Documents/UFMG/TCC/Sim/uaibot")
import uaibot as ub

solvers.options["show_progress"] = False


class ControlledSystem:
    """Controlled System.
    Attributes
    ----------
    states : list
        List of the states of the controlled system.
    controllers : list
        List of the controllers of the system.
    system : System
        System that will be controlled.
    """

    def __init__(self, states, system=None, controllers=None):
        self.states = states
        self.controllers = controllers or []
        self.system = system

    def add_controller(self, controller):
        self.controllers.append(controller)

    def add_system(self, system):
        self.system = system

    def evolve(self):
        states = self.system.get_states()
        output = np.zeros((len(self.states), 1))
        for controller in self.controllers:
            output = controller.control(states, input_=output)
        output = self.system.evolve(output)

    def _update(self):
        self.evolve()


class System(ABC):
    """Abstract class for systems."""

    def __init__(self, states) -> None:
        self.states = states

    @abstractmethod
    def evolve(self):
        pass

    @abstractmethod
    def get_states(self):
        pass

    def _update(self, *args, **kwargs):
        self.evolve(*args, **kwargs)

class DynamicSystem(System):
    """Abstract class for dynamic (2nd order) systems"""

    def __init__(self, states) -> None:
        super().__init__(states)

    @abstractmethod
    def get_dynamics(self, *args, **kwargs):
        pass

class Controller(ABC):
    """Abstract class for controllers."""

    def __init__(self, controlled_states, states_map=None) -> None:
        self.controlled_states = controlled_states
        self.map_function = states_map or self.default_states_map

    @abstractmethod
    def control(self, input_):
        pass

    def default_states_map(self, states, *args, **kwargs):
        states = {key: states[key] for key in self.controlled_states if key in states}
        return np.array(list(states.values())).reshape(-1, 1)

    def _update(self, *args, **kwargs):
        self.control(*args, **kwargs)


class Trajectory(ABC):
    """TODO"""

    pass


class PseudoInverseController(Controller):
    """Pseudo Inverse Controller.
     output = -K * pinv(J) * x, where x is the map of the states.

    Attributes
    ----------
    controlled_states : list
        List of the states that will be controlled.
    K : np.array
        Gain matrix.
    J : function
        Function that returns the Jacobian matrix.
    """

    def __init__(self, controlled_states, K, J, states_map=None):
        super().__init__(controlled_states, states_map)
        self.K = K
        self.J = J

    def control(self, states, input_):
        x = self.map_function(states, input_)
        J = self.J(self.default_states_map(states))
        return -self.K @ np.linalg.pinv(J) @ x

class QPController(Controller):
    """Quadratic Programming Controller. Solves the following optimization problem:
    min (1/2)x^THx + f^Tx
    s.t. Ax <= b
         Cx = d
    """

    def __init__(self, controlled_states, K, A=None, b=None, C=None, d=None) -> None:
        super().__init__(controlled_states)
        self.K = K
        self.A = A
        self.b = b
        self.C = C
        self.d = d

    def control(self, J, eps=1e-3, ignore_constraints=False):
        n = len(self.controlled_states)
        H = 2 * (J * J + eps * np.identity(n))
        f = self.K * 2 * (e @ J).T
        error_qp = False

        if ignore_constraints:
            A = spmatrix([], [], [], (0, n), "d")
            b = matrix(0.0, (0, 1))
            C = spmatrix([], [], [], (0, n), "d")
            d = matrix(0.0, (0, 1))
        else:
            A = matrix(self.A)
            b = matrix(self.b)
            C = matrix(self.C)
            d = matrix(self.d)
        try:
            output = solvers.qp(H, f, A, b, C, d)["x"]
        except:
            output = np.matrix(np.zeros((n, 1)))
            error_qp = True
            warnings.warn("Quadratic Programming did not converge")

        output = np.array(output).reshape(-1, 1)
        return output, error_qp
    
class ComputedTorque(Controller):
    def __init__(self, controlled_states, K, states_map=None) -> None:
        super().__init__(controlled_states, states_map)
        self.K = K
    
    def control(self, states, input_, M, C, G):
        """ input_ = [[q_des], [qdot_des], [qddot_des]]
        """
        # x = self.map_function(states, input_)
        n = len(states)
        x = np.vstack()
        torque = C + G + (M @ (qddot_des + Kp @ (q_des - q) + Kd @ (qdot_des - qdot)) )
        return -self.K @ np.linalg.pinv(J) @ x
    
class UAIBotSystem(System):
    def __init__(self, states, robot, dt=1e-3) -> None:
        super().__init__(states)
        self.robot = robot
        self.n = len(robot.links)
        self.dt = dt
        self.histq = [self.robot.q]
    
    def get_states(self):
        q = self.robot.q
        states = {self.states[i]: q[i,0] for i in range(self.n)}
        return states
    
    def evolve(self, qdot):
        q = self.robot.q + qdot * self.dt
        self.histq.append(q)
        i = len(self.histq) - 1
        robot.add_ani_frame(time=i*self.dt, q=q)

class UAIBotDynSystem(DynamicSystem):
    def __init__(self, states, robot, dt=1e-3) -> None:
        super().__init__(states)
        self.robot = robot
        self.n = len(robot.links)
        self.dt = dt
        self.hist = [np.vstack([np.array(self.robot.q).T, np.zeros((1, self.n)), np.zeros((1, self.n))])] # q, qdot, qddot
    
    def get_states(self):
        q = self.robot.q
        states = {self.states[i]: q[i,0] for i in range(self.n)}
        return states
    
    def get_dynamics(self, q, qdot):
        return self.robot.dyn_model(q, qdot)
    
    def evolve(self, torque):
        M, C, G = self.get_dynamics(self.hist[-1][0, :], self.hist[-1][1, :])
        qddot = np.array(np.linalg.inv(M) @ (-C - G + torque))
        qdot = self.hist[-1][1, :].reshape(-1, 1) + qddot * self.dt
        q = self.hist[-1][0, :].reshape(-1, 1) + qdot * self.dt
        self.hist.append(np.vstack([q.T, qdot.T, qddot.T]))
        i = len(self.hist) - 1
        robot.add_ani_frame(time=i*self.dt, q=q)


# %%
robot = ub.Robot.create_kuka_kr5(name="jacotest")
sim = ub.Simulation.create_sim_grid([robot])
n = len(robot.links)
kukasys = UAIBotSystem([f"q{i}" for i in range(len(robot.links))], robot, dt=1e-2)


def states_map(states, *args):
    states = pinv.default_states_map(states)
    p = robot.fkm()[:3, -1]
    return p - np.array([0.5, 0.5, 0.5]).reshape(-1, 1)


pinv = PseudoInverseController(
    [f"q{i}" for i in range(len(robot.links))],
    K=np.eye(n),
    J=lambda x: robot.jac_geo(x)[0][:3, :],
    states_map=states_map,
)
kukacontrol = ControlledSystem(kukasys.states, kukasys, [pinv])

for i in range(300):
    kukacontrol.evolve()
# %%
from uaibot.demo.demo import Demo

Demo.control_demo_1()
# %%
