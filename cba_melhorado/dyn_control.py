import numpy as np
import uaibot as ub
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance
from uaibot.utils import Utils
import matplotlib.pyplot as plt
from pathlib import Path
import webbrowser


def simulate_dynamics(robot, q0, qdot0, qd, Kp, Kd, dt=1e-3, T=5.0, g0=np.array([0, 0, -9.81])):
    n = len(q0)
    steps = int(T / dt)
    q0 = np.array(q0)
    qdot0 = np.array(qdot0)
    q, qdot = q0.copy().reshape(-1, 1), qdot0.copy().reshape(-1, 1)
    qd = np.array(qd).reshape(-1, 1)
    q_log, qdot_log = [q.copy()], [qdot.copy()]

    for i in range(steps):
        # Compute dynamics
        M, C_, g = robot.dynamic_model(q, qdot)  # from C++ NE (trusted one)
        
        # Control law (PD + gravity compensation)
        tau = Kp @ (qd - q) + Kd @ (0 - qdot) + g
        # print(f"q shape: {q.shape}, tau shape: {tau.shape}, qdot shape: {qdot.shape}, g shape: {g.shape}")
        tau = tau.reshape(-1, 1)

        # Forward dynamics: M qddot = tau - C qdot - g
        qddot = np.linalg.solve(M, tau - C_ - g)

        # Integrate (semi-implicit Euler)
        qdot += qddot * dt
        q += qdot * dt

        # Log
        q_log.append(q.copy())
        qdot_log.append(qdot.copy())
        robot.add_ani_frame(time=i * dt, q=q)



robot = ub.Robot.create_kinova_gen3()
rng = np.random.default_rng(42)
N = len(robot.links)
robot._joint_limit = np.matrix(
        [
            [-10*np.pi, 10*np.pi],
            [-10*np.pi, 10*np.pi],
            [-10*np.pi, 10*np.pi],
            [-10*np.pi, 10*np.pi],
            [-10*np.pi, 10*np.pi],
            [-10*np.pi, 10*np.pi],
            [-10*np.pi, 10*np.pi],
                    ]
    )
print(robot.joint_limit)
qd = robot.ikm(
    # htm_tg=robot.fkm(axis='eef'),
    htm_tg=Utils.trn([0.1, 0, 0.4]) @ Utils.rotx(np.pi/2),
    check_joint=False, check_auto=False
)

sim = ub.Simulation.create_sim_grid(robot)
q0 = robot.q.copy()
T = 10.0
dt = 1e-3
qdot0 = np.zeros(N)
n = len(q0)
steps = int(T / dt)
q0 = np.array(q0)
qdot0 = np.array(qdot0)
q, qdot = q0.copy().reshape(-1, 1), qdot0.copy().reshape(-1, 1)
qd = np.array(qd).reshape(-1, 1)
q_int = np.zeros_like(q0)
q_log, qdot_log = [q.copy()], [qdot.copy()]
factor = 1
Kp = np.diag([10]*N) * factor
Kd = np.diag([5]*N) * factor
Ki = np.diag([5]*N) * factor

for i in range(steps):
    # Compute dynamics
    M, C_, g = robot.dynamic_model(q, qdot)  # from C++ NE (trusted one)
    
    # Control law (PD + gravity compensation)

    prop = Kp @ (qd.ravel() - q.ravel()).reshape(-1, 1)
    integ = Ki @ (q_int.ravel()).reshape(-1, 1)
    deriv = Kd @ (-qdot.ravel()).reshape(-1, 1)
    tau = g + prop + integ + deriv
    # tau = g + Kp @ (q.ravel() - qd.ravel()) # + Kd @ (q_dot_d - qdot) + Ki @ (q_int) + g
    # print(f"q shape: {q.shape}, tau shape: {tau.shape}, qdot shape: {qdot.shape}, g shape: {g.shape}")
    tau = tau.reshape(-1, 1)
    tau = np.clip(tau, -10, 10)
    # print([link.mass for link in robot.links])
    C_ = np.array(C_).reshape(-1, 1)
    print(f"q error: {np.linalg.norm(qd - q)}")

    # Forward dynamics: M qddot = tau - C qdot - g
    # qddot = np.linalg.solve(M, tau - C_ - g)
    Minv = ub.Utils.dp_inv(M)
    qddot = Minv @ (tau - C_ - g)

    # Integrate (semi-implicit Euler)
    qdot += qddot * dt
    q_int += (qd - q) * dt
    q += qdot * dt

    # Log
    q_log.append(q.copy())
    qdot_log.append(qdot.copy())
    robot.add_ani_frame(time=i * dt, q=q.ravel())


# simulate_dynamics(
#     robot,
#     q0=robot.q,
#     qdot0=np.zeros(N),
#     qd=qd,
#     Kp=np.diag([40]*N),
#     Kd=np.diag([5]*N),
#     dt=1e-2,
#     T=1.0
# )
#
sim.set_parameters(width=800, height=600)
sim.save("./", f"dyn_test")
print("Saved")

# open browser with html file (neovim workaround)
def open_in_browser(filename: str):
    """
    Opens an HTML file in the system's default web browser.
    Works cross-platform (Linux, macOS, Windows).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())

open_in_browser("./dyn_test.html")
