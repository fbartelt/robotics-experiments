# A Differentiable Distance Metric for Robotics Through Generalized Alternating Projection

This repository contains the code and scripts required to **reproduce the results** presented in the paper:

> **A Differentiable Distance Metric for Robotics Through Generalized Alternating Projection**
> *Vinicius M. Gonçalves, Shiqing Wei, Eduardo Malacarne Soeiro de Souza,*
> *Prashanth Krishnamurthy, Anthony Tzes, Farshad Khorrami*
> **IEEE Robotics and Automation Letters (RA-L)**

---

## 📄 Paper overview

The paper introduces a methodology to compute a **differentiable variant of the Euclidean distance** between two convex objects.
The proposed distance is guaranteed to be differentiable and is obtained by generalizing the classical **von Neumann alternating projection algorithm** to convex sets.

For background on the classical method, see:
[https://en.wikipedia.org/wiki/Projections_onto_convex_sets](https://en.wikipedia.org/wiki/Projections_onto_convex_sets)

---

## 🔧 Requirements

To run the scripts in this repository, you must install the **UAIBot** Python package:

* **UAIBot GitHub repository:**
  [https://github.com/UAIbot/UAIbotPy](https://github.com/UAIbot/UAIbotPy)
* **Installation:**

  ```bash
  pip install uaibot
  ```

---

## 📐 Differentiable distance in UAIBot

The differentiable distance between two objects is implemented in **UAIBot** through the function `compute_dist`, using the parameters `h` and `eps`.
By default, `compute_dist` returns the standard Euclidean distance if these parameters are not provided.

In the notation of the paper:

* The parameter **σ (sigma)** is internally computed by UAIBot based on `eps`
* The parameter **k** is fixed to `k = 1`

---

### Example usage

```python
import uaibot as ub

obj1 = ub.Box(
    htm=ub.Utils.trn([1, 0, 0]),
    width=0.1,
    depth=0.2,
    height=0.3
)

obj2 = ub.Cylinder(
    htm=ub.Utils.trn([-1, 0, 0]) * ub.Utils.rotx(0.3),
    radius=0.2,
    height=0.4
)

p1, p2, dist, hist_error = obj1.compute_dist(obj2, h=0.05, eps=0.001)
```

* `p1` and `p2`: **differentiable witness points**, obtained from the generalized von Neumann algorithm
* `dist`: **differentiable distance**
* `hist_error`: history of the convergence error across iterations

Additional parameters include:

* `tol`, that specifies the tolerance for convergence, that is, the algorithm is deemed 
to converge when ||a[k+1]-a[k]|| <= tol, in which a[k] is the differentiable witness point of the first object at the 
k-th step.

* `no_iter_max`, that specifies the maximum number of iterations.

It is also possible to compute the distance between a robot and object for a given configuration `q_try`:

```python
import uaibot as ub

robot = ub.Robot.create_kuka_kr5()
q_try = [0.,0.,0.,0.,0.,0.]

obj1 = ub.Box(
    htm=ub.Utils.trn([1, 0, 0]),
    width=0.1,
    depth=0.2,
    height=0.3
)

dr = robot.compute_dist(q = q_try, obj = obj1, h = 0.05, eps=0.001)

```

`dr` is a structure that contains a lot of information about this distance computation. Among them:

* `dr.dist_vect` is a mx1 `np.matrix`, in which m is the number of primitives that compose the robot. Each row of this matrix is the differentiable distance between the i-th primitive and the object `obj`.

* `dr.jac_dist_mat` is a mxn `np.matrix`, in which m is the number of primitives that compose the robot and n is the number of joints. The i-th row of this matrix is the gradient of the differentiable distance between the i-th primitive of the robot and `obj` with respect to the configuration q

  
---

## 📂 Repository contents

### `paper_experiment.py`

Reproduces the main experiment reported in the paper.

* When run as provided, it generates a **UAIBot simulation** of the experiment.
* To deploy on a **real robot**, the user must replace:

  * `get_joint_config`
  * `send_joint_velocity`
    with their hardware-specific counterparts.

---

### `time_comparison.py`

Generates a time comparison between:

* the proposed **differentiable distance**, and
* the standard **Euclidean distance**.

The paper also reports a comparison with the distance introduced in:

> V. M. Gonçalves, A. Tzes, F. Khorrami, and P. Fraisse,
> *“Smooth distances for second order kinematic robot control,”*
> **IEEE Transactions on Robotics**, vol. 40, pp. 2950–2966, 2024.

This comparison is **not included** in the present repository, since that distance is not implemented by default in UAIBot.
If you wish to perform this comparison, please contact:

**[vinicius.marianog@gmail.com](mailto:vinicius.marianog@gmail.com)**

---

### `scalability_sizes.py`

Generates a scalability experiment by varying the **number of faces** of the polyhedral objects.

The results show that the **computational time scales approximately linearly** with the number of faces.

---


