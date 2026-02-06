#include <eigen3/Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "headers.hpp"
#include <eigen3/Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(adaptive_cpp, m) {
  m.doc() = "Cpp adaptive controller module"; // optional module docstring
  py::class_<AdaptiveController>(m, "AdaptiveController")
      .def(py::init<const MatrixXd &, const MatrixXd &, const MatrixXd &,
                    const VectorXd &, std::vector<Vector3d>, const Matrix3d &,
                    const float, const int, const Vector3d &>(),
           py::arg("Gamma_o_inv_init"), py::arg("Gamma_r_inv_init"),
           py::arg("Kd_init"), py::arg("o_i_init"), py::arg("r_i_init"),
           py::arg("I_p_init"), py::arg("m_init"), py::arg("N_init"),
           py::arg("r_p_init"))
      .def("adaptiveDynamics", &AdaptiveController::adaptiveDynamics,
           py::arg("p"), py::arg("R"), py::arg("xi"), py::arg("psi"),
           py::arg("psi_dot"), py::arg("o_hat"), py::arg("r_hat"))
      .def_readonly("aprox_hist", &AdaptiveController::aprox_hist)
      .def_readonly("input_hist", &AdaptiveController::input_hist)
      .def_readonly("r_i", &AdaptiveController::r_i)
      .def_readonly("taui_hist", &AdaptiveController::taui_hist);
  py::class_<ControlLoop>(m, "ControlLoop")
      .def(
          py::init<const Matrix4d &, const VectorXd &, const VectorXd &,
                   const std::vector<Matrix4d> &, const std::vector<MatrixXd> &,
                   const AdaptiveController &, const std::vector<VectorXd> &,
                   const std::vector<Vector3d> &, const float, const float,
                   const float, const float, const float, const float,
                   const float, const float, const double, const double>(),
          py::arg("H0"), py::arg("xi0"), py::arg("xi_dot0"), py::arg("curve"),
          py::arg("curve_derivative"), py::arg("controller"), py::arg("o_hat0"),
          py::arg("r_hat0"), py::arg("kt1"), py::arg("kt2"), py::arg("kt3"),
          py::arg("kn1"), py::arg("kn2"), py::arg("delta"), py::arg("ds"),
          py::arg("deadband"), py::arg("dt"), py::arg("T"))
      .def("simulate", &ControlLoop::simulate)
      .def_readonly("controller", &ControlLoop::controller)
      .def_readonly("H_hist", &ControlLoop::H_hist)
      .def_readonly("xi_hist", &ControlLoop::xi_hist)
      .def_readonly("xi_dot_hist", &ControlLoop::xi_dot_hist)
      .def_readonly("psi_hist", &ControlLoop::psi_hist)
      .def_readonly("o_hat_hist", &ControlLoop::o_hat_hist)
      .def_readonly("r_hat_hist", &ControlLoop::r_hat_hist)
      .def_readonly("closest_indexes", &ControlLoop::closest_indexes)
      .def_readonly("min_distances", &ControlLoop::min_distances)
      .def_readonly("zeta_hist", &ControlLoop::zeta_hist);
}
