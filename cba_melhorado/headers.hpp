#pragma once
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class AdaptiveController {
public:
  MatrixXd Gamma_o_inv, Gamma_r_inv, Kd;
  VectorXd o_i;
  std::vector<Vector3d> r_i;
  Matrix3d I_p;
  float m;
  int N;
  Vector3d r_p;
  double tol;
  std::vector<MatrixXd> aprox_hist;
  std::vector<VectorXd> input_hist;
  std::vector<std::vector<VectorXd>> taui_hist;

  AdaptiveController(const MatrixXd &Gamma_o_inv_init,
                     const MatrixXd &Gamma_r_inv_init, const MatrixXd &Kd_init,
                     const VectorXd &o_i_init, std::vector<Vector3d> r_i_init,
                     const Matrix3d &I_p_init, const float m_init,
                     const int N_init, const Vector3d &r_p_init)
      : Gamma_o_inv(Gamma_o_inv_init), Gamma_r_inv(Gamma_r_inv_init),
        Kd(Kd_init), o_i(o_i_init), r_i(r_i_init), I_p(I_p_init), m(m_init),
        N(N_init), r_p(r_p_init) {}

  std::tuple<VectorXd, std::vector<VectorXd>, std::vector<VectorXd>, 
             std::vector<VectorXd>>
  adaptiveDynamics(const MatrixXd &p, const MatrixXd &R, const MatrixXd &xi,
                   const VectorXd &psi, const VectorXd &psi_dot,
                   const std::vector<VectorXd> &o_hat,
                   const std::vector<Vector3d> &r_hat);
};

class ControlLoop {
public:
  // System parameters
  Matrix4d H0;      // HTM state
  VectorXd xi0;     // Twist state
  VectorXd xi_dot0; // Twist derivative state
  std::vector<Matrix4d> curve;
  std::vector<MatrixXd> curve_derivative;
  AdaptiveController controller;
  std::vector<VectorXd> o_hat0;
  std::vector<Vector3d> r_hat0;
  // Vector Field params
  float kt1, kt2, kt3, kn1, kn2;
  float delta, ds;
  // Adaptive params
  float deadband;
  // Simulation params
  double dt;
  double T;
  // Logging
  std::vector<Matrix4d> H_hist;
  std::vector<VectorXd> xi_hist;
  std::vector<VectorXd> xi_dot_hist;
  std::vector<VectorXd> psi_hist;
  std::vector<std::vector<VectorXd>> o_hat_hist;
  std::vector<std::vector<Vector3d>> r_hat_hist;
  std::vector<int> closest_indexes;
  std::vector<double> min_distances;
  std::vector<double> zeta_hist;

  ControlLoop(const Matrix4d &H0_init, const VectorXd &xi0_init,
              const VectorXd &xi_dot0_init,
              const std::vector<Matrix4d> &curve_init,
              const std::vector<MatrixXd> &curve_derivative_init,
              const AdaptiveController &controller_init,
              const std::vector<VectorXd> &o_hat0_init,
              const std::vector<Vector3d> &r_hat0_init, float kt1_init,
              float kt2_init, float kt3_init, float kn1_init, float kn2_init,
              float delta_init, float ds_init, float deadband_init,
              double dt_init, double T_init)
      : H0(H0_init), xi0(xi0_init), xi_dot0(xi_dot0_init), curve(curve_init),
        curve_derivative(curve_derivative_init), controller(controller_init),
        o_hat0(o_hat0_init), r_hat0(r_hat0_init), kt1(kt1_init), kt2(kt2_init),
        kt3(kt3_init), kn1(kn1_init), kn2(kn2_init), delta(delta_init),
        ds(ds_init), deadband(deadband_init), dt(dt_init), T(T_init) {}

  void simulate();
};
