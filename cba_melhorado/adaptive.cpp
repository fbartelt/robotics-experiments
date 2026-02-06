#include "headers.hpp"
#include "declarations.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <tuple>
#include <vector>

using namespace Eigen;
void printProgressBar2(int current, int imax) {
  // Calculate percentage
  int percent = static_cast<int>(100.0 * current / imax);

  // Calculate the number of "=" to show in the progress bar
  int barWidth = 50; // Width of the progress bar in characters
  int pos = barWidth * current / imax;

  // Create the progress bar
  std::string progressBar =
      "[" + std::string(pos, '=') + std::string(barWidth - pos, ' ') + "]";

  // Print the progress bar with the percentage
  std::cout << "\r" << progressBar << " " << percent << "%";
  std::cout.flush(); // Ensure the output is immediately printed
}

Eigen::Matrix3d skew2(const Eigen::Vector3d &q) {
  Eigen::Matrix3d skewMatrix;
  skewMatrix << 0, -q(2), q(1), q(2), 0, -q(0), -q(1), q(0), 0;
  return skewMatrix;
}

// Maps a 3x1 vector to the custom iota matrix
MatrixXd iota(const Eigen::Vector3d &v) {
  // Create the iota matrix (3x6) as a dynamic-sized matrix
  MatrixXd LMatrix(3, 6);
  LMatrix << v(0), v(1), v(2), 0, 0, 0, 0, v(0), 0, v(1), v(2), 0, 0, 0, v(0),
      0, v(1), v(2);
  return LMatrix;
}

// class AdaptiveController {
// public:
//   // Class properties to replace global variables
//   MatrixXd Gamma_o_inv, Gamma_r_inv, Kd;
//   VectorXd o_i;
//   Vector3d r_p;
//   Matrix3d I_p;
//   float m;
//   int N;
//   double tol;
//   std::vector<MatrixXd> aprox_hist;
//   std::vector<VectorXd> input_hist;
//   std::vector<Vector3d> r_i;
//   std::vector<std::vector<VectorXd>> taui_hist;
//
//   AdaptiveController(const MatrixXd &Gamma_o_inv_init, const MatrixXd
//   &Gamma_r_inv_init,
//                      const MatrixXd &Kd_init, const VectorXd &o_i_init,
//                      std::vector<Vector3d> r_i_init, const Matrix3d
//                      &I_p_init, const float m_init, const int N_init, const
//                      Vector3d &r_p_init)
//       : Gamma_o_inv(Gamma_o_inv_init), Gamma_r_inv(Gamma_r_inv_init),
//       Kd(Kd_init),
//         o_i(o_i_init), r_i(r_i_init), I_p(I_p_init), m(m_init), N(N_init),
//         r_p(r_p_init) {}
//
std::tuple<VectorXd, std::vector<VectorXd>, std::vector<VectorXd>,
           std::vector<VectorXd>>
AdaptiveController::adaptiveDynamics(const MatrixXd &p, const MatrixXd &R,
                                     const MatrixXd &xi, const VectorXd &psi,
                                     const VectorXd &psi_dot,
                                     const std::vector<VectorXd> &o_hat,
                                     const std::vector<Vector3d> &r_hat) {

  // std::cout << "[DEBUG] Entered adaptiveDynamics" << std::endl;
  // std::cout << "p: " << p.rows() << "x" << p.cols() << "\n";
  // std::cout << "R: " << R.rows() << "x" << R.cols() << "\n";
  // std::cout << "xi: " << xi.rows() << "x" << xi.cols() << "\n";
  // std::cout << "psi: " << psi.rows() << "x" << psi.cols() << "\n";
  // std::cout << "psi_dot: " << psi_dot.rows() << "x" << psi_dot.cols() <<
  // "\n";
  //
  // for (size_t i = 0; i < o_hat.size(); i++)
  //   std::cout << "o_hat[" << i << "]: " << o_hat[i].rows() << "x"
  //             << o_hat[i].cols() << "\n";
  // for (size_t i = 0; i < r_hat.size(); i++)
  //   std::cout << "r_hat[" << i << "]: " << r_hat[i].rows() << "x"
  //             << r_hat[i].cols() << "\n";
  Vector3d v = xi.block<3, 1>(0, 0);
  Vector3d omega = xi.block<3, 1>(3, 0);

  double norm_vel = psi.norm();

  // Reference signals
  Vector3d omega_r = psi.segment<3>(3);
  Vector3d v_r = psi.segment<3>(0);

  VectorXd zeta = xi - psi;
  Vector3d omega_dot_r = psi_dot.segment<3>(3);
  Vector3d v_dot_r = psi_dot.segment<3>(0);

  // Compute regressors
  MatrixXd Y_o_l(3, 10);
  Y_o_l.block<3, 1>(0, 0) = v_dot_r;
  Y_o_l.block<3, 3>(0, 1) =
      -skew2(omega_dot_r) * R - skew2(omega) * skew2(omega_r) * R;
  Y_o_l.block<3, 6>(0, 4) = MatrixXd::Zero(3, 6);

  MatrixXd R_transpose = R.transpose().eval();

  MatrixXd Y_o_r(3, 10);
  Y_o_r.block<3, 1>(0, 0) = MatrixXd::Zero(3, 1); // First column is zero
  Y_o_r.block<3, 3>(0, 1) = skew2(v_dot_r) * R + skew2(omega) * skew2(v_r) * R -
                            skew2(omega_r) * skew2(v) * R;
  Y_o_r.block<3, 6>(0, 4) = R * iota(R_transpose * omega_dot_r) +
                            skew2(omega) * R * iota(R_transpose * omega_r);

  MatrixXd Y_o(6, 10);
  // Concatenate Y_l and Y_r into Y_o
  Y_o.block<3, 10>(0, 0) = Y_o_l;
  Y_o.block<3, 10>(3, 0) = Y_o_r;

  // std::cout << "[DEBUG] Created Y_o" << std::endl;
  // True dynamics matrices
  MatrixXd M(6, 6), C(6, 6), M_dot(6, 6);
  // Compute blocks for H
  M.block<3, 3>(0, 0) = m * Matrix3d::Identity();       // Top-left block
  M.block<3, 3>(0, 3) = m * skew2(R * r_p);              // Top-right block
  M.block<3, 3>(3, 0) = -m * skew2(R * r_p);             // Bottom-left block
  M.block<3, 3>(3, 3) = R * I_p * R.transpose().eval(); // Bottom-right block
  // std::cout << "[DEBUG] Created M" << std::endl;

  // Compute blocks for C
  C.block<3, 3>(0, 0) = Matrix3d::Zero();                 // Top-left block
  C.block<3, 3>(0, 3) = m * skew2(omega) * skew2(R * r_p);  // Top-right block
  C.block<3, 3>(3, 0) = -m * skew2(omega) * skew2(R * r_p); // Bottom-left block
  C.block<3, 3>(3, 3) =
      skew2(omega) * R * I_p * R_transpose // Bottom-right block
      - m * skew2(skew2(R * r_p) * v);
  // std::cout << "[DEBUG] Created C" << std::endl;
  // Adaptive control law
  VectorXd input = VectorXd::Zero(6);
  std::vector<VectorXd> eta(N, VectorXd::Zero(6));
  std::vector<VectorXd> kth_taui;

  for (int i = 0; i < N; ++i) {
    // std::cout << "[DEBUG] Iteration " << i << std::endl;
    MatrixXd Lambda_inv(6, 6);
    Lambda_inv.block<3, 3>(0, 0) = Matrix3d::Identity();
    Lambda_inv.block<3, 3>(0, 3) = Matrix3d::Zero();
    Lambda_inv.block<3, 3>(3, 0) = -skew2(R * r_hat[i]);
    Lambda_inv.block<3, 3>(3, 3) = Matrix3d::Identity();
    // std::cout << "[DEBUG] Created Lambda_inv" << std::endl;
    // Real grasp matrix Lambda
    MatrixXd Lambda(6, 6);
    Lambda.block<3, 3>(0, 0) = Matrix3d::Identity();
    Lambda.block<3, 3>(0, 3) = Matrix3d::Zero();
    Lambda.block<3, 3>(3, 0) = skew2(R * r_i[i]);
    Lambda.block<3, 3>(3, 3) = Matrix3d::Identity();

    eta[i] = Y_o * o_hat[i] - Kd * zeta;
    // std::cout << "[DEBUG] Created eta[" << i << "]" << std::endl;
    VectorXd tau_i = Lambda_inv * eta[i];
    // std::cout << "[DEBUG] Created tau_i" << std::endl;
    kth_taui.push_back(tau_i);
    // Compute real applied torque
    input += Lambda * tau_i;
  }

  taui_hist.push_back(kth_taui);
  input_hist.push_back(input);

  MatrixXd M_pinv =
      (M.transpose() * M + 0.01 * MatrixXd::Identity(6, 6)).inverse() *
      M.transpose();
  // std::cout << "[DEBUG] Created M_pinv" << std::endl;
  VectorXd ddchi = M_pinv * (input - C * xi);
  // std::cout << "[DEBUG] Created ddchi" << std::endl;

  std::vector<MatrixXd> Y_r(N, MatrixXd(6, 3)), g_o(N, MatrixXd(10, 10)),
      g_r(N, MatrixXd(3, 3));
  std::vector<VectorXd> r_hat_dot(N, VectorXd(3)), o_hat_dot(N, VectorXd(10));

  for (int i = 0; i < N; ++i) {
    Y_r[i].block<3, 3>(0, 0) = Matrix3d::Zero();
    Y_r[i].block<3, 3>(3, 0) = skew2(eta[i].head(3)) * R;
    g_o[i] = Gamma_o_inv.inverse();
    g_r[i] = Gamma_r_inv.inverse();

    MatrixXd Y_o_transpose = Y_o.transpose().eval();
    MatrixXd Y_r_tranpose = Y_r[i].transpose().eval();
    o_hat_dot[i] = -g_o[i] * Y_o_transpose * zeta;
    r_hat_dot[i] = -g_r[i] * Y_r_tranpose * zeta;
    // a_t[i] = o_hat[i] - a_i;
    // r_t[i] = r_hat[i] - r_i[i];
  }

  return {ddchi, o_hat_dot, r_hat_dot, kth_taui};
}
// };

void ControlLoop::simulate() {
  // Initialize states
  Matrix4d H = H0;
  VectorXd xi = xi0;
  VectorXd xi_dot = xi_dot0;
  std::vector<VectorXd> o_hat = o_hat0;
  std::vector<Vector3d> r_hat = r_hat0;

  // double tol = 1e-5;
  int imax = T / dt;
  int N = controller.N;

  for (int i = 0; i < imax; ++i) {
    printProgressBar2(i, imax);
    // Extract position and orientation
    Matrix3d R = H.block<3, 3>(0, 0);
    Vector3d p = H.block<3, 1>(0, 3);
    // Compute the twist
    VectorFieldResult vfres = vectorfield_SE3(H, curve, kt1, kt2, kt3, kn1, kn2,
                                   curve_derivative, delta, ds);
    VectorXd psi = vfres.twist.cast<double>();
    double dist = vfres.dist;
    int nearest_index = vfres.index;

    VectorXd psi_next =
        vectorfield_SE3(expSE3(dt * SmapSE3(psi)) * H, curve, kt1, kt2, kt3,
                        kn1, kn2, curve_derivative, delta, ds).twist.cast<double>();
    VectorXd psi_prev =
        vectorfield_SE3(expSE3(-dt * SmapSE3(psi)) * H, curve, kt1, kt2, kt3,
                        kn1, kn2, curve_derivative, delta, ds).twist.cast<double>();
    VectorXd psi_dot = (psi_next - psi_prev) / (2 * dt);

    // Compute the next pose
    VectorXd s = xi - psi;
    if (i % 100 == 0){
      std::cout << "s norm: " << s.norm() << std::endl;
    }

    // First step of Heun -- Euler method
    auto [ddq, do_hat, dr_hat, tau_i_list] =
        controller.adaptiveDynamics(p, R, xi, psi, psi_dot, o_hat, r_hat);

    std::vector<VectorXd> o_hat_int(N, VectorXd::Zero(10));
    std::vector<Vector3d> r_hat_int(N, Vector3d::Zero());

    for (int j = 0; j < N; ++j) {
      if (s.norm() > deadband) {
        o_hat_int[j] = o_hat[j] + do_hat[j] * dt;
        r_hat_int[j] = r_hat[j] + dr_hat[j] * dt;
      } else {
        do_hat[j] = VectorXd::Zero(10);
        dr_hat[j] = Vector3d::Zero();
        o_hat_int[j] = o_hat[j];
        r_hat_int[j] = r_hat[j];
      }
    }

    // Second Step of Heuns Method
    Matrix4d H_ref_int = expSE3(dt * SmapSE3(psi)) * H; // TODO: CHANGED --- H_ref -> H_real
    Matrix4d H_real_int = expSE3(dt * SmapSE3(xi)) * H;
    Matrix3d R_d_int = H_ref_int.block(0, 0, 3, 3);
    Vector3d p_d_int = H_ref_int.block(0, 3, 3, 1);
    Matrix3d R_int = H_real_int.block(0, 0, 3, 3);
    Vector3d p_int = H_real_int.block(0, 3, 3, 1);
    VectorXd xi_int = xi + ddq * dt;

    VectorXd psi_int = vectorfield_SE3(H_real_int, curve, kt1, kt2, kt3, kn1,
                                       kn2, curve_derivative, delta, ds).twist.cast<double>();
    VectorXd psi_int_next =
        vectorfield_SE3(expSE3(dt * SmapSE3(psi_int)) * H_real_int, curve, kt1,
                        kt2, kt3, kn1, kn2, curve_derivative, delta, ds).twist.cast<double>();
    VectorXd psi_int_prev =
        vectorfield_SE3(expSE3(-dt * SmapSE3(psi_int)) * H_real_int, curve, kt1,
                        kt2, kt3, kn1, kn2, curve_derivative, delta, ds).twist.cast<double>();
    VectorXd psi_int_dot = (psi_int_next - psi_int_prev) / (2 * dt);

    auto [ddq_int, do_hat_int, dr_hat_int, dummy] = controller.adaptiveDynamics(
        p_int, R_int, xi_int, psi_int, psi_int_dot, o_hat_int, r_hat_int);

    for (int j = 0; j < N; ++j) {
      if (s.norm() > deadband) {
        o_hat[j] += 0.5 * (do_hat[j] + do_hat_int[j]) * dt;
        r_hat[j] += 0.5 * (dr_hat[j] + dr_hat_int[j]) * dt;
      } else {
        do_hat_int[j] = VectorXd::Zero(10);
        dr_hat_int[j] = Vector3d::Zero();
      }
    }

    H = expSE3(0.5 * dt * SmapSE3(xi + xi_int)) * H; // TODO: CHANGED EXP
    xi += 0.5 * dt * (ddq + ddq_int);

    // Print the norm of the error between r_hat and r_i, and a_hat and a_i
    VectorXd error_a = VectorXd::Zero(N);
    VectorXd error_r = VectorXd::Zero(N);
    VectorXd norm_da = VectorXd::Zero(N);
    VectorXd norm_dr = VectorXd::Zero(N);
    for (int j = 0; j < N; ++j) {
      error_a(j) = (o_hat[j] - controller.o_i).norm();
      error_r(j) = (r_hat[j] - controller.r_i[j]).norm();
      norm_da(j) = do_hat[j].norm();
      norm_dr(j) = dr_hat[j].norm();
    }
    // std::cout << "Error a norm: " << error_a.norm()
    //           << ".  Error r norm: " << error_r.norm()
    //           << ".  Norm da: " << norm_da.norm()
    //           << ".  Norm dr: " << norm_dr.norm() << std::endl;
    //
    // Log data
    H_hist.push_back(H);
    xi_hist.push_back(xi);
    xi_dot_hist.push_back(ddq);
    psi_hist.push_back(psi);
    o_hat_hist.push_back(o_hat);
    r_hat_hist.push_back(r_hat);
    closest_indexes.push_back(nearest_index);
    min_distances.push_back(dist);
    zeta_hist.push_back(s.norm());
    // H_ref = (dt * S(psi)).exp() * H_ref;
    // H_real = (dt * S(dq)).exp() * H_real;
    // dq += ddq * dt;
    // H_ref = H; // TODO: MAJOR CHANGE -- H_ref -> H_real (mirror an MPC
                    // behavior)
    // R_d = H_ref.block(0, 0, 3, 3);
    // x_d = H_ref.block(3, 6, 3, 1);
    // R = H_real.block(0, 0, 3, 3);
    // x = H_real.block(3, 6, 3, 1);
    // dx = dq.head(3);
    // w = dq.tail(3);
    // H_hist.push_back(H_real);
    // dq_hist.push_back(dq);
    // norm_s_hist.push_back(s.norm());
  }

  // saveToCSV("cpp_adaptive.csv", NEAREST_POINTS, H_hist, XI_T_HIST,
      //           dq_hist, norm_s_hist, DISTANCES, DISTANCES_APPROX);

}
