#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <iostream>
#include "cruzancona.hpp"

using namespace Eigen;
using namespace boost::numeric::odeint;

// Define the global history data structure
HistoryData globalHistory;

/**
 * @brief Constructs a System object with the given parameters.
 *
 * @param l1_ The length of link 1.
 * @param l2_ The length of link 2.
 * @param lc1_ The distance from the origin to the center of mass of link 1.
 * @param lc2_ The distance from the origin to the center of mass of link 2.
 * @param m1_ The mass of link 1.
 * @param m2_ The mass of link 2.
 * @param m1_bar_ The mass uncertainty of link 1.
 * @param m2_bar_ The mass uncertainty of link 2.
 * @param g__ The acceleration due to gravity.
 * @param I1_ The moment of inertia of link 1.
 * @param I2_ The moment of inertia of link 2.
 * @param epsilon_ The barrier width.
 * @param l_ The adaptive parameter \ell.
 * @param n_ The number of degrees of freedom.
 * @param varrho_ The parameter varrho.
 * @param A_ The matrix A.
 * @param B_ The matrix B.
 * @param L_ The matrix L.
 * @param Xi_ The matrix Xi.
 */
System::System(double l1_, double l2_, double lc1_, double lc2_, double m1_,
               double m2_, double m1_bar_, double m2_bar_, double g__,
               double I1_, double I2_, double epsilon_, double l_, double n_,
               double varrho_, const MatrixXd& A_, const MatrixXd& B_,
               const MatrixXd& L_, const MatrixXd& Xi_)
    : l1(l1_),
      l2(l2_),
      lc1(lc1_),
      lc2(lc2_),
      m1(m1_),
      m2(m2_),
      m1_bar(m1_bar_),
      m2_bar(m2_bar_),
      g_(g__),
      I1(I1_),
      I2(I2_),
      epsilon(epsilon_),
      varrho(varrho_),
      l(l_),
      n(n_),
      A(A_),
      B(B_),
      L(L_),
      Xi(Xi_) {
      }

void System::operator()(const Eigen::VectorXd& z, Eigen::VectorXd& dzdt,
                        double t) {
  // Call sysdiffeq() method with provided parameters
  sysdiffeq(z, dzdt, t);
}

Matrix2d System::getJ(const Vector2d& q) const {
  double J11 = m1 * lc1 * lc1 + I1 +
               m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * cos(q[1])) + I2;
  double J12 = m2 * (lc2 * lc2 + l1 * lc2 * cos(q[1])) + I2;
  double J21 = m2 * (lc2 * lc2 + l1 * lc2 * cos(q[1])) + I2;
  double J22 = m2 * lc2 * lc2 + I2;

  Matrix2d J;
  J << J11, J12, J21, J22;

  return J;
}

Matrix2d System::getDelta(const Vector2d& q) const {
  double delta1 = m1_bar * lc1 * lc1 +
                  m2_bar * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * cos(q[1]));
  double delta2 = m2_bar * (lc2 * lc2 + l1 * lc2 * cos(q[1]));
  double delta3 = m2_bar * lc2 * lc2;

  Matrix2d Delta;
  Delta << delta1, delta2, delta2, delta3;

  return Delta;
}

Matrix2d System::getC(const Vector2d& qdot) const {
  double h = m2 * l1 * lc2 * sin(qdot[1]);

  Matrix2d C;
  C << -h * qdot[1], -h * (qdot[0] + qdot[1]), h * qdot[0], 0;

  return C;
}

Vector2d System::getG(const Vector2d& q) const {
  double g1 = m1 * lc1 * g_ * sin(q[0]) +
              m2 * g_ * (l1 * sin(q[0]) + lc2 * sin(q[0] + q[1]));
  double g2 = m2 * lc2 * g_ * sin(q[0] + q[1]);

  Vector2d g;
  g << g1, g2;

  return g;
}

void System::setP(const MatrixXd& P_) {
  // Expects P as the solution of the Algebraic Riccati Equation
  P = P_;
  // K = (varrho * MatrixXd::Identity(B.cols(), B.cols())).inverse() *
  // B.transpose() * P;
}

void System::sysdiffeq(const VectorXd& z, VectorXd& dzdt, const double t) {
    int i = static_cast<int>(t / 1e-6); // Current iteration index
    int imax = static_cast<int>(10 / 1e-6); 
    progressBar(i, imax);
    VectorXd q_d, qdot_d, qddot_d;
    std::tie(q_d, qdot_d, qddot_d) = qqdotqddot(t);

    // std::cout << "z: " << z << std::endl;
    for (int i = 0; i < z.size(); ++i) {
        if (std::isnan(z(i))) {
            throw std::runtime_error("Vector contains NaN values!");
        }
    }
    VectorXd x = z.segment(0, 4).eval();
    VectorXd b = z.segment(4, 3).eval();
    double rho = z(7);
    VectorXd x1 = x.segment(0, 2).eval();
    VectorXd q = x1 + q_d;
    VectorXd x2 = x.segment(2, 2).eval();
    VectorXd qdot = x2 + qdot_d;

    Matrix2d J = getJ(q);
    Matrix2d Delta = getDelta(q);
    Matrix2d C = getC(qdot);
    Vector2d g = getG(q);

    Matrix2d Jtilde = J + Delta;
    Matrix2d G = J.inverse();
    Matrix2d DeltaG = J * (Jtilde.inverse()) - Matrix2d::Identity(n, n);
    Vector2d phi;
    phi.setOnes();
    phi *= eta(t);
    Vector2d h = G * ((Matrix2d::Identity(n, n) + DeltaG) *
                      (-(C * x2) - g + phi - (Jtilde * qddot_d)));

    VectorXd test = P * x;
    VectorXd w = (B.transpose() * P) * x;
    VectorXd psi = -varrho * w;
    MatrixXd B_bar = B.transpose() * P * B * G;
    VectorXd w_bar = B_bar.transpose() * w;
    double w_norm = w.norm();
    double w_bar_norm = w_bar.norm();
    VectorXd kappa(3);
    kappa << 1, x.norm(), x.transpose() * x;
    double Gamma_bar = kappa.transpose() * b;
    bool psbf_active = false;
    double H = 0.01;
    double S = 1 / (exp(-(w_norm - epsilon / 2) / H) + 1);
    double k;
    // double k = S * w_bar_norm + Gamma_bar + rho / w_bar_norm +
    //            (1 - S) * psbf(w, epsilon);
    if (w_norm > epsilon / 2){
      k = w_bar_norm + Gamma_bar + rho / w_bar_norm;
    }
    else {
      k = psbf(w, epsilon);
    }
    VectorXd v = -k * w_bar / w_bar_norm;
    VectorXd tau(4);
    tau = psi + v;

    VectorXd xdot =
        A * x + B * (G * ((Matrix2d::Identity(n, n) + DeltaG) * tau + h));
    VectorXd bdot = L * (kappa * w_bar_norm - Xi * b);
    double rhodot = l - rho;
    VectorXd zdot(8);
    zdot << xdot, bdot, rhodot;

    // Update history
    const double saveInterval = 0.01;
    const double tolerance = 1e-6;
    if (std::abs(t / saveInterval - std::round(t / saveInterval)) < tolerance) {
      updateHistory(z, t, tau, w, q, qdot);
    }

    dzdt = zdot;
  }

void System::updateHistory(const Eigen::VectorXd& z, const double t, const VectorXd& tau, const VectorXd& w, const VectorXd& q, const VectorXd& qdot){
  globalHistory.timeHistory.push_back(t);
  globalHistory.zHistory.push_back(z);
  globalHistory.tauHistory.push_back(tau);
  globalHistory.wHistory.push_back(w);
  globalHistory.qHistory.push_back(q);
  globalHistory.qdotHistory.push_back(qdot);
}

void System::integrateSystem(VectorXd& z0, double t0, double t_end, double dt) {
  // Reset history vectors
  globalHistory.timeHistory.clear();
  globalHistory.tauHistory.clear();
  globalHistory.wHistory.clear();
  globalHistory.zHistory.clear();
  globalHistory.qHistory.clear();
  globalHistory.qdotHistory.clear();

  // Define your stepper
  auto stepper = runge_kutta4<VectorXd>();

  // Integrate the system
  integrate_const(stepper, *this, z0, t0, t_end, dt);
}

void progressBar(int i, int imax) {
  std::cout << "\r[";
  int progress = static_cast<int>(20 * i / (imax - 1));
  for (int j = 0; j < 20; ++j) {
    if (j < progress) {
      std::cout << "=";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "] " << std::setw(3) << std::fixed << std::setprecision(0)
            << (100.0 * i / (imax - 1)) << "%";
  std::cout.flush();
}

double psbf(const Vector2d& z, double epsilon) {
  return z.norm() / (epsilon - z.norm());
}

std::tuple<Vector2d, Vector2d, Vector2d> qqdotqddot(double t) {
  Vector2d q_d, qdot_d, qddot_d;
  q_d << std::sin(t), std::cos(t);
  qdot_d << std::cos(t), -std::sin(t);
  qddot_d << -std::sin(t), -std::cos(t);
  return std::make_tuple(q_d, qdot_d, qddot_d);
}

double eta(double t) {
  if (t < 4 * M_PI) {
    return 2 * std::sin(4 * t);
  } else if (t >= 4 * M_PI && t < 8 * M_PI) {
    return 5 * std::sin(4 * t);
  } else {
    return 0.5 * std::sin(4 * t);
  }
}
