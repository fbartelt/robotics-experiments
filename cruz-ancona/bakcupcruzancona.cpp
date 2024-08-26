#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <iostream>

using namespace Eigen;
using namespace boost::numeric::odeint;

typedef Matrix<double, Dynamic, 1> State;

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
    return 2 * std::sin(4 * t) * 0;
  } else if (t >= 4 * M_PI && t < 8 * M_PI) {
    return 5 * std::sin(4 * t) * 0;
  } else {
    return 0.5 * std::sin(4 * t) * 0;
  }
}

class System {
 private:
  double l1, l2, lc1, lc2, m1, m2, m1_bar, m2_bar, g_, I1, I2, epsilon, l, n,
      varrho;
  MatrixXd A, B, L, Xi, P, K;
  std::vector<double> timeHistory, rhoHistory;
  std::vector<Eigen::VectorXd> zHistory, tauHistory, wHistory, bHistory;

 public:
  System(double l1_, double l2_, double lc1_, double lc2_, double m1_,
         double m2_, double m1_bar_, double m2_bar_, double g__, double I1_,
         double I2_, double epsilon_, double l_, double n_, double varrho_,
         const MatrixXd& A_, const MatrixXd& B_, const MatrixXd& L_,
         const MatrixXd& Xi_)
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
        Xi(Xi_) {}

  Matrix2d getJ(const Vector2d& q) const {
    double J11 = m1 * lc1 * lc1 + I1 +
                 m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * cos(q[1])) + I2;
    double J12 = m2 * (lc2 * lc2 + l1 * lc2 * cos(q[1])) + I2;
    double J21 = m2 * (lc2 * lc2 + l1 * lc2 * cos(q[1])) + I2;
    double J22 = m2 * lc2 * lc2 + I2;

    Matrix2d J;
    J << J11, J12, J21, J22;

    return J;
  }

  Matrix2d getDelta(const Vector2d& q) const {
    double delta1 = m1_bar * lc1 * lc1 +
                    m2_bar * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * cos(q[1]));
    double delta2 = m2_bar * (lc2 * lc2 + l1 * lc2 * cos(q[1]));
    double delta3 = m2_bar * lc2 * lc2;

    Matrix2d Delta;
    Delta << delta1, delta2, delta2, delta3;

    return Delta;
  }

  Matrix2d getC(const Vector2d& qdot) const {
    double h = m2 * l1 * lc2 * sin(qdot[1]);

    Matrix2d C;
    C << -h * qdot[1], -h * (qdot[0] + qdot[1]), h * qdot[0], 0;

    return C;
  }

  Vector2d getG(const Vector2d& q) const {
    double g1 = m1 * lc1 * g_ * sin(q[0]) +
                m2 * g_ * (l1 * sin(q[0]) + lc2 * sin(q[0] + q[1]));
    double g2 = m2 * lc2 * g_ * sin(q[0] + q[1]);

    Vector2d g;
    g << g1, g2;

    return g;
  }

  void setP(const MatrixXd& P_) {
    // Expects P as the solution of the Algebraic Riccati Equation
    P = P_;
    // K = (varrho * MatrixXd::Identity(B.cols(), B.cols())).inverse() * B.transpose() * P;
  }

  void sysdiffeq(const VectorXd& z, VectorXd& dzdt, const double t) {
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
    VectorXd w_bar = B_bar * w;
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
      k = k = w_bar_norm + Gamma_bar + rho / w_bar_norm;
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

    timeHistory.push_back(t);
    tauHistory.push_back(tau);
    wHistory.push_back(w);
    bHistory.push_back(b);
    zHistory.push_back(z);
    rhoHistory.push_back(rho);

    dzdt = zdot;
  }

  void operator()(const VectorXd& z, VectorXd& dzdt, const double t) {
    // Call sysdiffeq() method with provided parameters
    sysdiffeq(z, dzdt, t);
  }
};

struct push_back_state_and_time {
  std::vector<VectorXd>& m_states;
  std::vector<double>& m_times;

  push_back_state_and_time(std::vector<VectorXd>& states,
                           std::vector<double>& times)
      : m_states(states), m_times(times) {}

  void operator()(const VectorXd& x, double t) {
    m_states.push_back(x);
    m_times.push_back(t);
  }
};

void integrateSystem(const System& system, VectorXd& z0, double t0,
                     double t_end, double dt, std::vector<VectorXd>& states,
                     std::vector<double>& times) {
  // Reset states and times
  states.clear();
  times.clear();

  // Define your stepper
  auto stepper = runge_kutta4<VectorXd>();

  // Define the observer
  push_back_state_and_time observer(states, times);

  // Integrate the system
  integrate_const(stepper, system, z0, t0, t_end, dt, observer);
}

int main() {
  // Define system parameters
  double l1 = 0.45, l2 = 0.45;
  double lc1 = 0.091, lc2 = 0.048;
  double m1 = 23.902, m2 = 3.88;
  double g_ = 9.81;
  double I1 = 1.266, I2 = 0.093;
  double m1_bar = 0, m2_bar = 2;
  int n = 2;
  double varrho = 5.03;
  double l = 10;
  double epsilon = 0.05;
  Matrix3d L, Xi;
  L.setZero();
  Xi.setZero();
  L.diagonal() << 0.1, 0.1, 1;
  Xi.diagonal() << 0.01, 0.1, 0.01;
  std::cout << "L: " << L << std::endl;
  std::cout << "Xi: " << Xi << std::endl;
  MatrixXd A = MatrixXd::Zero(2 * n, 2 * n);
  A.block(0, n, n, n) = MatrixXd::Identity(n, n);
  MatrixXd B = MatrixXd::Zero(2 * n, n);
  B.block(n, 0, n, n) = MatrixXd::Identity(n, n);

  // Create system object
  System system(l1, l2, lc1, lc2, m1, m2, m1_bar, m2_bar, g_, I1, I2, epsilon,
                l, n, varrho, A, B, L, Xi);

  // Define initial conditions
  VectorXd x0(4), b0(3);
  x0 << 1, -1, 0, 0;
  b0 << 0.01, 0.01, 0.01;
  double rho0 = 2;
  VectorXd z(8);
  z << x0, b0, rho0;
  Eigen::MatrixXd P(4, 4);
  P << 4.08497261, 0.0, 3.17175031, 0.0,
            0.0, 4.08497261, 0.0, 3.17175031,
            3.17175031, 0.0, 6.47825656, 0.0,
            0.0, 3.17175031, 0.0, 6.47825656;
  system.setP(P);

  // Define integration parameters
  double t0 = 0.0;
  double tf = 10.0;
  double dt = 1e-4;
  // int steps = static_cast<int>((t0 - tf) / dt);
  typedef runge_kutta4<State> stepper_type;

  // Create vectors to store states and times
  std::vector<VectorXd> states;
  std::vector<double> times;

  // Integrate the system and store states and times
  integrateSystem(system, z, t0, tf, dt, states, times);
  // auto stepper = runge_kutta4<VectorXd>();
  // std::cout << "z0: " << z << std::endl;
  // std::cout << "z0size: " << z.size() << std::endl;
  // size_t steps = integrate( system , z , 0.0 , 10.0 , 0.1 );
  // integrate_const(stepper, system, z0, t0, tf, dt);

  // Process or save the history as needed
  // ...
  // for (size_t i = 0; i < times.size(); ++i) {
  //   // Output time and state vector
  //   std::cout << times[i] << "\t";
  //   for (int j = 0; j < states[i].size(); ++j) {
  //     std::cout << states[i][j] << "\t";
  //   }
  //   std::cout << std::endl;
  // }

  return 0;
}