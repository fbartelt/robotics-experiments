#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <functional>
#include <iostream>

#include "cruzancona.hpp"

using namespace Eigen;
using namespace boost::numeric::odeint;

typedef Matrix<double, Dynamic, 1> State;

void saveEigenHistoryToCSV(const std::vector<Eigen::VectorXd>& zHistory,
                           const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Write the column headers (optional)
  for (int j = 0; j < zHistory.front().size(); ++j) {
    file << "z" << j + 1;
    if (j != zHistory.front().size() - 1) {
      file << ",";
    }
  }
  file << std::endl;

  // Write each row of zHistory to the file
  for (const auto& z : zHistory) {
    for (int i = 0; i < z.size(); ++i) {
      file << z(i);
      if (i != z.size() - 1) {
        file << ",";
      }
    }
    file << std::endl;
  }

  file.close();
}

void saveTimeHistoryToCSV(const std::vector<double>& timeHistory, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write the column header (optional)
    file << "Time" << std::endl;

    // Write each value of timeHistory to the file
    for (const auto& time : timeHistory) {
        file << time << std::endl;
    }

    file.close();
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
  // double varrho = 0.503; //multiplied by 1/10
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
  // P << 4.08497261, 0.0, 3.17175031, 0.0, 0.0, 4.08497261, 0.0, 3.17175031,
  //     3.17175031, 0.0, 6.47825656, 0.0, 0.0, 3.17175031, 0.0, 6.47825656;
  // P << 2.20360132, 0.0, 0.86861959, 0.0,
  //        0.0, 2.20360132, 0.0, 0.86861959,
  //        0.86861959, 0.0, 1.27606086, 0.0,
  //        0.0, 0.86861959, 0.0, 1.27606086; // Change Q to 0.75*Q and R = 0.1varrho*I
  // P << 4.08497261, 0.        , 3.17175031, 0.,
  //       0.        , 4.08497261, 0.        , 3.17175031,
  //       3.17175031, 0.        , 6.47825656, 0.,
  //       0.        , 3.17175031, 0.        , 6.47825656;
  P << 2.5539         0    0.6306         0;
          0    2.5539         0    0.6306;
     0.6306         0    0.8052         0;
          0    0.6306         0    0.8052
  system.setP(P);

  // Define integration parameters
  double t0 = 0.0;
  double tf = 15.0;
  double dt = 1e-5;
  // int steps = static_cast<int>((t0 - tf) / dt);
  typedef runge_kutta4<State> stepper_type;

  // Create vectors to store states and times
  std::vector<VectorXd> states;
  std::vector<double> times;

  // Integrate the system and store states and times
  system.integrateSystem(z, t0, tf, dt);
  // auto stepper = runge_kutta4<VectorXd>();
  // std::cout << "z0: " << z << std::endl;
  // std::cout << "z0size: " << z.size() << std::endl;
  // size_t steps = integrate( system , z , 0.0 , 10.0 , 0.1 );
  // integrate_const(stepper, system, z0, t0, tf, dt);

  // Process or save the history as needed
  // ...
  saveTimeHistoryToCSV(globalHistory.timeHistory, "timeHistory_nominal15sx_1e6.csv");
  saveEigenHistoryToCSV(globalHistory.zHistory, "zHistory_nominal15sx_1e6.csv");
  saveEigenHistoryToCSV(globalHistory.tauHistory, "tauHistory_nominal15sx_1e6.csv");
  saveEigenHistoryToCSV(globalHistory.wHistory, "wHistory_nominal15sx_1e6.csv");
  saveEigenHistoryToCSV(globalHistory.qHistory, "qHistory_nominal15sx_1e6.csv");
  saveEigenHistoryToCSV(globalHistory.qdotHistory, "qdotHistory_nominal15sx_1e6.csv");
  return 0;
}