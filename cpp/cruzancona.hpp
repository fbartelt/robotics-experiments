#ifndef CRUZANCONA_HPP
#define CRUZANCONA_HPP

#include <eigen3/Eigen/Dense>
#include <functional>

using namespace Eigen;

void progressBar(int i, int imax);
double psbf(const Eigen::Vector2d& z, double epsilon);
std::tuple<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d> qqdotqddot(
    double t);
double eta(double t);

struct HistoryData {
    std::vector<double> timeHistory;
    std::vector<Eigen::VectorXd> zHistory;
    std::vector<Eigen::VectorXd> tauHistory;
    std::vector<Eigen::VectorXd> wHistory;
    std::vector<Eigen::VectorXd> qHistory;
    std::vector<Eigen::VectorXd> qdotHistory;
};

class System {
 private:
  double l1, l2, lc1, lc2, m1, m2, m1_bar, m2_bar, g_, I1, I2, epsilon, l, n,
      varrho;
  Eigen::MatrixXd A, B, L, Xi, P, K;

 public:
  System(double l1_, double l2_, double lc1_, double lc2_, double m1_,
         double m2_, double m1_bar_, double m2_bar_, double g__, double I1_,
         double I2_, double epsilon_, double l_, double n_, double varrho_,
         const MatrixXd& A_, const MatrixXd& B_, const MatrixXd& L_,
         const MatrixXd& Xi_);
  void operator()(const Eigen::VectorXd& z, Eigen::VectorXd& dzdt, double t);
  Matrix2d getJ(const Vector2d& q) const ;
  Matrix2d getDelta(const Vector2d& q) const;
  Matrix2d getC(const Vector2d& qdot) const;
  Vector2d getG(const Vector2d& q) const;
  void setP(const MatrixXd& P_);
  void sysdiffeq(const VectorXd& z, VectorXd& dzdt, const double t);
  void integrateSystem(VectorXd& z0, double t0, double t_end, double dt);
  void updateHistory(const Eigen::VectorXd& z, const double t, const VectorXd& tau, const VectorXd& w, const VectorXd& q, const VectorXd& qdot);
};

// Declare the global history data structure
extern HistoryData globalHistory;

#endif