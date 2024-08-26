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

/**
 * @brief Class representing a system.
 * 
 * This class represents a system with various parameters and matrices.
 */
class System {
 private:
    double l1, l2, lc1, lc2, m1, m2, m1_bar, m2_bar, g_, I1, I2, epsilon, l, n,
            varrho;
    Eigen::MatrixXd A, B, L, Xi, P, K;

 public:
    /**
     * @brief Constructor for the System class.
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
    System(double l1_, double l2_, double lc1_, double lc2_, double m1_,
                 double m2_, double m1_bar_, double m2_bar_, double g__, double I1_,
                 double I2_, double epsilon_, double l_, double n_, double varrho_,
                 const MatrixXd& A_, const MatrixXd& B_, const MatrixXd& L_,
                 const MatrixXd& Xi_);

    /**
     * @brief Function call operator.
     * 
     * This operator calculates the derivative of the state vector.
     * 
     * @param z The state vector.
     * @param dzdt The derivative of the state vector.
     * @param t The current time.
     */
    void operator()(const Eigen::VectorXd& z, Eigen::VectorXd& dzdt, double t);

    /**
     * @brief Calculates the Inertia matrix.
     * 
     * This function calculates the Inertia matrix based on the given joint angles.
     * 
     * @param q The joint angles.
     * @return The Inertia matrix.
     */
    Matrix2d getJ(const Vector2d& q) const;

    /**
     * @brief Calculates the delta matrix.
     * 
     * This function calculates the mass uncertainty matrix based on the given joint angles.
     * 
     * @param q The joint angles.
     * @return The delta matrix.
     */
    Matrix2d getDelta(const Vector2d& q) const;

    /**
     * @brief Calculates the Coriolis matrix.
     * 
     * This function calculates the Coriolis matrix based on the given joint velocities.
     * 
     * @param qdot The joint velocities.
     * @return The Coriolis matrix.
     */
    Matrix2d getC(const Vector2d& qdot) const;

    /**
     * @brief Calculates the gravity vector.
     * 
     * This function calculates the gravity vector based on the given joint angles.
     * 
     * @param q The joint angles.
     * @return The gravity vector.
     */
    Vector2d getG(const Vector2d& q) const;

    /**
     * @brief Sets the P matrix.
     * 
     * This function sets the P matrix. P is positive definite and symmetric. Such that V = z^T P z.
     * 
     * @param P_ The new P matrix.
     */
    void setP(const MatrixXd& P_);

    /**
     * @brief Solves the system's differential equation.
     * 
     * This function solves the system's differential equation.
     * 
     * @param z The state vector.
     * @param dzdt The derivative of the state vector.
     * @param t The current time.
     */
    void sysdiffeq(const VectorXd& z, VectorXd& dzdt, const double t);

    /**
     * @brief Integrates the system over a given time interval.
     * 
     * This function integrates the system over a given time interval using a specified time step.
     * 
     * @param z0 The initial state vector.
     * @param t0 The initial time.
     * @param t_end The end time.
     * @param dt The time step.
     */
    void integrateSystem(VectorXd& z0, double t0, double t_end, double dt);

    /**
     * @brief Updates the system's history.
     * 
     * This function updates the system's history with the current state, time, and control inputs.
     * 
     * @param z The current state vector.
     * @param t The current time.
     * @param tau The control inputs.
     * @param w The sliding variable.
     * @param q The joint angles.
     * @param qdot The joint velocities.
     */
    void updateHistory(const Eigen::VectorXd& z, const double t, const VectorXd& tau, const VectorXd& w, const VectorXd& q, const VectorXd& qdot);
};

// Declare the global history data structure
extern HistoryData globalHistory;

#endif