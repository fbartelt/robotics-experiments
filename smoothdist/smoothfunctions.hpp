#pragma once
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;

// ----------------------------------------------------------------------------------------
// // Smooth Min / Max functions
// ----------------------------------------------------------------------------------------

float holderMean(float x, float y, float r);
Eigen::VectorXf holderMeanGradient(float x, float y, float r);
tuple<float, Eigen::VectorXf>
holderMeanWithGradient(float x, float y, float r);
// Min
float smoothMin2Elements(float x, float y, float r);
Eigen::VectorXf smoothMin2ElementsGradient(float x, float y, float r);
tuple<float, Eigen::VectorXf> smoothMin2ElementsWithGradient(float x, float y,
                                                             float r);
float smoothMinList(const Eigen::VectorXf &values, float r);
Eigen::VectorXf smoothMinListGradient(const Eigen::VectorXf &values, float r);
tuple<float, Eigen::VectorXf>
smoothMinListWithGradient(const Eigen::VectorXf &values, float r);
// Max
float smoothMax2Elements(float x, float y, float r);
Eigen::VectorXf smoothMax2ElementsGradient(float x, float y, float r);
tuple<float, Eigen::VectorXf> smoothMax2ElementsWithGradient(float x, float y,
                                                             float r);
float smoothMaxList(const Eigen::VectorXf &values, float r);
Eigen::VectorXf smoothMaxListGradient(const Eigen::VectorXf &values, float r);
tuple<float, Eigen::VectorXf>
smoothMaxListWithGradient(const Eigen::VectorXf &values, float r);
// ----------------------------------------------------------------------------------------
// // Distance related smooth functions
// ----------------------------------------------------------------------------------------
float phi(float s, float eps = 0.01f);
float phiGradient(float s, float eps = 0.01f);
tuple<float, float> phiWithGradient(float s, float eps = 0.1f);
tuple<float, Eigen::VectorXf>
signedDist2Convex(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
                  const Eigen::VectorXf &b, float r = 0.1f, float eps = 0.01f,
                  string test = "");
