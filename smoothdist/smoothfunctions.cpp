#include "smoothfunctions.hpp"
#include <cassert>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/squared_distance_3.h>

// This is uaibot header file
#include "declarations.h"

using namespace std;

// CGAL
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Line_2 Line_2;
typedef Kernel::Plane_3 Plane_3;

// ----------------------------------------------------------------------------------------
// Smooth Min / Max functions
// ----------------------------------------------------------------------------------------

float holderMean(float x, float y, float r) {
  // Eigen::VectorXf powered = values.array().pow(-1.0f / r);
  // Stabler version:
  // If any value is zero, return 0
  if (x == 0.0f || y == 0.0f) {
    return 0.0f;
  }
  // Compute true minimum and 'normalize' values
  float minValue = std::min(x, y);
  float xNorm = x / minValue;
  float yNorm = y / minValue;
  float sumPowered = pow(xNorm, -1.0f / r) + pow(yNorm, -1.0f / r);
  return minValue * pow(sumPowered, -r);
}

Eigen::VectorXf holderMeanGradient(float x, float y, float r) {
  float eps = 1e-6f;
  Eigen::VectorXf gradient(2);
  float dfdx;
  float dfdy;
  // Define cases x=0 and y>0, x>0 and y=0, x=y, and general case
  if (x == 0.0f && y > 0.0f) {
    dfdx = 1.0f;
    dfdy = 0.0f;
  } else if (x > 0.0f && y == 0.0f) {
    dfdx = 0.0f;
    dfdy = 1.0f;
  } else if (abs(x - y) < eps) {
    dfdx = pow(2.0f, -r - 1.0f);
    dfdy = pow(2.0f, -r - 1.0f);
  } else {
    // General case
    dfdx = pow((1.0f + pow(x / y, 1.0f / r)), -r - 1.0f);
    dfdy = pow((1.0f + pow(y / x, 1.0f / r)), -r - 1.0f);
  }
  gradient << dfdx, dfdy;
  // Check if any value in gradient is NaN
  for (Eigen::Index i = 0; i < gradient.size(); ++i) {
    if (isnan(gradient(i))) {
      std::cout << "gradient: " << gradient.transpose() << std::endl;
      // std::cout << "values: " << values.transpose() << std::endl;
      // std::cout << "raised: " << raised.transpose() << std::endl;
      // std::cout << "sumRaised: " << sumRaised << std::endl;
      // std::cout << "outerDer: " << outerDer << std::endl;
      // std::cout << "innerDer: " << innerDer.transpose() << std::endl;
      throw runtime_error("Gradient contains NaN values");
    }
  }
  return gradient;
}

tuple<float, Eigen::VectorXf> holderMeanWithGradient(float x, float y,
                                                     float r) {
  float mean = holderMean(x, y, r);
  Eigen::VectorXf gradient = holderMeanGradient(x, y, r);
  return make_tuple(mean, gradient);
}

// Min
float smoothMin2Elements(float x, float y, float r) {
  if (x >= 0.0f && y >= 0.0f) {
    return holderMean(x, y, r);
  } else if (x < 0.0f && y < 0.0f) {
    float xbar = -1.0f / x;
    float ybar = -1.0f / y;
    float res = holderMean(xbar, ybar, r);
    return -1.0f / res;
  } else {
    return std::min(x, y);
  }
}

Eigen::VectorXf smoothMin2ElementsGradient(float x, float y, float r) {
  if (x >= 0.0f && y >= 0.0f) {
    return holderMeanGradient(x, y, r);
  } else if (x < 0.0f && y < 0.0f) {
    float xbar = -1.0f / x;
    float ybar = -1.0f / y;
    tuple<float, Eigen::VectorXf> res = holderMeanWithGradient(xbar, ybar, r);
    float value = get<0>(res);
    Eigen::VectorXf grad = get<1>(res);
    Eigen::VectorXf chain(2);
    // Avoid near-zero division by adding small epsilon
    float eps = 1e-6f;
    chain << 1.0f / (x * x + eps), 1.0f / (y * y + eps);
    // Apply chain rule d(-1/f(-1/x, -1/y))/dx = ( -1 / f^2 ) * d(-1/x) * df/df
    grad = (grad / (value * value)).cwiseProduct(chain);
    // Check if any value in grad is NaN
    for (Eigen::Index i = 0; i < grad.size(); ++i) {
      if (isnan(grad(i))) {
        std::cout << "values: " << x << ", " << y << std::endl;
        std::cout << "min: " << value << std::endl;
        std::cout << "holder grad: " << get<1>(res).transpose() << std::endl;
        std::cout << "chain: " << chain.transpose() << std::endl;
        std::cout << "grad: " << grad.transpose() << std::endl;
        throw runtime_error("Gradient contains NaN values at pos: " +
                            to_string(i));
      }
    }
    return grad;
  } else {
    Eigen::VectorXf gradient(2);
    if (x < y) {
      gradient << 1.0, 0.0;
    } else {
      gradient << 0.0, 1.0;
    }
    return gradient;
  }
}

tuple<float, Eigen::VectorXf> smoothMin2ElementsWithGradient(float x, float y,
                                                             float r) {
  float value = smoothMin2Elements(x, y, r);
  Eigen::VectorXf gradient = smoothMin2ElementsGradient(x, y, r);
  return make_tuple(value, gradient);
}

float smoothMinList(const Eigen::VectorXf &values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    return values[0];
  }
  float minValue = values[0];
  for (Eigen::Index i = 1; i < values.size(); ++i) {
    minValue = smoothMin2Elements(minValue, values[i], r);
  }
  return minValue;
}

Eigen::VectorXf smoothMinListGradient(const Eigen::VectorXf &values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    Eigen::VectorXf gradient(1);
    gradient << 1.0f;
    return gradient;
  }

  size_t n = values.size();
  Eigen::VectorXf gradient = Eigen::VectorXf::Ones(n);
  float minValue = values[n - 1];

  for (int i = n - 2; i >= 0; --i) {
    tuple<float, Eigen::VectorXf> res =
        smoothMin2ElementsWithGradient(values[i], minValue, r);
    minValue = get<0>(res);
    Eigen::VectorXf localGrad = get<1>(res);
    float left = localGrad(0);
    float right = localGrad(1);
    gradient.segment(i + 1, n - i - 1) *= right;
    gradient(i) *= left;
  }
  return gradient;
}

tuple<float, Eigen::VectorXf>
smoothMinListWithGradient(const Eigen::VectorXf &values, float r) {
  float value = smoothMinList(values, r);
  Eigen::VectorXf gradient = smoothMinListGradient(values, r);
  return make_tuple(value, gradient);
}

// Max
float smoothMax2Elements(float x, float y, float r) {
  return -smoothMin2Elements(-x, -y, r);
}

Eigen::VectorXf smoothMax2ElementsGradient(float x, float y, float r) {
  return smoothMin2ElementsGradient(-x, -y, r);
}

tuple<float, Eigen::VectorXf> smoothMax2ElementsWithGradient(float x, float y,
                                                             float r) {
  float value = smoothMax2Elements(x, y, r);
  Eigen::VectorXf gradient = smoothMax2ElementsGradient(x, y, r);
  return make_tuple(value, gradient);
}

float smoothMaxList(const Eigen::VectorXf &values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    return values[0];
  }
  float maxValue = -smoothMinList(-values, r);
  return maxValue;
}

Eigen::VectorXf smoothMaxListGradient(const Eigen::VectorXf &values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    Eigen::VectorXf gradient(1);
    gradient << 1.0f;
    return gradient;
  }
  Eigen::VectorXf gradient = smoothMinListGradient(-values, r);
  return gradient;
}

tuple<float, Eigen::VectorXf>
smoothMaxListWithGradient(const Eigen::VectorXf &values, float r) {
  float value = smoothMaxList(values, r);
  Eigen::VectorXf gradient = smoothMaxListGradient(values, r);
  return make_tuple(value, gradient);
}

// ----------------------------------------------------------------------------------------
// // Distance related smooth functions
// ----------------------------------------------------------------------------------------
// //
float phi(float s, float eps) {
  if (s < 0.0f) {
    return 0.0f;
  } else {
    return (s * s * s) / (2.0f * (s + eps));
  }
}

float phiGradient(float s, float eps) {
  if (s < 0.0f) {
    return 0.0f;
  } else {
    return ((s * s) * (2.0f * s + 3.0f * eps)) / (2.0f * (s + eps) * (s + eps));
  }
}

tuple<float, float> phiWithGradient(float s, float eps) {
  float value = phi(s, eps);
  float gradient = phiGradient(s, eps);
  return make_tuple(value, gradient);
}

tuple<float, Eigen::VectorXf>
signedDist2Convex(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
                  const Eigen::VectorXf &b, float r, float eps, string test) {
  int N = A.rows();
  int m = A.cols();
  Eigen::VectorXf rawInnerDistances(N);
  Eigen::VectorXf rawOuterDistances(N);
  Eigen::MatrixXf rawInnerGradients(N, m);
  Eigen::MatrixXf rawOuterGradients(N, m);

  for (int i = 0; i < N; ++i) {
    Eigen::VectorXf ai = A.row(i).transpose();
    float s = b(i) - ai.dot(p);
    float s_out = -s;
    rawInnerDistances(i) = phi(s, eps);
    rawOuterDistances(i) = phi(s_out, eps);
    rawInnerGradients.row(i) = -phiGradient(s, eps) * ai.transpose();
    rawOuterGradients.row(i) = phiGradient(s_out, eps) * ai.transpose();
  }

  if (test == "in") {
    tuple<float, Eigen::VectorXf> res =
        smoothMinListWithGradient(rawInnerDistances, r);
    float dist = -get<0>(res);
    Eigen::VectorXf grad = -get<1>(res).transpose() * rawInnerGradients;
    return make_tuple(dist, grad);
  } else if (test == "out") {
    tuple<float, Eigen::VectorXf> res =
        smoothMaxListWithGradient(rawOuterDistances, r);
    float dist = get<0>(res);
    Eigen::VectorXf grad = get<1>(res).transpose() * rawOuterGradients;
    return make_tuple(dist, grad);
  } else {
    tuple<float, Eigen::VectorXf> res_min =
        smoothMinListWithGradient(rawInnerDistances, r);
    tuple<float, Eigen::VectorXf> res_max =
        smoothMaxListWithGradient(rawOuterDistances, r);
    float dist = -get<0>(res_min) + get<0>(res_max);
    Eigen::VectorXf grad_in = -get<1>(res_min).transpose() * rawInnerGradients;
    Eigen::VectorXf grad_out = get<1>(res_max).transpose() * rawOuterGradients;
    Eigen::VectorXf grad = grad_in + grad_out;
    // Check if any value in grad is NaN and print every gradient:
    for (Eigen::Index i = 0; i < grad.size(); ++i) {
      if (isnan(grad(i))) {
        cout << "Gradient values:" << endl;
        cout << grad << endl;
        cout << "grad_in:" << endl;
        cout << grad_in << endl;
        cout << "grad_out:" << endl;
        cout << grad_out << endl;
        throw runtime_error("Gradient contains NaN values");
      }
    }
    return make_tuple(dist, grad);
  }
}

// ----------------------------------------------------------------------------------------
// Euclidean Sign Distance Function (ESDF) -- CGAL + Uaibot Implementation
// ----------------------------------------------------------------------------------------

struct Polytope2D {
  Eigen::MatrixXf A;         // Halfspace normals
  Eigen::VectorXf b;         // Halfspace offsets
  Eigen::VectorXf inv_norms; // 1 / ||a_i|| for each halfspace
};

// CGAL convertions
Point_2 toCGALPoint2D(const Eigen::VectorXf &p) {
  assert(p.size() == 2);
  return Point_2(p(0), p(1));
}

Point_3 toCGALPoint3D(const Eigen::VectorXf &p) {
  assert(p.size() == 3);
  return Point_3(p(0), p(1), p(2));
}

std::vector<Line_2> toCGALLines2D(const Eigen::MatrixXf &A,
                                  const Eigen::VectorXf &b) {
  // Convert halfspace representation to CGAL lines
  // a₁x + a₂y ≤ b  →  a₁x + a₂y - b = 0
  assert(A.cols() == 2);
  assert(A.rows() == b.size());

  std::vector<Line_2> lines;
  lines.reserve(A.rows());

  for (int i = 0; i < A.rows(); ++i) {
    lines.emplace_back(A(i, 0), A(i, 1), -b(i));
  }
  return lines;
}

std::vector<Plane_3> toCGALPlanes3D(const Eigen::MatrixXf &A,
                                    const Eigen::VectorXf &b) {
  // Convert halfspace representation to CGAL planes
  // aᵀx ≤ b → aᵀx - b = 0
  assert(A.cols() == 3);
  assert(A.rows() == b.size());

  std::vector<Plane_3> planes;
  planes.reserve(A.rows());

  for (int i = 0; i < A.rows(); ++i) {
    planes.emplace_back(A(i, 0), A(i, 1), A(i, 2), -b(i));
  }
  return planes;
}

inline bool isInsidePolytope(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
                             const Eigen::VectorXf &b) {
  for (int i = 0; i < A.rows(); ++i) {
    if (A.row(i).dot(p) > b(i)) {
      return false;
    }
  }
  return true;
}

tuple<float, Eigen::VectorXf, Eigen::VectorXf>
ESDF2D(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
       const Eigen::VectorXf &b) {
  // Check if p is inside the polytope
  bool inside = isInsidePolytope(p, A, b);
  // float eps = 1e-6f;
  float distance = std::numeric_limits<float>::max();
  Eigen::VectorXf closest_point(2);
  Eigen::VectorXf grad(2);
  if (inside) {
    // find h = max_i (a_i^T p - b_i) / ||a_i||
    float max_dist = -std::numeric_limits<float>::max();
    for (int i = 0; i < A.rows(); ++i) {
      Eigen::VectorXf ai = A.row(i).transpose();
      float dist = (ai.dot(p) - b(i)) / ai.norm();
      // min_dist in this case is negative, so we get the maximum (negative)
      // distance
      if (dist > max_dist) {
        max_dist = dist;
        grad = ai.normalized();
        closest_point = p - dist * (grad);
      }
      distance = max_dist;
    }
  } else {
    // We solve QP using uaibot solveQP function to find the closest point
    // min ||x - p||^2 st. A x <= b
    // => min x^T x - 2 p^T x + p^T p st. A x <= b
    // => min 1/2 x^T (2 I) x + (-2 p)^T x + p^T p st. A x <= b
    // => H = 2 I, f = -2 p
    // solveQP uses pattern x^T H x + f^T x st. A >= b
    Eigen::MatrixXf H = 2.0 * Eigen::MatrixXf::Identity(2, 2);
    Eigen::VectorXf f = -2.0 * p;
    Eigen::MatrixXf A_ineq = -A;
    Eigen::VectorXf b_ineq = -b;
    closest_point = solveQP(H, f, A_ineq, b_ineq);
    distance = (p - closest_point).norm();
    grad = (p - closest_point).normalized();
  }
  return make_tuple(distance, grad, closest_point);
}

tuple<float, Eigen::VectorXf, Eigen::VectorXf>
ESDF3D(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
       const Eigen::VectorXf &b) {
  // Check if p is inside the polytope
  bool inside = isInsidePolytope(p, A, b);
  // float eps = 1e-6f;
  float distance = std::numeric_limits<float>::max();
  Eigen::VectorXf closest_point(3);
  Eigen::VectorXf grad(3);
  if (inside) {
    // find h = max_i (a_i^T p - b_i) / ||a_i||
    float max_dist = -std::numeric_limits<float>::max();
    for (int i = 0; i < A.rows(); ++i) {
      Eigen::VectorXf ai = A.row(i).transpose();
      float dist = (ai.dot(p) - b(i)) / ai.norm();
      // min_dist in this case is negative, so we get the maximum (negative)
      // distance
      if (dist > max_dist) {
        max_dist = dist;
        grad = ai.normalized();
        closest_point = p - dist * (grad);
      }
      distance = max_dist;
    }
  } else {
    // We solve QP using uaibot solveQP function to find the closest point
    // min ||x - p||^2 st. A x <= b
    // => min x^T x - 2 p^T x + p^T p st. A x <= b
    // => min 1/2 x^T (2 I) x + (-2 p)^T x + p^T p st. A x <= b
    // => H = 2 I, f = -2 p
    // solveQP uses pattern x^T H x + f^T x st. A >= b
    Eigen::MatrixXf H = 2.0 * Eigen::MatrixXf::Identity(3, 3);
    Eigen::VectorXf f = -2.0 * p;
    Eigen::MatrixXf A_ineq = -A;
    Eigen::VectorXf b_ineq = -b;
    closest_point = solveQP(H, f, A_ineq, b_ineq);
    distance = (p - closest_point).norm();
    grad = (p - closest_point).normalized();
  }
  return make_tuple(distance, grad, closest_point);
}

tuple<float, Eigen::VectorXf, Eigen::VectorXf>
ESDF2D_CGAL(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
            const Eigen::VectorXf &b) {
  // Convert point and halfspace representation to CGAL types
  Point_2 point = toCGALPoint2D(p);
  std::vector<Line_2> lines = toCGALLines2D(A, b);

  // Check if p is inside the polytope
  bool inside = isInsidePolytope(p, A, b);
  float eps = 1e-6f;
  if (inside) {
    // If inside, compute the minimum distance to each hyperplane
    // nearest point is necessarily on the boundary of the polytope
    std::vector<float> distances;
    std::vector<Eigen::VectorXf> closest_points;
    for (const auto &line : lines) {
      Point_2 closest_point = line.projection(point);
      float dist = sqrt(CGAL::squared_distance(point, closest_point));
      // Debug for NaN values
      if (isnan(dist) || isinf(dist)) {
        std::cout << "Point: (" << point.x() << ", " << point.y() << ")"
                  << std::endl;
        std::cout << "Closest Point: (" << closest_point.x() << ", "
                  << closest_point.y() << ")" << std::endl;
        std::cout << "Squared Distance: "
                  << CGAL::squared_distance(point, closest_point) << std::endl;
        std::cout << "Distance: " << dist << std::endl;
      }
      distances.push_back(dist);
      Eigen::VectorXf cp(2);
      cp << closest_point.x(), closest_point.y();
      closest_points.push_back(cp);
    }
    auto minIt = std::min_element(distances.begin(), distances.end());
    int minIndex = std::distance(distances.begin(), minIt);
    float minDist = -distances[minIndex];
    Eigen::VectorXf closest_point = closest_points[minIndex];
    // Gradient computation (Gradient is a row vector)
    Eigen::VectorXf grad = (p - closest_point).normalized();
    return make_tuple(minDist, -grad, closest_point);
  } else {
    // If outside, then another strategy is needed since hyperplanes
    // are infinite and nearest point may not lie on polytope boundary.
    // In this case we rely on Uaibot distance computation
    // First param is a 4x4 identity matrix
    // Uaibot functions are for 3d only, so we create a 3D polytope with z=0
    Eigen::MatrixXf A_aux = Eigen::MatrixXf::Zero(A.rows() + 2, 3);
    A_aux.block(0, 0, A.rows(), 2) = A;
    // Add planes for z >= 0 and z <= 0
    A_aux.row(A.rows()) << 0.0f, 0.0f, 1.0f;      // For z >= 0
    A_aux.row(A.rows() + 1) << 0.0f, 0.0f, -1.0f; // For z <= 0
    Eigen::VectorXf b_aux = Eigen::VectorXf::Zero(b.size() + 2);
    b_aux.head(b.size()) = b;
    b_aux(b.size()) = 0.0f;     // For z >= 0
    b_aux(b.size() + 1) = 0.0f; // For z <= 0
    GeometricPrimitives polytope = GeometricPrimitives::create_convexpolytope(
        Eigen::Matrix4f::Identity(), A_aux, b_aux);
    Eigen::MatrixXf htm = Eigen::MatrixXf::Identity(4, 4);
    htm.block<2, 1>(0, 3) = p;
    GeometricPrimitives point_geom =
        GeometricPrimitives::create_sphere(htm, eps);
    // This will return points at infinity in square test case !!!
    // PrimDistResult dist_res = point_geom.dist_to(polytope, 0.0f, 0.0f, 1e-3f,
    // 20); Testing for the smallest smoothing parameters that avoid infinities
    // 1e-5f works for most cases, but not all
    float test_eps = 1e-4f;
    PrimDistResult dist_res =
        point_geom.dist_to(polytope, test_eps, test_eps, 1e-3f, 20);
    // PrimDistResult dist_res = point_geom.dist_to(polytope, 0.1f, 0.05f,
    // 1e-3f, 20);
    Eigen::Vector3f closest_point = dist_res.proj_B;
    Eigen::VectorXf nearest_point_2d(2);
    nearest_point_2d << closest_point(0), closest_point(1);
    // Compute distance manually since dist_res.dist is a smoothed distance
    float distance = (p - nearest_point_2d).norm();
    Eigen::VectorXf grad = (p - nearest_point_2d).normalized();
    // For some reason this returns inf sometimes. E.g.:
    // Distance is NaN. Point is inside
    // Point p: -17 -17
    // Closest point:  6.87796 -8.06637        0
    // Grad: -0.936595 -0.350415
    // distance: inf

    if (isnan(distance) || isinf(distance)) {
      std::cout << "Distance is NaN. Point is outside" << std::endl;
      std::cout << "Point p: " << p.transpose() << std::endl;
      std::cout << "Closest point: " << closest_point.transpose() << std::endl;
      std::cout << "Nearest point 2D: " << nearest_point_2d.transpose()
                << std::endl;
      std::cout << "Grad: " << grad.transpose() << std::endl;
      std::cout << "distance: " << distance << std::endl;
      throw runtime_error("Distance is NaN");
    }
    return make_tuple(distance, grad, nearest_point_2d);
  }
}

tuple<float, Eigen::VectorXf, Eigen::VectorXf>
ESDF3D_CGAL(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
            const Eigen::VectorXf &b) {
  // Convert point and halfspace representation to CGAL types
  Point_3 point = toCGALPoint3D(p);
  std::vector<Plane_3> planes = toCGALPlanes3D(A, b);

  // Check if p is inside the polytope
  bool inside = isInsidePolytope(p, A, b);
  float eps = 1e-6f;
  if (inside) {
    // If inside, compute the minimum distance to each hyperplane
    // nearest point is necessarily on the boundary of the polytope
    std::vector<float> distances;
    std::vector<Eigen::VectorXf> closest_points;
    for (const auto &plane : planes) {
      Point_3 closest_point = plane.projection(point);
      float dist = sqrt(CGAL::squared_distance(point, closest_point));
      distances.push_back(dist);
      Eigen::VectorXf cp(3);
      cp << closest_point.x(), closest_point.y(), closest_point.z();
      closest_points.push_back(cp);
    }
    auto minIt = std::min_element(distances.begin(), distances.end());
    int minIndex = std::distance(distances.begin(), minIt);
    float minDist = -distances[minIndex];
    Eigen::VectorXf closest_point = closest_points[minIndex];
    // Gradient computation (Gradient is a row vector)
    Eigen::VectorXf grad = (p - closest_point).normalized();
    return make_tuple(minDist, -grad, closest_point);
  } else {
    // If outside, then another strategy is needed since hyperplanes
    // are infinite and nearest point may not lie on polytope boundary.
    // In this case we rely on Uaibot distance computation
    // First param is a 4x4 identity matrix
    GeometricPrimitives polytope = GeometricPrimitives::create_convexpolytope(
        Eigen::Matrix4f::Identity(), A, b);
    Eigen::MatrixXf htm = Eigen::MatrixXf::Identity(4, 4);
    htm.block<3, 1>(0, 3) = p;
    GeometricPrimitives point_geom =
        GeometricPrimitives::create_sphere(htm, eps);
    PrimDistResult dist_res =
        point_geom.dist_to(polytope, 0.0f, 0.0f, 1e-3f, 20, p * 0);
    float distance = dist_res.dist;
    Eigen::VectorXf closest_point = dist_res.proj_B;
    Eigen::VectorXf grad = (p - closest_point).normalized();
    return make_tuple(distance, grad, closest_point);
  }
}

tuple<float, Eigen::VectorXf, Eigen::VectorXf>
ESDF_CGAL(const Eigen::VectorXf &p, const Eigen::MatrixXf &A,
          const Eigen::VectorXf &b) {
  if (p.size() == 2) {
    // return ESDF2D_CGAL(p, A, b);
    return ESDF2D(p, A, b);
  } else if (p.size() == 3) {
    // return ESDF3D_CGAL(p, A, b);
    return ESDF3D(p, A, b);
  } else {
    throw invalid_argument("Point dimension must be 2 or 3 for ESDF_CGAL");
  }
}
