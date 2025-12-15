#include <cmath>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "smoothfunctions.hpp"

using namespace std;

// ----------------------------------------------------------------------------------------
// Smooth Min / Max functions
// ----------------------------------------------------------------------------------------
//

float holderMean(const Eigen::VectorXf &values, float r) {
  // Eigen::VectorXf powered = values.array().pow(-1.0f / r);
  // Stabler version:
  // If any value is zero, return 0
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (values(i) == 0.0f) {
      return 0.0f;
    }
  }
  // Compute true minimum and 'normalize' values
  float minValue = values.minCoeff();
  Eigen::VectorXf normalized = values.array() / minValue;
  Eigen::VectorXf powered = normalized.array().pow(-1.0f / r);
  float sumPowered = powered.sum();
  return minValue * pow(sumPowered, -r);
}

Eigen::VectorXf holderMeanGradient(const Eigen::VectorXf &values, float r) {
  float eps = 1e-6f;
  Eigen::VectorXf raised = values.array().pow(-1.0f / r);
  float sumRaised = raised.sum();
  float outerDer = -r * pow(sumRaised, -r - 1.0f);
  Eigen::VectorXf innerDer =
      (-1.0f / r) * (values.array() + eps).pow((-1.0f / r) - 1.0f);
  // Check if any value in innerDer is infinite and replace with 1e8
  for (Eigen::Index i = 0; i < innerDer.size(); ++i) {
    if (isinf(innerDer(i))) {
      innerDer(i) = 1e8f;
    }
  }
  Eigen::VectorXf gradient = outerDer * innerDer;
  // Check if any value in gradient is NaN
  for (Eigen::Index i = 0; i < gradient.size(); ++i) {
    if (isnan(gradient(i))) {
      std::cout << "values: " << values.transpose() << std::endl;
      std::cout << "raised: " << raised.transpose() << std::endl;
      std::cout << "sumRaised: " << sumRaised << std::endl;
      std::cout << "outerDer: " << outerDer << std::endl;
      std::cout << "innerDer: " << innerDer.transpose() << std::endl;
      throw runtime_error("Gradient contains NaN values");
    }
  }
  return gradient;
}

tuple<float, Eigen::VectorXf>
holderMeanWithGradient(const Eigen::VectorXf &values, float r) {
  float mean = holderMean(values, r);
  Eigen::VectorXf gradient = holderMeanGradient(values, r);
  return make_tuple(mean, gradient);
}

// Min
float smoothMin2Elements(float x, float y, float r) {
  if (x >= 0.0f && y >= 0.0f) {
    Eigen::VectorXf values(2);
    values << x, y;
    return holderMean(values, r);
  } else if (x < 0.0f && y < 0.0f) {
    Eigen::VectorXf values(2);
    values << -1.0f / x, -1.0f / y;
    float res = holderMean(values, r);
    return -1.0f / res;
  } else {
    return std::min(x, y);
  }
}

Eigen::VectorXf smoothMin2ElementsGradient(float x, float y, float r) {
  if (x >= 0.0f && y >= 0.0f) {
    Eigen::VectorXf values(2);
    values << x, y;
    return holderMeanGradient(values, r);
  } else if (x < 0.0f && y < 0.0f) {
    Eigen::VectorXf values(2);
    values << -1.0f / x, -1.0f / y;
    tuple<float, Eigen::VectorXf> res = holderMeanWithGradient(values, r);
    float value = get<0>(res);
    Eigen::VectorXf grad = get<1>(res);
    Eigen::VectorXf chain(2);
    chain << 1.0f / (x * x), 1.0f / (y * y);
    // Apply chain rule d(-1/f(-1/x, -1/y))/dx = ( -1 / f^2 ) * d(-1/x) * df/df
    grad = (grad / (value * value)).cwiseProduct(chain);
    // Check if any value in grad is NaN
    for (Eigen::Index i = 0; i < grad.size(); ++i) {
      if (isnan(grad(i))) {
        std::cout << "values: " << values.transpose() << std::endl;
        std::cout << "min: " << value << std::endl;
        std::cout << "holder grad: " << get<1>(res).transpose() << std::endl;
        std::cout << "chain: " << chain.transpose() << std::endl;
        std::cout << "grad: " << grad.transpose() << std::endl;
        throw runtime_error("Gradient contains NaN values");
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

  std::cout << "[DEBUG] --------------------------" << std::endl;
  for (int i = n - 2; i >= 0; --i) {
    tuple<float, Eigen::VectorXf> res =
        smoothMin2ElementsWithGradient(values[i], minValue, r);
    minValue = get<0>(res);
    Eigen::VectorXf localGrad = get<1>(res);
    float left = localGrad(0);
    float right = localGrad(1);
    gradient.segment(i + 1, n - i - 1) *= right;
    gradient(i) *= left;
    std::cout << "[debug] smin_grad_loop grad: " << gradient.transpose()
              << ", left: " << left << ", right: " << right << std::endl;
    std::cout << "values: " << values[i] << ", minValue: " << minValue << std::endl;
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
                  const Eigen::VectorXf &b, float r, float eps,
                  string test) {
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
