#include <eigen3/Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "smoothfunctions.hpp"
#include <eigen3/Eigen/Dense>

using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(smoothfunctions, m) {
  m.doc() = "Smooth functions module";
  m.def("holderMean", &holderMean, "Compute the Holder mean of a list of values",
        py::arg("values"), py::arg("r"));
  m.def("holderMeanGradient", &holderMeanGradient,
        "Compute the gradient of the Holder mean", py::arg("values"), py::arg("r"));
  m.def("holderMeanWithGradient", &holderMeanWithGradient,
        "Compute the Holder mean and its gradient", py::arg("values"), py::arg("r"));
  m.def("smoothMin2Elements", &smoothMin2Elements,
        "Compute the smooth minimum of two elements", py::arg("x"), py::arg("y"),
        py::arg("r"));
  m.def("smoothMin2ElementsGradient", &smoothMin2ElementsGradient,
        "Compute the gradient of the smooth minimum of two elements", py::arg("x"),
        py::arg("y"), py::arg("r"));
  m.def("smoothMin2ElementsWithGradient", &smoothMin2ElementsWithGradient,
        "Compute the smooth minimum of two elements and its gradient", py::arg("x"),
        py::arg("y"), py::arg("r"));
  m.def("smoothMinList", &smoothMinList,
        "Compute the smooth minimum of a list of values", py::arg("values"), py::arg("r"));
  m.def("smoothMinListGradient", &smoothMinListGradient,
        "Compute the gradient of the smooth minimum of a list of values", py::arg("values"),
        py::arg("r"));
  m.def("smoothMinListWithGradient", &smoothMinListWithGradient,
        "Compute the smooth minimum of a list of values and its gradient", py::arg("values"),
        py::arg("r"));
  m.def("smoothMax2Elements", &smoothMax2Elements,
        "Compute the smooth maximum of two elements", py::arg("x"), py::arg("y"),
        py::arg("r"));
  m.def("smoothMax2ElementsGradient", &smoothMax2ElementsGradient,
        "Compute the gradient of the smooth maximum of two elements", py::arg("x"),
        py::arg("y"), py::arg("r"));
  m.def("smoothMax2ElementsWithGradient", &smoothMax2ElementsWithGradient,
        "Compute the smooth maximum of two elements and its gradient", py::arg("x"),
        py::arg("y"), py::arg("r"));
  m.def("smoothMaxList", &smoothMaxList,
        "Compute the smooth maximum of a list of values", py::arg("values"), py::arg("r"));
  m.def("smoothMaxListGradient", &smoothMaxListGradient,
        "Compute the gradient of the smooth maximum of a list of values", py::arg("values"),
        py::arg("r"));
  m.def("smoothMaxListWithGradient", &smoothMaxListWithGradient,
        "Compute the smooth maximum of a list of values and its gradient", py::arg("values"),
        py::arg("r"));
  m.def("phi", &phi, "Compute the smooth function phi", py::arg("s"), py::arg("eps") = 0.01f);
  m.def("phiGradient", &phiGradient,
        "Compute the gradient of the smooth function phi", py::arg("s"), py::arg("eps") = 0.01f);
  m.def("phiWithGradient", &phiWithGradient,
        "Compute the smooth function phi and its gradient", py::arg("s"), py::arg("eps") = 0.1f);
  m.def("signedDist2Convex", &signedDist2Convex,
        "Compute the signed distance to a convex shape defined by linear inequalities",
        py::arg("p"), py::arg("A"), py::arg("b"), py::arg("r") = 0.1f,
        py::arg("eps") = 0.01f, py::arg("test") = "");
}

