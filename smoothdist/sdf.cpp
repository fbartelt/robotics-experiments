#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <memory>
#include <stdexcept>

// FCL includes
#include <fcl/fcl.h>
// Or include individual shapes:
// #include <fcl/geometry/shape/sphere.h>
// #include <fcl/geometry/shape/box.h>
// ...

#include <vector>
// uAIBot GeometricPrimitive header – adapt the path to your actual file
#include "declarations.h"  // Contains GeometricPrimitive class

namespace py = pybind11;

static std::shared_ptr<fcl::CollisionGeometryd> makeFCLGeometry(
    const GeometricPrimitives& prim,
    const Eigen::MatrixXf& vertices = Eigen::MatrixXf(),
    const std::vector<int>& faces = std::vector<int>()) {
  int type = prim.type;

  switch (type) {
    case 0: {  // SPHERE
      float radius = prim.lx;
      return std::make_shared<fcl::Sphered>(radius);
    }
    case 1: {  // BOX
      float lx = prim.lx;
      float ly = prim.ly;
      float lz = prim.lz;
      return std::make_shared<fcl::Boxd>(lx, ly, lz);
    }
    case 2: {  // CYLINDER
      float radius = prim.lx;
      float length = prim.lz;
      return std::make_shared<fcl::Cylinderd>(radius, length);
    }
    case 3: {  // POINT CLOUD
      throw std::runtime_error(
          "Point clouds are not directly supported in FCL. Consider using a "
          "different representation or a custom distance function.");
    }
    case 4: {  // CONVEX POLYTOPE
      if (vertices.rows() == 0 || faces.empty())
        throw std::runtime_error(
            "Polytope requires pre‑computed vertices (Nx3) and faces (triangle "
            "indices).");
      // std::cout << "[DEBUG makeFCLGeometry] Polytope with " << vertices.rows()
      //           << " vertices, " << faces.size() << " face indices ("
      //           << faces.size() / 3 << " triangles)" << std::endl;

      // Check face index bounds
      // if (!faces.empty()) {
      //   int max_idx = *std::max_element(faces.begin(), faces.end());
      //   int min_idx = *std::min_element(faces.begin(), faces.end());
      //   std::cout << "[DEBUG] Face index range: [" << min_idx << ", " << max_idx
      //             << "]"
      //             << " (vertex count: " << vertices.rows() << ")" << std::endl;
      // }
      Eigen::MatrixXd vertices_d = vertices.cast<double>();
      auto verts_vec = std::make_shared<std::vector<fcl::Vector3d>>();
      verts_vec->reserve(vertices_d.rows());
      for (int i = 0; i < vertices_d.rows(); ++i)
        verts_vec->emplace_back(vertices_d(i, 0), vertices_d(i, 1),
                                vertices_d(i, 2));
      // std::cout << "[DEBUG] Created " << verts_vec->size() << " FCL vertices."
                // << std::endl;
      std::shared_ptr<const std::vector<fcl::Vector3d>> verts_ptr = verts_vec;

      // std::vector<int> fcl_faces;
      // fcl_faces.reserve(faces.size() + faces.size() / 3);
      // for (size_t i = 0; i < faces.size(); i += 3) {
      //   fcl_faces.push_back(faces[i]);
      //   fcl_faces.push_back(faces[i + 1]);
      //   fcl_faces.push_back(faces[i + 2]);
      //   fcl_faces.push_back(-1);
      // }
      // auto faces_ptr =
      //     std::make_shared<const std::vector<int>>(std::move(fcl_faces));
      // Build a new vector with -1 after each triangle
      std::vector<int> fcl_faces;
      fcl_faces.reserve(faces.size() + faces.size() / 3);
      for (size_t i = 0; i + 2 < faces.size(); i += 3) {
        fcl_faces.push_back(faces[i]);
        fcl_faces.push_back(faces[i + 1]);
        fcl_faces.push_back(faces[i + 2]);
        fcl_faces.push_back(-1);  // terminator
      }
      auto faces_ptr =
          std::make_shared<const std::vector<int>>(std::move(fcl_faces));
      int num_faces = fcl_faces.size() / 4;  // recalculate after terminators
      // int num_faces = static_cast<int>(faces.size()) / 3;
      // auto faces_ptr = std::make_shared<const std::vector<int>>(faces);
      // std::cout << "[DEBUG] faces_ptr->size() = " << faces_ptr->size()
                // << " (expected " << num_faces * 4 << ")" << std::endl;
      // std::cout << "[DEBUG] First 20 ints: ";
      // for (int i = 0; i < std::min(20, (int)faces_ptr->size()); ++i)
      //   std::cout << (*faces_ptr)[i] << " ";
      // std::cout << std::endl;
      // std::cout << "[DEBUG] Creating FCL Convexd with " << num_faces
                // << " faces..." << std::endl;
      auto convex = std::make_shared<fcl::Convexd>(verts_ptr, num_faces,
                                                   faces_ptr, false);
      // std::cout << "[DEBUG] FCL Convexd created successfully." << std::endl;

      return convex;
    }
    default:
      throw std::runtime_error("Unsupported GeometricPrimitives type");
  }
}

fcl::Transform3d makeFCLTransform(const GeometricPrimitives& prim) {
  // uAIBot primitives usually store a pose as an Eigen::Affine3d or Matrix4d
  Eigen::Matrix4f htm = prim.htm;
  // FCL expects an Eigen::Isometry3d (which is Affine3d with no scaling)
  return fcl::Transform3d(htm.cast<double>());
}

// ---------- Distance computation ----------
std::pair<double, double> compute_distance(const GeometricPrimitives& a,
                        const GeometricPrimitives& b,
                        const Eigen::MatrixXf& vertices_a = Eigen::MatrixXf(),
                        const std::vector<int>& faces_a = std::vector<int>(),
                        const Eigen::MatrixXf& vertices_b = Eigen::MatrixXf(),
                        const std::vector<int>& faces_b = std::vector<int>()) {
  // Create FCL geometries
  // std::cout << "[DEBUG] Creating FCL geometries for primitives A and B"
  //           << std::endl;
  auto geom_a = makeFCLGeometry(a, vertices_a, faces_a);
  // std::cout << "[DEBUG] Geometry A created: " << std::endl;
  auto geom_b = makeFCLGeometry(b, vertices_b, faces_b);
  // std::cout << "[DEBUG] Geometry B created: " << std::endl;

  // Wrap into collision objects with their transforms
  auto obj_a =
      std::make_shared<fcl::CollisionObjectd>(geom_a, makeFCLTransform(a));
  auto obj_b =
      std::make_shared<fcl::CollisionObjectd>(geom_b, makeFCLTransform(b));
  // std::cout << "[DEBUG] Collision objects created for A and B" << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  // Set up distance request (default parameters)
  fcl::DistanceRequestd request;
  fcl::DistanceResultd result;
  // First compute distance (works when separated)
  fcl::distance(obj_a.get(), obj_b.get(), request, result);
  double min_dist = result.min_distance;
  // std::cout << "[DEBUG] Initial distance computed: " << min_dist << std::endl;

  // If distance > 0, we are done
  if (min_dist > 0.0) {
    auto t2 = std::chrono::high_resolution_clock::now();
    return {min_dist, std::chrono::duration<double, std::micro>(t2 - t1).count()};
    // return min_dist;
  }

  // Otherwise, we need collision info to get penetration depth
  fcl::CollisionRequestd collision_request;
  collision_request.enable_contact = true;
  collision_request.num_max_contacts = 1;
  fcl::CollisionResultd collision_result;

  fcl::collide(obj_a.get(), obj_b.get(), collision_request, collision_result);
  // std::cout << "[DEBUG] Collision check performed" << std::endl;

  auto t2 = std::chrono::high_resolution_clock::now();
  if (collision_result.isCollision()) {
    double penetration = collision_result.getContact(0).penetration_depth;
    return {-penetration, std::chrono::duration<double, std::micro>(t2 - t1).count()};
    // return -penetration;  // negative sign for penetration
  }
  // If not colliding but distance <= 0 (e.g., exactly touching), return 0
  return {0.0, std::chrono::duration<double, std::micro>(t2 - t1).count()};
  // return 0.0;
}

// ---------- pybind11 module ----------
PYBIND11_MODULE(sdf, m) {
  m.doc() = "Minimum distance between GeometricPrimitives using FCL";
  m.def("compute_distance", &compute_distance,
        "Compute the minimum distance between two GeometricPrimitives",
        py::arg("a"), py::arg("b"), py::arg("vertices_a") = Eigen::MatrixXf(),
        py::arg("faces_a") = std::vector<int>(),
        py::arg("vertices_b") = Eigen::MatrixXf(),
        py::arg("faces_b") = std::vector<int>());
}
