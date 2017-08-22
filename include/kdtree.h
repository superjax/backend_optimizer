#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nanoflann.hpp>
#include <eigen3/Eigen/Core>
#include <map>

namespace py = pybind11;
using namespace nanoflann;



class KDTree
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KDTree();

  void add_points(py::list new_points);
  pybind11::list find_closest_point(py::list point, double distance);

private:
  typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, Eigen::Dynamic, 3>> KDTree3D;

  Eigen::Matrix<double, Eigen::Dynamic, 3> points_mat;

  std::map<std::string, uint64_t> node_name_to_id_map_;
  std::map<uint64_t, std::string> node_id_to_name_map_;

  KDTree3D* kd_tree_;

  int num_points_;
};
