#ifndef BACKEND_OPTIMIZER_H
#define BACKEND_OPTIMIZER_H

#include <boost/python.hpp>
#include <ros/ros.h>

// GTSAM includes
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Key.h>
//#include <gtsam/slam/PriorFactor.h>
//#include <gtsam/slam/BetweenFactor.h>
//#include <gtsam/nonlinear/NonlinearFactorGraph.h>
//#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
//#include <gtsam/nonlinear/Marginals.h>
//#include <gtsam/nonlinear/Values.h>

// Other includes
#include <vector>
#include <time.h>
#include <map>

namespace py = boost::python;
using namespace gtsam;

namespace backend_optimizer
{

class backendOptimizer
{

public:

  backendOptimizer();
  int new_graph(py::list nodes, py::list edges, std::__cxx11::string fixed_node);
  void add(py::list nodes, py::list edges);
  void optimize();
  py::dict get_optimized();

private:
  int num_nodes_ = 0;
  int num_edges_ = 0;
  int fixed_node_index_ = 0;
  // NonlinearFactorGraph graph_;
  // Values initialEstimate_;
  // GaussNewtonParams parameters_;
//  GaussNewtonOptimizer optimizer_;

  std::map<std::string, int> node_id_to_index_map;
  std::map<int, std::string> index_to_node_id_map;
  std::vector<std::vector<std::string>> edge_list_;
  std::vector<std::vector<double>> optimized_poses_;
  std::vector<std::vector<double>> edge_constraints_;


};

class Edge
{

};

} // namespace backend_optimizer

#endif // backendOptimizer_H
