#ifndef BACKEND_OPTIMIZER_H
#define BACKEND_OPTIMIZER_H

#include <pybind11/pybind11.h>
#include <ros/ros.h>

// GTSAM includes
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

// Other includes
#include <vector>
#include <time.h>
#include <map>

namespace py = pybind11;
using namespace gtsam;

namespace backend_optimizer
{

class BackendOptimizer
{

public:

    BackendOptimizer();
    py::list cumsum(py::list list_to_sum);
    int new_graph(std::string fixed_node);
    void add_edge(py::list node, py::list edge);
    void add_loop_closure(py::list edge);
    void optimize();
    py::dict get_optimized();
    py::dict get_global_pose_and_covariance(std::string node);

private:
    std::string fixed_node_;
    Values initialEstimate_;
    ISAM2Params parameters_;
    ISAM2Result result_;
    ISAM2 optimizer_;

    uint64_t num_nodes_;
    uint64_t num_edges_;

    std::map<std::string, uint64_t> node_name_to_id_map_;
    std::map<uint64_t, std::string> node_id_to_name_map_;

    std::vector<std::vector<std::string>> edge_list_;
    std::vector<std::vector<double>> edge_constraints_;


};

class Edge
{

};

} // namespace backend_optimizer

#endif // backendOptimizer_H
