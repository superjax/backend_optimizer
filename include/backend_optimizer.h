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
    void add(py::list nodes, py::list edges);
    void optimize();
    py::dict get_optimized();

private:
    bool node_exists(std::string node_id);
    bool edge_exists(std::vector<std::string> edge);

    int num_nodes_;
    int num_edges_;
    std::string fixed_node_id_;
    bool graph_fixed_;
    NonlinearFactorGraph graph_;
    Values initialEstimate_;
    ISAM2Params parameters_;
    ISAM2Result result_;
    ISAM2 optimizer_;

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
