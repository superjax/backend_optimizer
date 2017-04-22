#include "backend_optimizer.h"
#include <boost/python/stl_iterator.hpp>
#include <iostream>

namespace py = boost::python;
using namespace backend_optimizer;
using namespace std;

// helper functions to convert to and from python lists
template<class T>
py::list std_vector_to_py_list(const std::vector<T>& v)
{
  py::object get_iter = py::iterator<std::vector<T> >();
  py::object iter = get_iter(v);
  py::list l(iter);
  return l;
}

template< typename T >
inline
std::vector< T > to_std_vector( const py::object& iterable )
{
  return std::vector< T >( py::stl_input_iterator< T >( iterable ),
                           py::stl_input_iterator< T >( ) );
}


backendOptimizer::backendOptimizer()
{
  // parameters_.relativeErrorTol = 1e-5;
  // parameters_.maxIterations = 1000;
  // parameters_.linearSolverType = NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;

  num_nodes_ = 0;
  edge_constraints_.clear();
  optimized_poses_.clear();
  edge_list_.clear();
  node_id_to_index_map.clear();
  index_to_node_id_map.clear();
}


int backendOptimizer::new_graph(py::list nodes, py::list edges, std::string fixed_node)
{
  num_nodes_ = 0;
  edge_constraints_.clear();
  optimized_poses_.clear();
  edge_list_.clear();
  node_id_to_index_map.clear();
  index_to_node_id_map.clear();

  // extract nodes
  std::vector<py::list> stl_nodes = to_std_vector<py::list>(nodes);
  for (int i = 0; i < stl_nodes.size(); i++)
  {
    std::string id = py::extract<std::string>(stl_nodes[i][0]);
    double x = py::extract<double>(stl_nodes[i][1]);
    double y = py::extract<double>(stl_nodes[i][2]);
    double z = py::extract<double>(stl_nodes[i][3]);

    // create connection between integer index and string node name
    node_id_to_index_map[id] = num_nodes_;
    index_to_node_id_map[num_nodes_] = id;

    std::vector<double> pose = {x, y, z};
    optimized_poses_.push_back(pose);
    // initialEstimate_.insert(num_nodes_, Pose2(x, y,  z));
    num_nodes_++;
  }

  // extract edges
  std::vector<py::list> stl_edges = to_std_vector<py::list>(edges);
  for (int i = 0; i < stl_edges.size(); i++)
  {
    std::string from = py::extract<std::string>(stl_edges[i][0]);
    std::string to = py::extract<std::string>(stl_edges[i][1]);
    double x = py::extract<double>(stl_edges[i][2]);
    double y = py::extract<double>(stl_edges[i][3]);
    double z = py::extract<double>(stl_edges[i][4]);
    double P11 = py::extract<double>(stl_edges[i][5]);
    double P22 = py::extract<double>(stl_edges[i][6]);
    double P33 = py::extract<double>(stl_edges[i][7]);

    // save off edge constraints so we can quickly load it if we want to add edges later
    std::vector<string> edge = {from, to};
    edge_list_.push_back(edge);
    std::vector<double> edge_constraint = {x, y, z, P11, P22, P33};
    edge_constraints_.push_back(edge_constraint);

    // Create the Noise model for this edge
    // noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas(Vector3(P11, P22, P33));

    // graph_.emplace_shared<BetweenFactor<Pose2> >(node_id_to_index_map[from], node_id_to_index_map[to],
                                                 // Pose2(x, y, z), model);
    num_edges_++;
  }

  // fix the fixed node
  int fixed_node_index_ = node_id_to_index_map[fixed_node];
  // noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.001, 0.001, 0.001));
  // graph_.emplace_shared<PriorFactor<Pose2> >(fixed_node_index_, Pose2(0, 0, 0), priorNoise);
}

void backendOptimizer::add(py::list nodes, py::list edges){}

void backendOptimizer::optimize()
{
  // GaussNewtonOptimizer optimizer(graph_, initialEstimate_, parameters_);
  clock_t start_time = std::clock();

  // Values optimized_values = optimizer.optimize();
  clock_t time = std::clock() - start_time;

  std::cout << "took " << time << " ticks or " << ((float)time)/CLOCKS_PER_SEC << " seconds " << std::endl;
  std::cout << "optimized " << num_nodes_ << " nodes and " << num_edges_ << " edges " << std::endl;

  // Pull optimized values into the proper arrays
  // optimized_poses_.clear();
  for (int i = 0; i < num_nodes_; i++)
  {
    // Pose2 output = optimized_values.at<Pose2>(i);
    // std::vector<double> pose = {output.x(), output.y(), output.theta()};
    // optimized_poses_.push_back(pose);
  }
}

py::dict backendOptimizer::get_optimized()
{
  py::dict out_dict;
  py::list node_list;
  for (int i = 0; i < optimized_poses_.size(); i++)
  {
    std::string node_name = index_to_node_id_map[i];
    py::list node;
    node.append(node_name);
    node.append(optimized_poses_[i][0]);
    node.append(optimized_poses_[i][1]);
    node.append(optimized_poses_[i][2]);
    node_list.append(node);
  }
  out_dict["nodes"] = node_list;

  py::list edge_list;
  for (int i = 0; i < edge_constraints_.size(); i++)
  {
    py::list edge;
    std::string from = edge_list_[i][0];
    std::string to = edge_list_[i][1];

    edge.append(from);
    edge.append(to);
    edge.append(edge_constraints_[i][0]);
    edge.append(edge_constraints_[i][1]);
    edge.append(edge_constraints_[i][2]);
    edge.append(edge_constraints_[i][3]);
    edge.append(edge_constraints_[i][4]);
    edge_list.append(edge);
  }
  out_dict["edges"] = edge_list;

  out_dict["fixed_node"] = index_to_node_id_map[fixed_node_index_];

  return out_dict;
}
