#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <backend_optimizer.h>

namespace py = pybind11;
using namespace backend_optimizer;

BackendOptimizer::BackendOptimizer() :
  parameters_(), optimizer_(parameters_)
{
  parameters_.relinearizeThreshold = 0.01;
  parameters_.relinearizeSkip = 1;

  edge_constraints_.clear();
  edge_list_.clear();
  fixed_node_ = "";
  node_name_to_id_map_.clear();
  node_id_to_name_map_.clear();

  num_nodes_ = 0;
  num_edges_ = 0;
}

// create a new graph, fixed at the fixed_node
int BackendOptimizer::new_graph(std::string fixed_node)
{
  edge_constraints_.clear();
  edge_list_.clear();
  optimizer_.clear();
  fixed_node_ = fixed_node;
  node_name_to_id_map_.clear();
  node_id_to_name_map_.clear();
  num_nodes_ = 0;
  num_edges_ = 0;
}

// add an odometry edge
void BackendOptimizer::add_edge(py::list node, py::list edge)
{
  NonlinearFactorGraph new_graph;
  Values new_initial_estimates;

  // convert the node
  std::string id = node[0].cast<std::string>();
  double x = node[1].cast<double>();
  double y = node[2].cast<double>();
  double z = node[3].cast<double>();

  // connect the name of this node to an integer index
  node_name_to_id_map_[id] = num_nodes_;
  node_id_to_name_map_[num_nodes_] = id;

  new_initial_estimates.insert(num_nodes_, Pose2(x, y,  z));
  num_nodes_++;

  // extract edge
  std::string from = edge[0].cast<std::string>();
  std::string to = edge[1].cast<std::string>();
  x = edge[2].cast<double>();
  y = edge[3].cast<double>();
  z = edge[4].cast<double>();
  double P11 = edge[5].cast<double>();
  double P22 = edge[6].cast<double>();
  double P33 = edge[7].cast<double>();

  // save off edge constraints so we can quickly load it if we want to add edges later
  std::vector<std::string> edge_vec = {from, to};
  edge_list_.push_back(edge_vec);
  std::vector<double> edge_constraint = {x, y, z, P11, P22, P33};
  edge_constraints_.push_back(edge_constraint);
  num_edges_++;

  // Create the Noise model for this edge
  noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas(Vector3(P11, P22, P33));

  // put this edge in the graph
  new_graph.emplace_shared<BetweenFactor<Pose2> >(node_name_to_id_map_[from], node_name_to_id_map_[to], Pose2(x, y, z), model);

  // Add the new edge to the graph
  result_ = optimizer_.update(new_graph, new_initial_estimates);
}

// add a loop closure edge
void BackendOptimizer::add_loop_closure(py::list edge)
{
  NonlinearFactorGraph new_graph;

  // extract edge
  std::string from = edge[0].cast<std::string>();
  std::string to = edge[1].cast<std::string>();
  double x = edge[2].cast<double>();
  double y = edge[3].cast<double>();
  double z = edge[4].cast<double>();
  double P11 = edge[5].cast<double>();
  double P22 = edge[6].cast<double>();
  double P33 = edge[7].cast<double>();

  // save off edge constraints so we can quickly load it if we want to add edges later
  std::vector<std::string> edge_vec = {from, to};
  edge_list_.push_back(edge_vec);
  std::vector<double> edge_constraint = {x, y, z, P11, P22, P33};
  edge_constraints_.push_back(edge_constraint);
  num_edges_++;

  // Create the Noise model for this edge
  noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas(Vector3(P11, P22, P33));
  new_graph.emplace_shared<BetweenFactor<Pose2> >(node_name_to_id_map_[from], node_name_to_id_map_[to], Pose2(x, y, z), model);

  // Add the new edge to the graph
  result_ = optimizer_.update(new_graph);
}


void BackendOptimizer::optimize()
{
  clock_t start_time = std::clock();

  for(int i = 0; i < 10; i++)
  {
    result_ = optimizer_.update();
  }

  clock_t time = std::clock() - start_time;

  std::cout << "took " << time << " ticks or " << ((float)time)/CLOCKS_PER_SEC << " seconds " << std::endl;
  std::cout << "optimized " << num_nodes_ << " nodes and " << num_edges_ << " edges " << std::endl;
  result_.print("optimization results:");
}


py::dict BackendOptimizer::get_optimized()
{

  Values optimized_values = optimizer_.calculateBestEstimate();

  py::dict out_dict;
  py::list node_list;
  for (int i = 0; i < num_nodes_; i++)
  {
    // Get the optimized pose out of the graph
    Pose2 output = optimized_values.at<Pose2>(i);

    // pack up into a python list
    std::string node_name = node_id_to_name_map_[i];
    py::list node;
    node.append(node_name);
    node.append(output.x());
    node.append(output.y());
    node.append(output.theta());
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

  out_dict["fixed_node"] = fixed_node_;

  return out_dict;
}


PYBIND11_PLUGIN(backend_optimizer) {
  py::module m("backend_optimizer", "pybind11 backend_optimizer plugin");

  py::class_<BackendOptimizer>(m, "Optimizer")
      .def("__init__", [](BackendOptimizer &instance) {
    new (&instance) BackendOptimizer();
  })
  .def("new_graph", &BackendOptimizer::new_graph)
      .def("add_edge", &BackendOptimizer::add_edge)
      .def("add_loop_closure", &BackendOptimizer::add_loop_closure)
      .def("optimize", &BackendOptimizer::optimize)
      .def("get_optimized", &BackendOptimizer::get_optimized);

  return m.ptr();
}
