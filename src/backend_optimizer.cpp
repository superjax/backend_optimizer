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
  parameters_.enablePartialRelinearizationCheck = true;
  parameters_.factorization = ISAM2Params::QR;

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
  optimizer_ = ISAM2(parameters_);
  fixed_node_ = fixed_node;
  graph_fixed_ = false;
  node_name_to_id_map_.clear();
  node_id_to_name_map_.clear();
  num_nodes_ = 0;
  num_edges_ = 0;

  // Make space for the fixed node (the node will get added the first time we call add_edge)
  node_name_to_id_map_[fixed_node] = num_nodes_;
  node_id_to_name_map_[num_nodes_] = fixed_node;
  num_nodes_++;
}

double BackendOptimizer::add_edge_batch(py::list nodes, py::list edges)
{
  std::cout << "adding ";
  NonlinearFactorGraph new_graph;
  Values new_initial_estimates;

  // convert the nodes
  //  std::cout << "adding nodes \n";
  std::vector<py::list> stl_nodes = nodes.cast<std::vector<py::list>>();
  std::cout << stl_nodes.size() << " odometry edges ";
  for (int i = 0; i < stl_nodes.size(); i++)
  {
    std::string id = stl_nodes[i][0].cast<std::string>();
    double x = stl_nodes[i][1].cast<double>();
    double y = stl_nodes[i][2].cast<double>();
    double z = stl_nodes[i][3].cast<double>();
    //    std::cout << x << ", " << y << ", " << z << "\n";

    // connect the name of this node to an integer index
    node_name_to_id_map_[id] = num_nodes_;
    node_id_to_name_map_[num_nodes_] = id;

    new_initial_estimates.insert(num_nodes_, Pose2(x, y,  z));
    num_nodes_++;
  }

  // extract edges
  std::vector<py::list> stl_edges = edges.cast<std::vector<py::list>>();
  for (int i = 0; i < stl_edges.size(); i++)
  {
    std::string from = stl_edges[i][0].cast<std::string>();
    std::string to = stl_edges[i][1].cast<std::string>();
    double x = stl_edges[i][2].cast<double>();
    double y = stl_edges[i][3].cast<double>();
    double z = stl_edges[i][4].cast<double>();
    double P11 = stl_edges[i][5].cast<double>();
    double P22 = stl_edges[i][6].cast<double>();
    double P33 = stl_edges[i][7].cast<double>();

    // save off edge constraints so we can quickly load it if we want to add edges later
    std::vector<std::string> edge_vec = {from, to};
    edge_list_.push_back(edge_vec);
    std::vector<double> edge_constraint = {x, y, z, P11, P22, P33};
    edge_constraints_.push_back(edge_constraint);
    num_edges_++;

    // put this edge in the graph
    noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas(Vector3(P11, P22, P33));
    new_graph.emplace_shared<BetweenFactor<Pose2> >(node_name_to_id_map_[from], node_name_to_id_map_[to], Pose2(x, y, z), model);
  }

  // fix the fixed node if we haven't already
  if (!graph_fixed_)
  {
    int fixed_node_index = node_name_to_id_map_[fixed_node_];
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.001, 0.001, 0.001));
    new_graph.emplace_shared<PriorFactor<Pose2> >(fixed_node_index, Pose2(0, 0, 0), priorNoise);
    new_initial_estimates.insert(fixed_node_index, Pose2(0, 0, 0));
    graph_fixed_ = true;
  }

  // Add the new edge to the graph
  clock_t start_time = std::clock();
  result_ = optimizer_.update(new_graph, new_initial_estimates);
  clock_t time = std::clock() - start_time;

  std::cout << "took " << ((float)time)/CLOCKS_PER_SEC << " seconds " << std::endl;

  return ((float)time)/CLOCKS_PER_SEC;
}

double BackendOptimizer::add_lc_batch(pybind11::list edges)
{
  NonlinearFactorGraph new_graph;
  std::cout << "adding ";

  // extract edges
  std::vector<py::list> stl_edges = edges.cast<std::vector<py::list>>();
  std::cout << stl_edges.size() << " loop closures ";
  for (int i = 0; i < stl_edges.size(); i++)
  {
    std::string from = stl_edges[i][0].cast<std::string>();
    std::string to = stl_edges[i][1].cast<std::string>();
    double x = stl_edges[i][2].cast<double>();
    double y = stl_edges[i][3].cast<double>();
    double z = stl_edges[i][4].cast<double>();
    double P11 = stl_edges[i][5].cast<double>();
    double P22 = stl_edges[i][6].cast<double>();
    double P33 = stl_edges[i][7].cast<double>();

    // save off edge constraints so we can quickly load it if we want to add edges later
    std::vector<std::string> edge_vec = {from, to};
    edge_list_.push_back(edge_vec);
    std::vector<double> edge_constraint = {x, y, z, P11, P22, P33};
    edge_constraints_.push_back(edge_constraint);
    num_edges_++;

    // put this edge in the graph
    noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas(Vector3(P11, P22, P33));
    new_graph.emplace_shared<BetweenFactor<Pose2> >(node_name_to_id_map_[from], node_name_to_id_map_[to], Pose2(x, y, z), model);
  }

  // Add the new edges to the graph
  clock_t start_time = std::clock();
  result_ = optimizer_.update(new_graph);
  clock_t time = std::clock() - start_time;

  std::cout << "took " << ((float)time)/CLOCKS_PER_SEC << " seconds " << std::endl;
  return ((float)time)/CLOCKS_PER_SEC;
}

double BackendOptimizer::optimize()
{
  clock_t start_time = std::clock();
  result_ = optimizer_.update();
  clock_t time = std::clock() - start_time;

  std::cout << "optimization took " << ((float)time)/CLOCKS_PER_SEC << " seconds " << std::endl;

  return ((float)time)/CLOCKS_PER_SEC;
  //  std::cout << "optimized " << num_nodes_ << " nodes and " << num_edges_ << " edges " << std::endl;
  //  result_.print("optimization results:");
}

py::dict BackendOptimizer::get_optimized()
{
  //  std::cout << "\n\nname to id map\n\n";
  //  for(auto elem : node_id_to_name_map_)
  //  {
  //     std::cout << elem.first << " " << elem.second << "\n";
  //  }
  //  for(auto elem : node_name_to_id_map_)
  //  {
  //     std::cout << elem.first << " " << elem.second << "\n";
  //  }
  //  optimizer_.print("status");

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
    edge.append(edge_constraints_[i][5]);
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
      .def("add_edge_batch", &BackendOptimizer::add_edge_batch)
      .def("add_lc_batch", &BackendOptimizer::add_lc_batch)
      .def("optimize", &BackendOptimizer::optimize)
      .def("get_optimized", &BackendOptimizer::get_optimized);

  return m.ptr();
}
