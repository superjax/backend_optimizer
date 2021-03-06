#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <backend_optimizer.h>
#include <fstream>
#include <ostream>
#include <chrono>

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
int BackendOptimizer::new_graph(std::string fixed_node, int id)
{
  optimizer_.clear();
  agent_id_ = id;
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
  //  std::cout << "adding ";
  NonlinearFactorGraph new_graph;
  Values new_initial_estimates;

  // convert the nodes
  //  std::cout << "adding nodes \n";
  std::vector<py::list> stl_nodes = nodes.cast<std::vector<py::list>>();
  //  std::cout << stl_nodes.size() << " odometry edges ";
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
  std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
  result_ = optimizer_.update(new_graph, new_initial_estimates);
  std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

double BackendOptimizer::add_lc_batch(pybind11::list edges)
{
  NonlinearFactorGraph new_graph;
  //  std::cout << "adding ";

  // extract edges
  std::vector<py::list> stl_edges = edges.cast<std::vector<py::list>>();
  //  std::cout << stl_edges.size() << " loop closures ";
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

  std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
  result_ = optimizer_.update(new_graph);
  std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

double BackendOptimizer::optimize()
{
  std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
  result_ = optimizer_.update();
  std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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
    edge.append(edge_constraints_[i][5]);
    edge_list.append(edge);
  }
  out_dict["edges"] = edge_list;

  out_dict["fixed_node"] = fixed_node_;

  return out_dict;
}

py::dict BackendOptimizer::batch_optimize(pybind11::list nodes, pybind11::list edges, std::string fixed_node, int max_iterations, double epsilon)
{
  //  std::cout << "adding ";
  NonlinearFactorGraph new_graph;
  Values prev_estimates;

  // convert the nodes
//  std::cout << "adding nodes \n";
  std::vector<py::list> stl_nodes = nodes.cast<std::vector<py::list>>();
  std::map<std::string, uint64_t> node_name_to_id_map;
  std::map<uint64_t, std::string> node_id_to_name_map;
//  std::cout << stl_nodes.size() << " odometry edges ";
  for (int i = 0; i < stl_nodes.size(); i++)
  {
    std::string id = stl_nodes[i][0].cast<std::string>();
    double x = stl_nodes[i][1].cast<double>();
    double y = stl_nodes[i][2].cast<double>();
    double z = stl_nodes[i][3].cast<double>();
    //    std::cout << x << ", " << y << ", " << z << "\n";

    // connect the name of this node to an integer index
    node_name_to_id_map[id] = i;
    node_id_to_name_map[i] = id;

    prev_estimates.insert(i, Pose2(x, y, z));
  }

  // extract edges
  std::vector<std::vector<double>> edge_constraints;
  std::vector<py::list> stl_edges = edges.cast<std::vector<py::list>>();
  std::vector<std::vector<std::string>> edge_list;
  for (int i = 0; i < stl_edges.size(); i++)
  {
    std::string from = stl_edges[i][0].cast<std::string>();
    std::string to = stl_edges[i][1].cast<std::string>();
    double x = stl_edges[i][2].cast<double>();
    double y = stl_edges[i][3].cast<double>();
    double z = stl_edges[i][4].cast<double>();
    double P11 = 1.0/stl_edges[i][5].cast<double>();
    double P22 = 1.0/stl_edges[i][6].cast<double>();
    double P33 = 1.0/stl_edges[i][7].cast<double>();

    // save off edge constraints so we can quickly load them later
    std::vector<std::string> edge_vec = {from, to};
    edge_list.push_back(edge_vec);
    std::vector<double> edge_constraint = {x, y, z, P11, P22, P33};
    edge_constraints.push_back(edge_constraint);

    // put this edge in the graph
    noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas(Vector3(P11, P22, P33));
    new_graph.emplace_shared<BetweenFactor<Pose2> >(node_name_to_id_map[from], node_name_to_id_map[to], Pose2(x, y, z), model);
  }

  // fix the fixed node
  int fixed_node_index = node_name_to_id_map[fixed_node];
  noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.001, 0.001, 0.001));
  new_graph.emplace_shared<PriorFactor<Pose2> >(fixed_node_index, Pose2(0, 0, 0), priorNoise);

  // Optimize the Graph
//  new_graph.print();
  LevenbergMarquardtOptimizer optimizer(new_graph, prev_estimates);
  optimizer.optimize();
//  std::cout << "iter: " << optimizer.iterations() << " error = " << optimizer.error();
  int iter = 0;
  double error = 1e25;
  while (iter < max_iterations && error > epsilon)
  {
    double squared_error_sum = 0;
    optimizer.iterate();
    iter++;
    Values current_values = optimizer.values();
    for (int i = 0; i < prev_estimates.size(); i++)
    {
      Pose2 cur = current_values.at<Pose2>(i);
      Pose2 prev = prev_estimates.at<Pose2>(i);
      squared_error_sum += pow(cur.x() - prev.x(), 2);
      squared_error_sum += pow(cur.y() - prev.y(), 2);
      squared_error_sum += pow(cur.theta() - prev.theta(), 2);
    }
    error = sqrt(squared_error_sum);
    std::swap(current_values, prev_estimates);
//    std::cout << "iter: " << iter << " error: " << error << std::endl;
  }

  Values optimized_values = optimizer.values();
//  optimized_values.print();

  py::dict out_dict;
  py::list node_list;
  for (int i = 0; i < stl_nodes.size(); i++)
  {
    // Get the optimized pose out of the graph
    Pose2 output = optimized_values.at<Pose2>(i);

    // pack up into a python list
    std::string node_name = node_id_to_name_map[i];
    py::list node;
    node.append(node_name);
    node.append(output.x());
    node.append(output.y());
    node.append(output.theta());
    node_list.append(node);
  }
  out_dict["nodes"] = node_list;

  py::list out_edge_list;
  for (int i = 0; i < edge_constraints.size(); i++)
  {
    py::list edge;
    std::string from = edge_list[i][0];
    std::string to = edge_list[i][1];

    edge.append(from);
    edge.append(to);
    edge.append(edge_constraints[i][0]);
    edge.append(edge_constraints[i][1]);
    edge.append(edge_constraints[i][2]);
    edge.append(edge_constraints[i][3]);
    edge.append(edge_constraints[i][4]);
    edge.append(edge_constraints[i][5]);
    out_edge_list.append(edge);
  }
  out_dict["edges"] = out_edge_list;

  out_dict["fixed_node"] = fixed_node;
  out_dict["iter"] = iter;
  out_dict["error"] = error;
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
      .def("get_optimized", &BackendOptimizer::get_optimized)
      .def("batch_optimize", &BackendOptimizer::batch_optimize);

  return m.ptr();
}
