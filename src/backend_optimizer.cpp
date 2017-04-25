#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <backend_optimizer.h>
#include <algorithm>

namespace py = pybind11;
using namespace backend_optimizer;

BackendOptimizer::BackendOptimizer() :
  parameters_(), optimizer_(parameters_)
{
  parameters_.relinearizeThreshold = 0.01;
  parameters_.relinearizeSkip = 1;

  num_nodes_ = 0;
  num_edges_ = 0;
  fixed_node_id_.clear();
  graph_fixed_ = false;
  edge_constraints_.clear();
  optimized_poses_.clear();
  edge_list_.clear();
  node_id_to_index_map.clear();
  index_to_node_id_map.clear();
}

int BackendOptimizer::new_graph(std::string fixed_node)
{
  num_nodes_ = 0;
  edge_constraints_.clear();
  optimized_poses_.clear();
  edge_list_.clear();
  node_id_to_index_map.clear();
  index_to_node_id_map.clear();

  graph_fixed_ = false;
  fixed_node_id_ = fixed_node;

  optimizer_.clear();
}

void BackendOptimizer::add(py::list nodes, py::list edges)
{
  NonlinearFactorGraph new_graph;
  Values new_initial_estimates;

  std::vector<py::list> stl_nodes = nodes.cast<std::vector<py::list>>();
  for (int i = 0; i < stl_nodes.size(); i++)
  {
    std::string id = stl_nodes[i][0].cast<std::string>();

    // If this node already exists, then skip it
    if (node_exists(id))
      continue;

    double x = stl_nodes[i][1].cast<double>();
    double y = stl_nodes[i][2].cast<double>();
    double z = stl_nodes[i][3].cast<double>();

    // create connection between integer index and string node name
    node_id_to_index_map[id] = num_nodes_;
    index_to_node_id_map[num_nodes_] = id;

    std::vector<double> pose = {x, y, z};
    optimized_poses_.push_back(pose);
    new_initial_estimates.insert(num_nodes_, Pose2(x, y,  z));
    num_nodes_++;
  }

  // extract edges
  std::vector<py::list> stl_edges = edges.cast<std::vector<py::list>>();
  for (int i = 0; i < stl_edges.size(); i++)
  {
    std::string from = stl_edges[i][0].cast<std::string>();
    std::string to = stl_edges[i][1].cast<std::string>();
    std::vector<std::string> edge = {from, to};

    if (edge_exists(edge))
      continue;

    double x = stl_edges[i][2].cast<double>();
    double y = stl_edges[i][3].cast<double>();
    double z = stl_edges[i][4].cast<double>();
    double P11 = stl_edges[i][5].cast<double>();
    double P22 = stl_edges[i][6].cast<double>();
    double P33 = stl_edges[i][7].cast<double>();

    // save off edge constraints so we can quickly load it if we want to add edges later
    edge_list_.push_back(edge);
    std::vector<double> edge_constraint = {x, y, z, P11, P22, P33};
    edge_constraints_.push_back(edge_constraint);

    // Create the Noise model for this edge
    noiseModel::Diagonal::shared_ptr model = noiseModel::Diagonal::Sigmas(Vector3(P11, P22, P33));

    new_graph.emplace_shared<BetweenFactor<Pose2> >(node_id_to_index_map[from], node_id_to_index_map[to],
                                                    Pose2(x, y, z), model);
    num_edges_++;
  }
  if(!graph_fixed_)
  {
    // fix the fixed node
    std::cout << "fixing node " << fixed_node_id_ << "\n";
    int fixed_node_index = node_id_to_index_map[fixed_node_id_];
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.001, 0.001, 0.001));
    graph_.emplace_shared<PriorFactor<Pose2> >(fixed_node_index, Pose2(0, 0, 0), priorNoise);
    graph_fixed_ = true;
  }

  result_ = optimizer_.update(new_graph, new_initial_estimates);
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

  Values optimized_values = optimizer_.calculateBestEstimate();

  // Pull optimized values into the proper arrays
  optimized_poses_.clear();
  for (int i = 0; i < num_nodes_; i++)
  {
    Pose2 output = optimized_values.at<Pose2>(i);
    std::vector<double> pose = {output.x(), output.y(), output.theta()};
    optimized_poses_.push_back(pose);
  }
}


py::dict BackendOptimizer::get_optimized()
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

  out_dict["fixed_node"] = fixed_node_id_;

  return out_dict;
}

bool BackendOptimizer::node_exists(std::string node_id)
{
  return !(node_id_to_index_map.find(node_id) == node_id_to_index_map.end());
}

bool BackendOptimizer::edge_exists(std::vector<std::string> edge)
{
  return !(std::find(edge_list_.begin(), edge_list_.end(), edge) != edge_list_.end());
}




PYBIND11_PLUGIN(backend_optimizer) {
  py::module m("backend_optimizer", "pybind11 backend_optimizer plugin");

  py::class_<BackendOptimizer>(m, "Optimizer")
      .def("__init__", [](BackendOptimizer &instance) {
            new (&instance) BackendOptimizer();
           })
      .def("new_graph", &BackendOptimizer::new_graph)
      .def("add", &BackendOptimizer::add)
      .def("optimize", &BackendOptimizer::optimize)
      .def("get_optimized", &BackendOptimizer::get_optimized);

  return m.ptr();
}
