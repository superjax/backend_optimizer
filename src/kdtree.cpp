#include "kdtree.h"
#include <iostream>

KDTree::KDTree()
{
  num_points_ = 0;
  points_mat.resize(0, 3);
  kd_tree_ = NULL;
}

void KDTree::add_points(py::list new_points)
{
  std::vector<py::list> stl_points = new_points.cast<std::vector<py::list>>();

  // prepare the matrix container for the new points
  points_mat.conservativeResize(num_points_ + stl_points.size(), 3);

  // convert the python objects to the c++ matrix values
  for (int i = 0; i < stl_points.size(); i++)
  {
    std::string id = stl_points[i][0].cast<std::string>();
    points_mat.row(num_points_) << stl_points[i][1].cast<double>(),
                                    stl_points[i][2].cast<double>(),
                                    stl_points[i][3].cast<double>();

    // connect the name of this node to an integer index
    node_name_to_id_map_[id] = num_points_;
    node_id_to_name_map_[num_points_] = id;
    num_points_++;
  }

  if(kd_tree_)
    delete kd_tree_;

  kd_tree_ = new KDTree3D(3, points_mat, 10);
  kd_tree_->index->buildIndex();
}

py::list KDTree::find_closest_point(pybind11::list point, double distance)
{
  std::string id = point[0].cast<std::string>();
  std::vector<double> query_pt(3);

  std::string s = id;
  std::string delim = "_";
  int vehicle_id = std::stoi(s.substr(0, s.find(delim)));
  int node_id = std::stoi(s.erase(0, s.find(delim) + delim.length()));

  query_pt[0] = point[1].cast<double>(),
  query_pt[1] = point[2].cast<double>(),
  query_pt[2] = point[3].cast<double>();

  uint8_t num_results = 25;
  std::vector<uint64_t> ret_indexes(num_results);
  std::vector<double> distances(num_results);

  nanoflann::KNNResultSet<double> resultsSet(num_results);
  resultsSet.init(&ret_indexes[0], &distances[0]);

  // Find the nearest neighbors
  kd_tree_->index->findNeighbors(resultsSet, &query_pt[0], nanoflann::SearchParams(10));

  std::string closest_id = "none";
  double x = 0;
  double y = 0;
  double z = 0;

  uint64_t index_of_id = node_name_to_id_map_[id];
  py::list out;

  bool found_match = false;
  for (int i = 0; i < num_results; i++)
  {
    if (index_of_id != ret_indexes[i])
    {
      if (distances[i] < distance)
      {
        std::string s = node_id_to_name_map_[ret_indexes[i]];
        int proposed_vehicle_id = std::stoi(s.substr(0, s.find(delim)));
        int proposed_node_id = std::stoi(s.erase(0, s.find(delim) + delim.length()));

        if (proposed_vehicle_id == vehicle_id)
        {
          if (std::abs(proposed_node_id - node_id) < 10)
            continue;
        }

        closest_id = node_id_to_name_map_[ret_indexes[i]];
        x = points_mat(ret_indexes[i], 0);
        y = points_mat(ret_indexes[i], 1);
        z = points_mat(ret_indexes[i], 2);
//          printf("closest point to %s(%lu) is %s(%lu) at distance %f", id.c_str(), index_of_id, closest_id.c_str(), ret_indexes[i], distances[i]);
        found_match = true;
        break;
      }
      else
      {
//        printf("point %s(%lu) is %f away from %s(%lu)",  id.c_str(), index_of_id, distances[i], closest_id.c_str(), ret_indexes[i]);
        break;
      }
    }
  }

  out.append(closest_id);
  out.append(x);
  out.append(y);
  out.append(z);
  return out;
}

PYBIND11_PLUGIN(kdtree) {
  py::module m("kdtree", "kdtree using nanoflann in C++");

  py::class_<KDTree>(m, "KDTree")
      .def("__init__", [](KDTree &instance) {
    new (&instance) KDTree();
  })
  .def("add_points", &KDTree::add_points)
      .def("find_closest_point", &KDTree::find_closest_point);
  return m.ptr();
}



