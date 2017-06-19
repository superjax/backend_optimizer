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
//  std::cout << "built index\n";
}

py::list KDTree::find_closest_point(pybind11::list point, double distance)
{
  std::string id = point[0].cast<std::string>();
  std::vector<double> query_pt(3);

  query_pt[0] = point[1].cast<double>(),
  query_pt[1] = point[2].cast<double>(),
  query_pt[2] = point[3].cast<double>();

  uint8_t num_results = 2;
  std::vector<uint64_t> ret_indexes(num_results);
  std::vector<double> distances(num_results);

  nanoflann::KNNResultSet<double> resultsSet(num_results);
  resultsSet.init(&ret_indexes[0], &distances[0]);

  // Find the nearest neighbors
  kd_tree_->index->findNeighbors(resultsSet, &query_pt[0], nanoflann::SearchParams(10));

//  std::cout << "\n\n\n";
//  std::cout << "matching " << id << ": " << node_name_to_id_map_[id] << " = " << query_pt[0] << ", " << query_pt[1] << ", " << query_pt[2] << "\n";
//  for( int i = 0; i < num_results; i++)
//  {
//    double x = points_mat(ret_indexes[i], 0);
//    double y = points_mat(ret_indexes[i], 1);
//    double z = points_mat(ret_indexes[i], 2);
////    std::cout << "result[" << i << "]=" << ret_indexes[i] << ":" << node_id_to_name_map_[ret_indexes[i]] << " -> " << x << ", " << y << ", " << z << " dist = " << distances[i] << "\n";
//  }

  std::string closest_id = "none";
  double x = 0;
  double y = 0;
  double z = 0;

  uint64_t index_of_id = node_name_to_id_map_[id];
  py::list out;

  if (index_of_id == ret_indexes[0])
  {
    if (distances[1] < distance)
    {
      closest_id = node_id_to_name_map_[ret_indexes[1]];
      x = points_mat(ret_indexes[1], 0);
      y = points_mat(ret_indexes[1], 1);
      z = points_mat(ret_indexes[1], 2);
    }
//    std::cout << "found match 1 " << id << ":" << index_of_id << " with " << node_id_to_name_map_[ret_indexes[1]] << ":" << ret_indexes[1] << "\n";
//    std::cout << "in = "
//              <<   query_pt[0]  << ", "
//              <<   query_pt[1]  << ", "
//              <<   query_pt[2]  << "\n ";

//    std::cout << "to = "
//              << x << ", " << y << ", " << z << "\n";
//    std::cout << "dist = " << distances[1] << "\n";
  }
  else
  {
    if (distances[0] < distance)
    {
      closest_id = node_id_to_name_map_[ret_indexes[0]];
      x = points_mat(ret_indexes[0], 0);
      y = points_mat(ret_indexes[0], 1);
      z = points_mat(ret_indexes[0], 2);
    }
//    std::cout << "found match 2 " << id << ":" << index_of_id << " with " << node_id_to_name_map_[ret_indexes[0]] << ":" << ret_indexes[0] << "\n";
//    std::cout << "in = "
//              <<   query_pt[0]  << ", "
//              <<   query_pt[1]  << ", "
//              <<   query_pt[2]  << "\n ";

//    std::cout << "to = "
//              << x << ", " << y << ", " << z << "\n";
//    std::cout << "dist = " << distances[0] << "\n";
  }

//  std::cout << "\n\npoints_mat*****************\n " << points_mat << "\n\n";

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



