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


backendOptimizer::backendOptimizer(){}

int backendOptimizer::new_graph(py::list nodes, py::list edges, int fixed_node)
{
  std::cout << "new graph!\n";

  std::vector<py::list> stl_nodes = to_std_vector<py::list>(nodes);

  for (int i = 0; i < stl_nodes.size(); i++)
  {
    std::string name = py::extract<std::string>(stl_nodes[i][0]);
    std::vector <double> pose (3, 0.0);
    pose[0] = py::extract<double>(stl_nodes[i][1]);
    pose[1] = py::extract<double>(stl_nodes[i][2]);
    pose[2] = py::extract<double>(stl_nodes[i][3]);

    cout << name << ", " << pose[0] << ", " << pose[1] << ", " << pose[2] << endl;
  }
}

void backendOptimizer::add(py::list nodes, py::list edges){}
void backendOptimizer::optimize(){}
py::list backendOptimizer::get_optimized()
{
  cout << "get optimized" << endl;
  std::vector<std::vector<double> > optimized_node_poses(4, std::vector<double>(3, 0));
  std::vector<std::string> optimized_node_names(4, "");
  for (int i = 0; i < optimized_node_poses.size(); i++)
  {
    optimized_node_poses[i][0] = 1;
    optimized_node_poses[i][1] = 2;
    optimized_node_poses[i][2] = 3;
    optimized_node_names[i] = "0_00" + std::to_string(i);
  }

  py::list out_list;
  for (int i = 0; i < optimized_node_poses.size(); i++)
  {
    py::list node_list;
    node_list.append(optimized_node_names[i]);
    node_list.append(optimized_node_poses[i][0]);
    node_list.append(optimized_node_poses[i][1]);
    node_list.append(optimized_node_poses[i][2]);
    out_list.append(node_list);
  }
  return out_list;
}
