#ifndef BACKEND_OPTIMIZER_H
#define BACKEND_OPTIMIZER_H

#include <boost/python.hpp>
#include <ros/ros.h>

namespace py = boost::python;

namespace backend_optimizer
{

class backendOptimizer
{

public:

  backendOptimizer();
  int new_graph(py::list nodes, py::list edges, int fixed_node);
  void add(py::list nodes, py::list edges);
  void optimize();
  py::list get_optimized();

private:

};

class Edge
{

};

} // namespace backend_optimizer

#endif // backendOptimizer_H
