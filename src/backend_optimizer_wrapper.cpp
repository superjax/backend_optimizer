#include <boost/python.hpp>

#include <string>

#include <ros/serialization.h>
#include <std_msgs/Int64.h>

#include <backend_optimizer.h>

namespace backend_optimizer
{

BOOST_PYTHON_MODULE(_backend_optimizer_wrapper_cpp)
{
  boost::python::class_<backendOptimizer>("BackendOptimizer", boost::python::init<>())
    .def("new_graph", &backendOptimizer::new_graph)
    .def("add", &backendOptimizer::add)
    .def("optimize", &backendOptimizer::optimize)
    .def("get_optimized", &backendOptimizer::get_optimized)
    ;
}

}

