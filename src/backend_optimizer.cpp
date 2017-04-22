#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <backend_optimizer.h>

namespace py = pybind11;
using namespace backend_optimizer;

BackendOptimizer::BackendOptimizer()
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

py::list BackendOptimizer::cumsum(py::list list_to_sum)
{
  int sum = 0;
  py::list outputs;
  for (int i = 0; i < list_to_sum.size(); i++)
  {
    sum += list_to_sum[i].cast<int>();
    outputs.append(sum);
  }
  return outputs;
}



PYBIND11_PLUGIN(backend_optimizer) {
  py::module m("backend_optimizer", "pybind11 backend_optimizer plugin");

  py::class_<BackendOptimizer>(m, "BackendOptimizer")
      .def("__init__", [](BackendOptimizer &instance) {new (&instance) BackendOptimizer();})
      .def("cumsum", &BackendOptimizer::cumsum);

  return m.ptr();
}
