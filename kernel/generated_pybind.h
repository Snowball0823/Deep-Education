//#include <pybind11/pybind11.h>
//#include "dlpack.h"
//#include "kernel.h"
//#include "csr.h"

//namespace py = pybind11;

inline void export_kernel(py::module &m) { 
    m.def("spmm",
        [](graph_t& graph, py::capsule& input, py::capsule& output, bool reverse, bool norm, int64_t thread_number) {
            array2d_t<float> input_array = capsule_to_array2d(input);
            array2d_t<float> output_array = capsule_to_array2d(output);
            return invoke_spmm(graph, input_array, output_array, reverse, norm, thread_number);
        }
    );
}
