#include <pybind11/pybind11.h>

#include "dellve_cudnn_helper.hpp"

namespace py = pybind11;

namespace dellve_cudnn_helper {
    PYBIND11_PLUGIN(dellve_cudnn_helper) {
        py::module m("dellve_cudnn_helper");
    
        m.def("is_valid_gpu", &DELLve::isValidGpu);
    
        return m.ptr();
    }
}
