#include <pybind11/pybind11.h>

#include "dellve_cudnn_benchmark.hpp"

#include "dellve_cudnn_activation.hpp"
#include "dellve_cudnn_convolution.hpp"
#include "dellve_cudnn_softmax.hpp"
#include "dellve_cudnn_pooling.hpp"

namespace py = pybind11;

namespace dellve_cudnn_benchmark {
    PYBIND11_PLUGIN(dellve_cudnn_benchmark) {
        py::module m("dellve_cudnn_benchmark");
    
        py::class_<DELLve::BenchmarkController>(m, "BenchmarkController")
            .def("start_benchmark", &DELLve::BenchmarkController::startBenchmark)
            .def("start_stress_tool", &DELLve::BenchmarkController::startStressTool)
            .def("get_progress", &DELLve::BenchmarkController::getProgress)
            .def("get_avg_time_micro", &DELLve::BenchmarkController::getAvgTimeMicro);
    
        DELLve::registerBenchmark(m, "activation_forward", &DELLve::Activation::forward<float>);
       	DELLve::registerBenchmark(m, "activation_backward", &DELLve::Activation::backward<float>);
        DELLve::registerBenchmark(m, "softmax_forward", &DELLve::Softmax::forward<float>);
        DELLve::registerBenchmark(m, "softmax_backward", &DELLve::Softmax::backward<float>);
        DELLve::registerBenchmark(m, "convolution_forward", &DELLve::Convolution::forward<float>);
        DELLve::registerBenchmark(m, "convolution_backward_data", &DELLve::Convolution::backwardData<float>);
        DELLve::registerBenchmark(m, "convolution_backward_filter", &DELLve::Convolution::backwardFilter<float>);
        DELLve::registerBenchmark(m, "pooling_forward", &DELLve::Pooling::forward<float>);
        DELLve::registerBenchmark(m, "pooling_backward", &DELLve::Pooling::forward<float>);
    	
        return m.ptr();
    }
}
