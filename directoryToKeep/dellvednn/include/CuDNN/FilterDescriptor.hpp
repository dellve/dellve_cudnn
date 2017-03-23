#ifndef PYCUDNN_FILTER_DESCRIPTOR_HPP
#define PYCUDNN_FILTER_DESCRIPTOR_HPP

#include <cudnn.h>
#include <cstdint>

#include "RAII.hpp" // RAII

namespace CuDNN {
    class FilterDescriptor :
        public RAII< cudnnFilterDescriptor_t,
                    cudnnCreateFilterDescriptor,
                    cudnnDestroyFilterDescriptor > {};
}

#endif // PYCUDNN_FILTER_DESCRIPTOR_HPP
