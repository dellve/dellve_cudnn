#ifndef PYCUDNN_RNN_DESCRIPTOR_HPP
#define PYCUDNN_RNN_DESCRIPTOR_HPP

#include <cudnn.h>

#include "RAII.hpp" // RAII

namespace CuDNN {
    class RNNDescriptor :
        public RAII< cudnnRNNDescriptor_t,
                        cudnnCreateRNNDescriptor,
                        cudnnDestroyRNNDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_RNN_DESCRIPTOR_HPP
