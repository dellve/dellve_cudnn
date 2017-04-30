#ifndef DELLVE_CUDNNHELPER_HPP_
#define DELLVE_CUDNNHELPER_HPP_

#include <iostream>

#include <cuda.h>
#include <CuDNN/Handle.hpp>

namespace DELLve {
    static bool isValidGpu(int deviceId) {
        // check first if GPU device ID is valid
        auto cudaStatus = cudaSetDevice(deviceId);
        if (cudaStatus != cudaSuccess) {
            return false;
        }

        // try and instantiate CuDNN library context
        try {
            CuDNN::Handle();
        } catch (...) {
            return false;
        }

        return true;
    }
}

#endif //DELLVE_CUDNN_HELPER_HPP_
