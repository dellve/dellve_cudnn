#ifndef DELLVE_CUDNNPOOLING_H_
#define DELLVE_CUDNNPOOLING_H_

#include <vector>
#include <string>

#include <cudnn.h>

#include <iostream>

#include "dellve_cudnn_benchmark.hpp"
#include "dellve_constants.hpp"
#include "dellve_tensor_curand_helper.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"
#include "CuDNN/PoolingMode.hpp"
#include "CuDNN/PoolingDescriptor.hpp"

namespace DELLve {
    namespace Pooling {
        CuDNN::PoolingMode convertMode(std::string mode) {
            // std::cout << "Setting Pooling Mode to " << mode << std::endl;
            if(mode.compare("max") == 0) {
                return CUDNN_POOLING_MAX;
            } else if (mode.compare("avgpad") == 0) {
                return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            } else if (mode.compare("avgnopad") == 0) {
                return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            } else {
                std::cout << "Unrecognized Algorithm: " << mode << std::endl;
                std::cout << "Setting to Default Pooling Mode: MAX" << std::endl;
                return CUDNN_POOLING_MAX;  
            }
        }

        /**
         * CuDNN Pooling Forward
         *
         * Build 4D tensors using NCHW provided. Fill the input with random 
         * data using the cuRAND library. Then, calculate the forward 
         * output dimensions and set up a 4d tensor for output. Finally,
         * return the function to run the operation with forward 
         * propagation.
         *
         * @param w - Width of each feature map
         * @param h - Height of each feature map 
         * @param c - Number of feature maps per image
         * @param n - Number of feature maps
         * @param winH - Height of pooling window
         * @param winW - Width of pooling window
         * @param padH - Height of the zero padding
         * @param padW - Width of the zero padding
         * @param vStride - Pooling vertical stride
         * @param hStride - Pooling horizontal stride
         * @param mode - Pooling mode ot run. Can be max, avgpad, avgnopad
         */
        template <typename T>
        DELLve::Benchmark forward(int w, int h, int c, int n, 
                                  int winH, int winW, 
                                  int padH, int padW, 
                                  int vStride, int hStride,
                                  std::string mode) {

            CuDNN::Handle handle;
            CuDNN::PoolingMode poolingMode = convertMode(mode); 
          
            CuDNN::PoolingDescriptor descriptor = CuDNN::PoolingDescriptor::create(winH, winW,
                                                                                   padH, padW,
                                                                                   hStride, wStride,
                                                                                   poolingMode);
            auto x = CuDNN::Tensor<T>::createNCHW(n,c,h,w);
            DELLve::CurandTensor<T>::fillTensorsRand({x});
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
            CUDNN_CHECK_STATUS(
                cudnnGetPooling2dForwardOutputDim(
                    descriptor,
                    x.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
                )
            );
            auto y = CuDNN::Tensor<T>::createNCHW(outputDims);
    
            return [=]() {
                return cudnnPoolingForward(
                    handle,
                    descriptor,
                    &(CuDNN::Constants::alpha),
                    x.getDescriptor(),
                    x,
                    &(CuDNN::Constants::beta),
                    y.getDescriptor(),
                    y
                );
            };
        }

        /**
         * CuDNN Pooling Backward
         *
         * Build 4D tensors using NCHW provided. Fill the input with random 
         * data using the cuRAND library. Then, calculate the backward
         * output dimensions and set up a 4d tensor for output. Finally,
         * return the function to run the operation with backward
         * propagation.
         *
         * @see See forward for parameter details
         */
        template <typename T>
        DELLve::Benchmark backward(int w, int h, int c, int n, 
                                   int winH, int winW, 
                                   int padH, int padW, 
                                   int hStride, int wStride,
                                   std::string mode) {
            CuDNN::Handle handle;
            CuDNN::PoolingMode poolingMode = convertMode(mode); 
          
            CuDNN::PoolingDescriptor descriptor = CuDNN::PoolingDescriptor::create(winH, winW,
                                                                                   padH, padW,
                                                                                   hStride, wStride,
                                                                                   poolingMode);
            auto x = CuDNN::Tensor<T>::createNCHW(n,c,h,w);
            auto dX = CuDNN::Tensor<T>::createNCHW(n,c,h,w);
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
            CUDNN_CHECK_STATUS(
                cudnnGetPooling2dForwardOutputDim(
                    descriptor,
                    x.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
                )
            );
            auto y = CuDNN::Tensor<T>::createNCHW(outputDims);
            auto dY = CuDNN::Tensor<T>::createNCHW(outputDims);
            DELLve::CurandTensor<T>::fillTensorsRand({y, dY});

            return [=]() {
                return cudnnPoolingBackward(
                    handle,
                    descriptor,
                    &(CuDNN::Constants::alpha),
                    y.getDescriptor(),
                    y,
                    dY.getDescriptor(),
                    dY,
                    x.getDescriptor(),
                    x,
                    &(CuDNN::Constants::beta),
                    dX.getDescriptor(),
                    dX
                );
            };
        }
    }
}

#endif //DELLVE_CUDNNPOOLING_H_

