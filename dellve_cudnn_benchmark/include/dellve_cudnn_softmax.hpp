#ifndef DELLVE_CUDNNSOFTMAX_H_
#define DELLVE_CUDNNSOFTMAX_H_

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
#include "CuDNN/SoftmaxAlgorithm.hpp"

namespace DELLve {
    namespace Softmax {
        CuDNN::SoftmaxAlgorithm convAlgorithm(std::string alg) {
            // std::cout << "Setting Softmax Algorithm to " << alg << std::endl;
            if(alg.compare("fast") == 0) {
                return CUDNN_SOFTMAX_FAST;
            } else if (alg.compare("accurate") == 0) {
                return CUDNN_SOFTMAX_ACCURATE;
            } else if (alg.compare("log") == 0) {
                return CUDNN_SOFTMAX_LOG;
            } else {
                std::cout << "Unrecognized Algorithm: " << alg << std::endl;
                std::cout << "Setting to Default Softmax Algorithm: FAST" << std::endl;
                return CUDNN_SOFTMAX_FAST;  
            } 
             
        }

        /**
         * CuDNN Softmax Forward
         *
         * Build 4D tensors using NCHW provided. Fill the input with random 
         * data using the cuRAND library. Then, set up the function that 
         * will run the operation with forward propagation with algorithm
         * specified.
         *
         * @param w - Width of each feature map
         * @param h - Height of each feature map 
         * @param c - Number of feature maps per image
         * @param n - Number of feature maps
         * @param alg - Algorithm to run. Can be fast, accurate, or log
         */
        template <typename T>
        DELLve::Benchmark forward(int w, int h, int c, int n, std::string alg) {
            CuDNN::Handle handle;
            auto algorithm = convAlgorithm(alg);
            auto x = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            DELLve::CurandTensor<T>::fillTensorsRand({x});

            return [=]() {
                return cudnnSoftmaxForward(handle,
                                   	  	   algorithm,
                                   	  	   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   	  	   &CuDNN::Constants::alpha,
                                   	  	   x.getDescriptor(),
                                   	  	   x,
                                   	  	   &CuDNN::Constants::beta,
                                   	  	   y.getDescriptor(),
                                   	  	   y);
            };
        }

        /**
         * CuDNN Softmax Backward
         *
         * Build 4D tensors using NCHW provided. Fill the input with random 
         * data using the cuRAND library. Then, set up the function that 
         * will run the operation with backward propagation with algorithm
         * specified.
         *
         * @see See forward for parameter details
         */
        template <typename T>
        DELLve::Benchmark backward(int w, int h, int c, int n, std::string alg) {
            CuDNN::Handle handle;
            auto algorithm = convAlgorithm(alg);
            auto dX = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto dY = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            DELLve::CurandTensor<T>::fillTensorsRand({y,dY});

            return [=]() {
                return cudnnSoftmaxBackward(handle,
                                            algorithm,
                                            CUDNN_SOFTMAX_MODE_CHANNEL,
                                            &CuDNN::Constants::alpha,
                                            y.getDescriptor(),
                                            y,
                                            dY.getDescriptor(),
                                            dY,
                                            &CuDNN::Constants::beta,
                                            dX.getDescriptor(),
                                            dX);
            };
        }
    };
};

#endif // DELLVE_CUDNNSOFTMAX_H_

