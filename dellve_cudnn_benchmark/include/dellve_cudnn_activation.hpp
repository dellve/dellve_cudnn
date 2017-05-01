#ifndef DELLVE_CUDNNACTIVATION_HPP_
#define DELLVE_CUDNNACTIVATION_HPP_

#include <vector>
#include <string>

#include <cudnn.h>

#include <iostream>

#include "dellve_cudnn_benchmark.hpp"
#include "dellve_constants.hpp"
#include "dellve_tensor_curand_helper.hpp"
#include "CuDNN/ActivationDescriptor.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"

namespace DELLve {
    namespace Activation {
        CuDNN::ActivationDescriptor createDescriptor(void) {
            CuDNN::ActivationDescriptor descriptor;
            CUDNN_CHECK_STATUS (
                cudnnSetActivationDescriptor ( 
                    descriptor,
                    CUDNN_ACTIVATION_RELU,
                    CUDNN_NOT_PROPAGATE_NAN,
                    0.0
                )
            );

            return descriptor;
        }

        /**
         * CuDNN Activation Forward
         *
         * Build 4D tensors using NCHW provided. Fill the input with random 
         * data using the cuRAND library. Then, set up the function that 
         * will run the operation with forward propagation.
         *
         * @param w - Width of each feature map
         * @param h - Height of each feature map 
         * @param c - Number of feature maps per image
         * @param n - Number of feature maps
         */
        template <typename T>
        DELLve::Benchmark forward ( int w, int h, int c, int n ) {
	        CuDNN::Handle handle;
            auto descriptor = createDescriptor();
            auto x = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            DELLve::CurandTensor<T>::fillTensorsRand({x});

            return [=]() {
                return cudnnActivationForward (
                    handle,
                    descriptor,
                    &CuDNN::Constants::alpha,
                    x.getDescriptor(),
                    x,
                    &CuDNN::Constants::beta,
                    y.getDescriptor(),
                    y 
                );
            };	

        }

        /**
         * CuDNN Activation Backward
         *
         * Build 4D tensors using NCHW provided. Fill the input with random 
         * data using the cuRAND library. Then, set up the function that 
         * will run the operation with backward propagation.
         *
         * @see See forward for parameter details
         */
        template <typename T>
        DELLve::Benchmark backward ( int w, int h, int c, int n ) {
            CuDNN::Handle handle;
            auto descriptor = createDescriptor();
            auto x = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto y = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto dx = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            auto dy = CuDNN::Tensor<T>::createNCHW(n, c, h, w);
            DELLve::CurandTensor<T>::fillTensorsRand({y,dy});

            return [=]() {
                return cudnnActivationBackward (
                    handle,
                    descriptor,
                    &CuDNN::Constants::alpha,
                    y.getDescriptor(),
                    y,
                    dy.getDescriptor(),
                    dy,
                    x.getDescriptor(),
                    x,
                    &CuDNN::Constants::beta,
                    dx.getDescriptor(),
                    dx
                );
            };
        }
    };
};

#endif //DELLVE_CUDNNACTIVATION_HPP_

