#ifndef DELLVE_CUDNNSOFTMAX_H_
#define DELLVE_CUDNNSOFTMAX_H_

#include <vector>

#include <cudnn.h>

#include "cudnn_helper.hpp"
#include "tensor.hpp"

enum class CudnnSoftmaxAlgorithm { FAST, ACCURATE, LOG };

class CudnnSoftmax {
private:
    CudnnHandle cudnn_handle_;

    TensorDescriptor4d<float> x_desc_;
    TensorDescriptor4d<float> y_desc_;

    cudnnSoftmaxAlgorithm_t algorithm_;

    //TODO: No need for output_dims?
    std::vector<int> output_dims_;

    const float alpha_ = 1.f;
    const float beta_ = 0.f;
public:
    CudnnSoftmax (int w, int h, int c, int n, int device, CudnnSoftmaxAlgorithm alg) : 
                  cudnn_handle_(device),
                  x_desc_(CUDNN_TENSOR_NCHW, n, c, h, w),
                  y_desc_(CUDNN_TENSOR_NCHW, n, c, h, w) {
        switch (alg) {
            case CudnnSoftmaxAlgorithm::FAST:
                algorithm_ = CUDNN_SOFTMAX_FAST;
                break;
            case CudnnSoftmaxAlgorithm::ACCURATE:
                algorithm_ = CUDNN_SOFTMAX_ACCURATE;
                break;
            case CudnnSoftmaxAlgorithm::LOG:
                algorithm_ = CUDNN_SOFTMAX_LOG;
                break;
            default:
                algorithm_ = CUDNN_SOFTMAX_ACCURATE;
        }
    }
   
    // TODO: Change SOFTMAX Algorithm and Mode dynamically
    void forward(Tensor<float> x, Tensor<float> y) {
        // Softmax Forward
        CHECK_CUDNN_ERROR(cudnnSoftmaxForward(cudnn_handle_.handle(),
                                              algorithm_,
                                              CUDNN_SOFTMAX_MODE_CHANNEL,
                                              &alpha_,
                                              x_desc_.desc(),
                                              x.begin(),
                                              &beta_,
                                              y_desc_.desc(),
                                              y.begin()));
    }
    // TODO: Change SOFTMAX Algorithm and Mode dynamically
    void backward(Tensor<float> y, Tensor<float> dY, Tensor<float> dX) {
        // Softmax Backward
        CHECK_CUDNN_ERROR(cudnnSoftmaxBackward(cudnn_handle_.handle(),
                                               algorithm_,
                                               CUDNN_SOFTMAX_MODE_CHANNEL,
                                               &alpha_,
                                               y_desc_.desc(),
                                               y.begin(),
                                               y_desc_.desc(),
                                               dY.begin(),
                                               &beta_,
                                               x_desc_.desc(),
                                               dX.begin()));
    }
};

#endif // DELLVE_CUDNNSOFTMAX_H_

