#ifndef DELLVE_TENSOR_CURAND_HELPER_HPP_
#define DELLVE_TENSOR_CURAND_HELPER_HPP_

#include "CuRAND/CuRAND.hpp"
#include "CuDNN/Tensor.hpp"
#include <initializer_list>

namespace DELLve { 
    template <typename T>
    class CurandTensor {
        CuRAND::PseudoGenerator mGen;

        public: 
            CurandTensor() :
            mGen(CuRAND::PseudoGenerator::create(CURAND_RNG_PSEUDO_XORWOW, 42ULL)) {}

            void fillTensorRand(CuDNN::Tensor<T> tensor) {
                mGen.generateUniform(tensor, tensor.getSize());
            }

            static void fillTensorsRand(std::initializer_list<CuDNN::Tensor<T>> tensorList) {
                CurandTensor<T> ct;

                for (auto elem : tensorList) {
                    ct.fillTensorRand(elem);
                }
            }
    };
}

#endif //DELLVE_TENSOR_CURAND_HELPER_HPP_
