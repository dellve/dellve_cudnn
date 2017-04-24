#ifndef DELLVE_TENSOR_CURAND_HELPER_HPP_
#define DELLVE_TENSOR_CURAND_HELPER_HPP_

#include "CuRAND/CuRAND.hpp"
#include "CuDNN/Tensor.hpp"
#include "CuDNN/Filter.hpp"
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

            void fillFilterRand(CuDNN::Filter<T> filter) {
                mGen.generateUniform(filter, filter.getSize());
            }

            static void fillTensorsRand(std::initializer_list<CuDNN::Tensor<T>> tensorList) {
                CurandTensor<T> ct;

                for (auto elem : tensorList) {
                    ct.fillTensorRand(elem);
                }
            }

            static void fillTensorsRand(CuDNN::Tensor<T> tensor, CuDNN::Filter<T> filter) {
                CurandTensor<T> ct;

                ct.fillTensorRand(tensor);
                ct.fillFilterRand(filter);
            }
    };
}

#endif //DELLVE_TENSOR_CURAND_HELPER_HPP_
