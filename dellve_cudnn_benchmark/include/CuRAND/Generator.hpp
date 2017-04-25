#ifndef CURAND_GENERATOR_HPP
#define CURAND_GENERATOR_HPP

#include <memory>

#include <curand.h>

#include "Ordering.hpp"
#include "RngType.hpp"
#include "Status.hpp"

namespace CuRAND {
    class Generator {

        struct RawGenerator {
            curandGenerator_t gen;

            RawGenerator(RngType type) {
                CURAND_CHECK_STATUS(curandCreateGenerator(&gen, type));
            }

            ~RawGenerator() {
                CURAND_CHECK_STATUS(curandDestroyGenerator(gen));
            }

            operator curandGenerator_t () {
                return gen;
            }
        };

        std::shared_ptr<RawGenerator> generatorPtr;

    protected:

        Generator(RngType type) :
            generatorPtr(std::make_shared<RawGenerator>(type)) {}

    public:

        void setOffset(unsigned long long offset) {
            CURAND_CHECK_STATUS(curandSetGeneratorOffset(*this, offset));
        }

        void setOrdering(curandOrdering_t order) {
            CURAND_CHECK_STATUS(curandSetGeneratorOrdering(*this, order));
        }

        void SetPseudoRandomGeneratorSeed(unsigned long long seed) {
            CURAND_CHECK_STATUS(curandSetPseudoRandomGeneratorSeed(*this, seed));
        } 

        void generate(unsigned int *buffer, size_t n) {
            CURAND_CHECK_STATUS(curandGenerate(*this, buffer, n));
        }
        void generate(unsigned long long *buffer, size_t n){
            CURAND_CHECK_STATUS(curandGenerateLongLong(*this, buffer, n));
        }

        void generateUniform(float *buffer, size_t n){
            CURAND_CHECK_STATUS(curandGenerateUniform(*this, buffer, n));
        }
        void generateUniform(double *buffer, size_t n){
            CURAND_CHECK_STATUS(curandGenerateUniformDouble(*this, buffer, n));
        }

        void generateNormal(float *buffer, size_t n, float mean, float stddev){
            CURAND_CHECK_STATUS(curandGenerateNormal(*this, buffer, n, mean, stddev));
        }
        
        void generateNormal(double *buffer, size_t n, double mean, double stddev){
            CURAND_CHECK_STATUS(curandGenerateNormalDouble(*this, buffer, n, mean, stddev));
        }

        void generateLogNormal(float *buffer, size_t n, float mean, float stddev){
            CURAND_CHECK_STATUS(curandGenerateLogNormal(*this, buffer, n, mean, stddev));
        }
        void generateLogNormal(double *buffer, size_t n, double mean, double stddev){
            CURAND_CHECK_STATUS(curandGenerateLogNormalDouble(*this, buffer, n, mean, stddev));
        }

        void generatePoisson(unsigned int *buffer, size_t n, double lambda){
            CURAND_CHECK_STATUS(curandGeneratePoisson(*this, buffer, n, lambda));
        } 

        operator curandGenerator_t () {
            return generatorPtr->gen; 
        }
    };
}

#endif // CURAND_GENERATOR_HPP
