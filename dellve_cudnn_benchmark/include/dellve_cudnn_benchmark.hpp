#ifndef DELLVE_CUDNN_BENCHMARK_HPP
#define DELLVE_CUDNN_BENCHMARK_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cudnn.h>

#include "CuDNN/Status.hpp"

#include <iostream>
#include <functional>
#include <thread>
#include <tuple>

#include <chrono>

namespace DELLve {
    namespace Functional {
        template<int ...>
        struct seq {};
        template<int N, int ...S>
        struct gens : gens<N-1, N-1, S...> {};
        template<int ...S>
        struct gens<0, S...>{ typedef seq<S...> type; };
        
        template <typename T, typename ...Args>
        class delayed {
            std::tuple<Args...> args_;
            T (*func_)(Args...);
        public:
            delayed(T (*f)(Args...), std::tuple<Args...> a) :
                args_(a), func_(f) {};
            T call() const {
                return callWithArgs(typename gens<sizeof...(Args)>::type());
            }
        private:
            template<int ...S>
            T callWithArgs(seq<S...>) const {
                return func_(std::get<S>(args_)...);
            }
        };
    }
    
    typedef std::function<CuDNN::Status(void)> Benchmark;
    typedef std::chrono::duration<long int, std::micro> usec;
    typedef std::chrono::seconds sec;
    typedef std::chrono::high_resolution_clock clock;

    /**
     * Controller class that speficies the actual runtime execution of each
     * tool in Dellve CuDNN. Provides functions that allows profiling of current
     * status of the exeuction.
     *
     * Assumes that BenchmarkDriver has been ran with the function of current
     * operation as desired.
     */
    class BenchmarkController {
        volatile float progress_;

        int currRun_;
        usec totalTimeMicro_;

    public:
        /**
         * Starts a benchmark tool.
         *
         * Sets device to desired GPU. Warms up the test by running 
         * a singular operation and synchronizing. Then, runs the operation
         * the specified number of times. Through each run, calculates 
         * time it took to run the current method and sets the progress
         * of the current run by the number of loops ran.
         *
         * This is done in a different thread so that external calls can
         * be made to this class to profile the current status.
         *
         * @param deviceId - Device ID to run test on
         * @param numRuns - Number of repeats
         */ 
        void startBenchmark(int deviceId, int numRuns) {
            progress_ = 0.0f;
            currRun_ = -1;
            totalTimeMicro_ = usec(0);

            std::thread([=](){
                cudaSetDevice(deviceId);
                Benchmark benchmark = getBenchmark();

                // warm up
                CUDNN_CHECK_STATUS(benchmark());
                cudaDeviceSynchronize();

                for (currRun_ = 1; currRun_ <= numRuns; currRun_++) {
                    auto start = clock::now();
                    CUDNN_CHECK_STATUS(benchmark());
                    cudaDeviceSynchronize();
                    auto end = clock::now();

                    totalTimeMicro_ += std::chrono::duration_cast<usec>(end - start);
                    progress_ = ((float)(currRun_)) / numRuns;
                }
                cudaDeviceSynchronize();
            }).detach();
        }

        /**
         * Starts a stress test tool.
         *
         * Sets device to desired GPU. Starts the clock and runs the
         * benchmark until desired elapsed time has been reached. Through
         * each run, sets the progress of the test by the number of seconds
         * elapsed. 
         *
         * This is also done in a separate thread so that external calls can
         * be made to this class to profile the current status.
         *
         * @param deviceId - Device ID to run test on
         * @param seconds - Number of seconds to run the test
         */
        void startStressTool(int deviceId, int seconds) {
            progress_ = 0.0f;

            std::thread([=](){
                cudaSetDevice(deviceId);
                Benchmark benchmark = getBenchmark();
                CUDNN_CHECK_STATUS(benchmark());
                cudaDeviceSynchronize();
                
                auto startTime = clock::now();
                auto endTime = clock::time_point(startTime.time_since_epoch() + sec(seconds));
                for (;;) {
                    CUDNN_CHECK_STATUS(benchmark());
                    cudaDeviceSynchronize();

                    auto currentTime = clock::now();
                    int elapsedTime = std::chrono::duration_cast<sec>(currentTime - startTime).count();
                    progress_ =  ((float) (elapsedTime)) / seconds;
                    progress_ = progress_ > 1.00f ? 1.00f : progress_; 

                    if (elapsedTime >= seconds) {
                        progress_ = 1.00f;
                        break;
                    }
                } 

                cudaDeviceSynchronize();
            }).detach();
        }

        /**
         * Returns the progress set by the tests.
         *
         * @return float - Progress of the current tool.
         */
        float getProgress() const {
            return progress_;
        }

        /**
         * Returns the average time taken to run the current problem set in
         * the benchmarks.
         *
         * Should not be called by Stress Tools.
         *
         * @return int - Average time in microsends taken for each benchmark.
         */
        int getAvgTimeMicro() const {
            int totalTimeMicro = static_cast<int>(totalTimeMicro_.count());
            return totalTimeMicro / currRun_;
        }
        
    private:
        
        virtual Benchmark getBenchmark() = 0;
        
    };
    
    template <typename ... A>
    class BenchmarkDriver : public BenchmarkController {
        
        std::tuple<A...> args_;
        Benchmark (*func_)(A...);
    
    public:
        
        BenchmarkDriver(Benchmark (*func)(A...), A ... args) :
            args_(std::make_tuple(args...)),
            func_(func) {}
        
    private:
    
        Benchmark getBenchmark() {
            return Functional::delayed<Benchmark, A...>(func_, args_).call();
        }
        
    };

	template <typename T, typename ... A>
	void registerBenchmark (T module, const char* benchmarkName, Benchmark (*factory)(A...) )
	{
        module.def(benchmarkName, [=](A ... args) {
            return std::unique_ptr<BenchmarkController>(new BenchmarkDriver<A...>(factory, args...));
		});
	}
}

#endif // DELLVE_CUDNN_BENCHMARK_HPP

