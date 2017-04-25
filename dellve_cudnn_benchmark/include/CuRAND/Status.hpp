#ifndef CURAND_STATUS_HPP
#define CURAND_STATUS_HPP

#include <curand.h>
#include <iostream>

#define CURAND_CHECK_STATUS(STATUS) CuRAND::checkStatus((STATUS), __FILE__, __LINE__)

namespace CuRAND {
    typedef curandStatus_t Status;

    class Exception {
        std::string mFile;
        std::string mLine;
        Status mStatus;

    public:
    	Exception(Status status) : 
            mFile(""),
            mLine(""),
            mStatus(status) {}

        Exception(Status status, std::string file, int line) :
            mFile(file),
            mLine(std::to_string(line)),
            mStatus(status) {}

        const char* what() const noexcept {
            std::string info = (mFile.size() ? (" " + mFile) : "") + \
                               (mLine.size() ? (" " + mLine) : "");
            switch (mStatus) {
                case CURAND_STATUS_SUCCESS:
                    return std::string("CuRAND.Exception: CURAND_STATUS_SUCCESS" + info +
                            "\nCuRAND Successfully completed operation").c_str();
                case CURAND_STATUS_VERSION_MISMATCH:
                    return std::string("CuRAND.Exception: CURAND_STATUS_VERSION_MISMATCH" + info +
                            "\nVerify that curand header file matches with library version").c_str();
                case CURAND_STATUS_NOT_INITIALIZED:
                    return std::string("CuRAND.Exception: CURAND_STATUS_NOT_INITIALIZED" + info +
                            "\nCuRAND generator must be initialized before this call").c_str();
                case CURAND_STATUS_ALLOCATION_FAILED:
                    return std::string("CuRAND.Exception: CURAND_STATUS_ALLOCATION_FAILED" + info +
                            "\nFailed to allocate memory. Ensure that GPU has enough space for"
                            "the requested operation").c_str();
                case CURAND_STATUS_TYPE_ERROR:
                    return std::string("CuRAND.Exception: CURAND_STATUS_TYPE_ERROR" + info +
                            "\nVerify Generator is the same type as requested operation").c_str();
                case CURAND_STATUS_OUT_OF_RANGE:
                    return std::string("CuRAND.Exception: CURAND_STATUS_OUT_OF_RANGE" + info +
                            "\nVerify input argument is valid").c_str();
                case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                    return std::string("CuRAND.Exception: CURAND_STATUS_LENGTH_NOT_MULTIPLE" + info +
                            "\nEnsure the length requested is a multiple of dimension").c_str();
                case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                    return std::string("CuRAND.Exception: CURAND_STATUS_DOUBLE_PRECISION_REQUIRED" + info +
                            "\nRequested operation requires double precision. Double check input").c_str();
                case CURAND_STATUS_LAUNCH_FAILURE:
                    return std::string("CuRAND.Exception: CURAND_STATUS_LAUNCH_FAILURE" + info +
                            "\nKernel Error. Verify OS binary packages support CUDA version").c_str();
                case CURAND_STATUS_PREEXISTING_FAILURE:
                    return std::string("CuRAND.Exception: CURAND_STATUS_PREEXISTING_FAILURE" + info +
                            "\nPrevious kernel launch failed. Verify OS binary packages supports"
                            " CUDA version").c_str();
                case CURAND_STATUS_INITIALIZATION_FAILED:
                    return std::string("CuRAND.Exception: CURAND_STATUS_INITIALIZATION_FAILED" + info +
                            "\nInitialization of CUDA failed. Check the hardware, an appropriate"
                            " version of driver, and CUDA library are correctly installed").c_str();
                case CURAND_STATUS_ARCH_MISMATCH:
                    return std::string("CuRAND.Exception: CURAND_STATUS_ARCH_MISMATCH" + info +
                            "\nGPU architecture does not support requested operation").c_str();
                case CURAND_STATUS_INTERNAL_ERROR:
                    return std::string("CuRAND.Exception: CURAND_STATUS_INTERNAL_ERROR" + info +
                            "\nInternal cuRAND operation failed").c_str();
                default:
                    return std::string("CuRAND.Exception: CURAND_STATUS_UNKNOWN" + info).c_str();
            };
        }

        Status getStatus() const {
            return mStatus;
        }
    };

    void checkStatus(Status status) {
        if (status != CURAND_STATUS_SUCCESS) {
            Exception e = Exception(status);
            throw e;
        }
    }

    void checkStatus(Status status, std::string file, int line) {
        if(status != CURAND_STATUS_SUCCESS) {
            Exception e = Exception(status, file, line);
            throw e;
        }
    }
}

#endif // CURAND_STATUS_HPP
