#ifndef PYCUDNN_STATUS_HPP
#define PYCUDNN_STATUS_HPP

#include <cudnn.h>
#include <iostream>
#include <string>

#define CUDNN_CHECK_STATUS(STATUS) CuDNN::checkStatus((STATUS), __FILE__, __LINE__)

namespace CuDNN {
	typedef cudnnStatus_t Status;

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
				case CUDNN_STATUS_SUCCESS:
					return "CuDNN.Exception: CUDNN_STATUS_SUCCESS" + info + 
                           "\nCuDNN Successfully Completed Operation";
				case CUDNN_STATUS_NOT_INITIALIZED:
					return "CuDNN.Exception: CUDNN_STATUS_NOT_INITIALIZED" + info + 
                           "\nEnsure all parameters are initialized before calling function";
				case CUDNN_STATUS_ALLOC_FAILED:
					return "CuDNN.Exception: CUDNN_STATUS_ALLOC_FAILED" + info + 
                           "\nEnsure GPU has enough space for the memory alloc requested";
				case CUDNN_STATUS_BAD_PARAM:
					return "CuDNN.Exception: CUDNN_STATUS_BAD_PARAM" + info + 
                           "\nEnsure that all parameters being passed have valid values";
				case CUDNN_STATUS_ARCH_MISMATCH:
					return "CuDNN.Exception: CUDNN_STATUS_ARCH_MISMATCH" + info + 
                           "\nCompile and run the application on a device with appropriate"
                           " compute capability";
				case CUDNN_STATUS_MAPPING_ERROR:
					return "CuDNN.Exception: CUDNN_STATUS_MAPPING_ERROR" + info + 
                           "\nPrior to function call, unbind any previously bound textures";
				case CUDNN_STATUS_EXECUTION_FAILED:
					return "CuDNN.Exception: CUDNN_STATUS_EXECUTION_FAILED" + info + 
                           "\nCheck the hardware, an appropriate version of driver, and the"
                           " cuDNN library are correctly installed"; 
				case CUDNN_STATUS_INTERNAL_ERROR:
					return "CuDNN.Exception: CUDNN_STATUS_INTERNAL_ERROR" + info + 
                           "\nInternal cuDNN operation failed.";
				case CUDNN_STATUS_NOT_SUPPORTED:
					return "CuDNN.Exception: CUDNN_STATUS_NOT_SUPPORTED" + info + 
                           "\nRequested functionality is not supported in cuDNN. Refer to the"
                           " cuDNN docs to find version issues.";
				case CUDNN_STATUS_LICENSE_ERROR:
					return "CuDNN.Exception: CUDNN_STATUS_LICENSE_ERROR" + info + 
                           "\nError can occur if requested functionality requires a license and"
                           " the license is not present or expired of if NVIDIA_LICENSE_FILE env"
                           " is not set properly";
				default:
					return "CuDNN.Exception: CUDNN_STATUS_UNKNOWN";
			};
		};

		Status getStatus() const {
			return mStatus;
		}
    };

	void checkStatus(Status status) {
      	if (status != CUDNN_STATUS_SUCCESS) {
            Exception e = Exception(status);
        	throw e;
      	}
	}

    void checkStatus(Status status, std::string file, int line) {
        if (status != CUDNN_STATUS_SUCCESS) {
            Exception e = Exception(status, file, line);
            throw e;
        }
    }
}

#endif // PYCUDNN_STATUS_HPP
