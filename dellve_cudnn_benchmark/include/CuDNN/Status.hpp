#ifndef PYCUDNN_STATUS_HPP
#define PYCUDNN_STATUS_HPP

#include <cudnn.h>
#include <iostream>

namespace CuDNN {
	typedef cudnnStatus_t Status;

    class Exception {
    	Status mStatus;
    public:
    	Exception(Status status) : mStatus(status) {}
      
		const char* what() const noexcept {
            std::string msg = "";
			switch (mStatus) {
				case CUDNN_STATUS_SUCCESS:
					return "CuDNN.Exception: CUDNN_STATUS_SUCCESS\n"
                           "CuDNN Successfully Completed Operation";
				case CUDNN_STATUS_NOT_INITIALIZED:
					return "CuDNN.Exception: CUDNN_STATUS_NOT_INITIALIZED\n"
                           "Ensure all parameters are initialized before calling function";
				case CUDNN_STATUS_ALLOC_FAILED:
					return "CuDNN.Exception: CUDNN_STATUS_ALLOC_FAILED\n"
                           "Ensure GPU has enough space for the memory alloc requested";
				case CUDNN_STATUS_BAD_PARAM:
					return "CuDNN.Exception: CUDNN_STATUS_BAD_PARAM\n"
                           "Ensure that all parameters being passed have valid values";
				case CUDNN_STATUS_ARCH_MISMATCH:
					return "CuDNN.Exception: CUDNN_STATUS_ARCH_MISMATCH\n"
                           "Compile and run the application on a device with appropriate"
                           " compute capability";
				case CUDNN_STATUS_MAPPING_ERROR:
					return "CuDNN.Exception: CUDNN_STATUS_MAPPING_ERROR\n"
                           "Prior to function call, unbind any previously bound textures";
				case CUDNN_STATUS_EXECUTION_FAILED:
					return "CuDNN.Exception: CUDNN_STATUS_EXECUTION_FAILED\n"
                           "Check the hardware, an appropriate version of driver, and the"
                           " cuDNN library are correctly installed"; 
				case CUDNN_STATUS_INTERNAL_ERROR:
					return "CuDNN.Exception: CUDNN_STATUS_INTERNAL_ERROR\n"
                           "Internal cuDNN operation failed.";
				case CUDNN_STATUS_NOT_SUPPORTED:
					return "CuDNN.Exception: CUDNN_STATUS_NOT_SUPPORTED\n"
                           "Requested functionality is not supported in cuDNN. Refer to the"
                           " cuDNN docs to find version issues.";
				case CUDNN_STATUS_LICENSE_ERROR:
					return "CuDNN.Exception: CUDNN_STATUS_LICENSE_ERROR\n"
                           "Error can occur if requested functionality requires a license and"
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
            std::cout << e.what() << std::endl;
        	throw e;
      	}
	}
}

#endif // PYCUDNN_STATUS_HPP
