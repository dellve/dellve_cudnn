#ifndef PYCUDNN_FILTER_HPP
#define PYCUDNN_FILTER_HPP

#include <cudnn.h>
#include <cstdint>

#include "Buffer.hpp"
#include "Status.hpp"
#include "RAII.hpp"
#include "FilterDescriptor.hpp"

namespace CuDNN {

	template <typename T>
    class Filter {

		Buffer<T> 				mBuffer;
		FilterDescriptor<T> 	mDescriptor;
        std::vector<int>        mDims;

    	Filter ( const Buffer<T>& buffer, const FilterDescriptor<T>& descriptor,
                 std::vector<int> dims ) :
    		mBuffer(buffer),
    		mDescriptor(descriptor),
            mDims(dims) {}

    public:

		static Filter createNCHW ( int k, int c, int h, int w ) {
			return Filter ( Buffer<T>(k * c * h * w), 
							FilterDescriptor<T>::createNCHW(k, c, h, w),
                            {k, c, h, w} );
		}

		static Filter createNHWC ( int k, int c, int h, int w ) {
			return Filter ( Buffer<T>(k * c * h * w), 
							FilterDescriptor<T>::createNHWC(k, c, h, w),
                            {k, c, h, w} );
		}

        int getSize() const {
            int i;
            int sum = 0;
            for(i = 0; i < mDims.size(); i++) {
                sum += mDims[i];
            }

            return sum;
        }

		FilterDescriptor<T> getDescriptor() const {
			return mDescriptor;
		}

		operator T*() const {
			return mBuffer;
		}
	};
};

#endif // PYCUDNN_FILTER_HPP
