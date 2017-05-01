#ifndef DELLVE_CUDNN_CONVOLUTION_H_
#define DELLVE_CUDNN_CONVOLUTION_H_

#include "dellve_cudnn_benchmark.hpp"
#include "dellve_tensor_curand_helper.hpp"

#include "CuDNN/Convolution.hpp"
#include "CuDNN/Handle.hpp"
#include "CuDNN/Status.hpp"
#include "CuDNN/Tensor.hpp"		   
#include "CuDNN/Filter.hpp"
#include "CuDNN/ConvolutionFwdAlgo.hpp"
#include "CuDNN/ConvolutionBwdDataAlgo.hpp"

#include <iostream>

namespace DELLve {
    namespace Convolution {
		
        /**
         * CuDNN Convolution Forward
         *
         * Build 4D tensors using NCHW and KCRS provided for input and filter
         * respectively. Then, create the output tensor by calculating 
         * the forward output dimensions of convolution. Finally, set up the
         * workspace required and return the function that will run the
         * operation with forward propagation.
         *
         * @param w - Width of input image
         * @param h - Height of input image
         * @param c - Number of channels per input image
         * @param n - Number of input images
         * @param k - Number of filters
         * @param r - Height of filter
         * @param s - Width of filter
         * @param padW - Width of zero padding
         * @param padH - Height of zero padding
         * @param strideV -  Filter vertical stride
         * @param strideH - Filter horizontal stride
         */
		template <typename T>
		DELLve::Benchmark forward ( int w, int h, int c, int n, int k, int r, int s, 
			int padW, int padH, int strideV, int strideH ) 
		{
			CuDNN::Handle handle;

			/**
			 * Create convolution input tensor
			 */
		    auto input = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

		    /**
		     * Create convolution filter
		     */
			auto filter = CuDNN::Filter<T>::createNCHW(k, c, r, s);
            DELLve::CurandTensor<T>::fillTensorsRand(input, filter);

			/**
			 * Create convolution descriptor
			 */
			auto convDescriptor = CuDNN::ConvolutionDescriptor::create(padH, padW, strideH, strideW);

			/**
			 * Calculate convolution output dimensions
			 */
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
		   	CUDNN_CHECK_STATUS (
		   		cudnnGetConvolution2dForwardOutputDim (
		   			convDescriptor,
		   			input.getDescriptor(),
		   			filter.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
		   		)
		   	);

		   	/**
		   	 * Create output tensor
		   	 */
		   	auto output = CuDNN::Tensor<T>::createNCHW(outputDims);

            CuDNN::ConvolutionFwdAlgo algorithm;
		   	CUDNN_CHECK_STATUS (
		   		cudnnGetConvolutionForwardAlgorithm ( 
		   			handle,
					input.getDescriptor(),
					filter.getDescriptor(),
					convDescriptor,
					output.getDescriptor(),
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
					0,
					&algorithm )
		   	);

		   	/**
		   	 * Create workspace buffer
		   	 */
		   	auto workspace = CuDNN::Convolution::
		   		createForwardWorkspace<T> ( 
		   			handle, 		
			   		input.getDescriptor(), 
			   		filter.getDescriptor(), 
			   		convDescriptor, 
			   		output.getDescriptor(), 
			   		algorithm 
			);

		   	/**
		   	 * Retun new benchmark
		   	 */
			return [=]() {
				return cudnnConvolutionForward (
					handle,
					&CuDNN::Constants::alpha,
					input.getDescriptor(),
					input,
					filter.getDescriptor(),
					filter,
					convDescriptor,
					algorithm,
					workspace,
					workspace.getSize(),
					&CuDNN::Constants::beta,
					output.getDescriptor(),
					output
				);
			};
		}

        /**
         * CuDNN Convolution Backward Data
         *
         * Build 4D tensors using NCHW and KCRS provided for input and filter
         * respectively. Then, create the output tensor by calculating 
         * the forward output dimensions of convolution. Finally, set up the
         * workspace required and return the function that will run the
         * operation with backward propagation respective to data.
         * 
         * @see See forward for parameter details
         */
		template <typename T>
		DELLve::Benchmark backwardData ( int w, int h, int c, int n, int k, int r, int s, 
			int padW, int padH, int strideW, int strideH ) 
		{
			CuDNN::Handle handle;

			/**
			 * Create convolution input tensor
			 */
		    auto input = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

		    /**
		     * Create convolution filter
		     */
			auto filter = CuDNN::Filter<T>::createNCHW(k, c, r, s);

			/**
			 * Create convolution descriptor
			 */
			auto convDescriptor = CuDNN::ConvolutionDescriptor::create(padH, padW, strideH, strideW);

			/**
			 * Calculate convolution output dimensions
			 */
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
		   	CUDNN_CHECK_STATUS (
		   		cudnnGetConvolution2dForwardOutputDim (
		   			convDescriptor,
		   			input.getDescriptor(),
		   			filter.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
		   		)
		   	);

		   	/**
		   	 * Create output tensor
		   	 */
		   	auto output = CuDNN::Tensor<T>::createNCHW(outputDims);
            DELLve::CurandTensor<T>::fillTensorsRand(output, filter);

            CuDNN::ConvolutionBwdDataAlgo algorithm;
		   	CUDNN_CHECK_STATUS (
		   		cudnnGetConvolutionBackwardDataAlgorithm (
		   			handle,
					filter.getDescriptor(),
					output.getDescriptor(),
					convDescriptor,
					input.getDescriptor(),
					CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
					0,
					&algorithm )
		   	);

		   	/**
		   	 * Create workspace buffer
		   	 */
		   	auto workspace = CuDNN::Convolution::
		   		createBackwardDataWorkspace<T> ( 
		   			handle, 		
			   		input.getDescriptor(), 
			   		filter.getDescriptor(), 
			   		convDescriptor, 
			   		output.getDescriptor(), 
			   		algorithm 
			);

		   	/**
		   	 * Retun new benchmark
		   	 */
			return [=]() {
				return cudnnConvolutionBackwardData (
					handle,
					&CuDNN::Constants::alpha,
					filter.getDescriptor(),
					filter,
					output.getDescriptor(),
					output,
					convDescriptor,
					algorithm,
					workspace,
					workspace.getSize(),
					&CuDNN::Constants::beta,
					input.getDescriptor(),
					input 
				);
			};
		}

        /**
         * CuDNN Convolution Backward Filter
         *
         * Build 4D tensors using NCHW and KCRS provided for input and filter
         * respectively. Then, create the output tensor by calculating 
         * the forward output dimensions of convolution. Finally, set up the
         * workspace required and return the function that will run the
         * operation with backward propagation respective to filter.
         * 
         * @see See forward for parameter details
         */
		template <typename T>
		DELLve::Benchmark backwardFilter ( int w, int h, int c, int n, int k, int r, int s, 
			int padW, int padH, int strideW, int strideH ) 
		{
			CuDNN::Handle handle;

			/**
			 * Create convolution input tensor
			 */
		    auto input = CuDNN::Tensor<T>::createNCHW(n, c, h, w);

		    /**
		     * Create convolution filter
		     */
			auto filter = CuDNN::Filter<T>::createNCHW(k, c, r, s);

			/**
			 * Create convolution descriptor
			 */
			auto convDescriptor = CuDNN::ConvolutionDescriptor::create(padH, padW, strideH, strideW);

			/**
			 * Calculate convolution output dimensions
			 */
		   	std::tuple<int,int,int,int> outputDims; // NCHW tuple of output dimensions
		   	CUDNN_CHECK_STATUS (
		   		cudnnGetConvolution2dForwardOutputDim (
		   			convDescriptor,
		   			input.getDescriptor(),
		   			filter.getDescriptor(),
		   			&std::get<0>(outputDims),	// N
		   			&std::get<1>(outputDims),	// C
		   			&std::get<2>(outputDims),	// H
		   			&std::get<3>(outputDims)	// W
		   		)
		   	);

		   	/**
		   	 * Create output tensor
		   	 */
		   	auto output = CuDNN::Tensor<T>::createNCHW(outputDims);
            DELLve::CurandTensor<T>::fillTensorsRand({input, output});

            CuDNN::ConvolutionBwdFilterAlgo algorithm;
		   	CUDNN_CHECK_STATUS (
		   		cudnnGetConvolutionBackwardFilterAlgorithm(
		   			handle,
		   			input.getDescriptor(),
		   			output.getDescriptor(),
		   			convDescriptor,
		   			filter.getDescriptor(),
		   			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
		   			0,
					&algorithm )
		   	);

		   	/**
		   	 * Create workspace buffer
		   	 */
		   	auto workspace = CuDNN::Convolution::
		   		createBackwardFilterWorkspace<T> ( 
		   			handle, 		
			   		input.getDescriptor(), 
			   		filter.getDescriptor(), 
			   		convDescriptor, 
			   		output.getDescriptor(), 
			   		algorithm 
			);

		   	/**
		   	 * Retun new benchmark
		   	 */
			return [=]() {
				return cudnnConvolutionBackwardFilter (
					handle,
					&CuDNN::Constants::alpha,
					input.getDescriptor(),
					input,
					output.getDescriptor(),
					output,
					convDescriptor,
					algorithm,
					workspace,
					workspace.getSize(),
					&CuDNN::Constants::beta,
					filter.getDescriptor(),
					filter
				);
			};
		}
	}
}

#endif // DELLVE_CUDNN_CONVOLUTION_H_

