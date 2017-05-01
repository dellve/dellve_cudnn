import dellve
import dellve_cudnn_benchmark as dcb
import time
from abc import abstractmethod
from helper import problem_size as ps, gpu_info
from StressFactory import StressToolFactory

class ForwardActivationStressTool(StressToolFactory): 
    name = 'ForwardActivationStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the activation\noperation"
                   " in cuDNN library with forward propagation."
                  )

    def get_controller(self):
        n,c,h,w = ps.calculate_nchw_forward(self.gpu_id, self.mem_util)
        return dcb.activation_forward(w,h,c,n)

    @classmethod
    def init_config(cls):
        available_gpus = gpu_info.get_valid_gpus()
        super(ForwardActivationStressTool, cls).config['gpu'] = available_gpus[0]
        super(ForwardActivationStressTool, cls).schema['properties']['gpu']['enum'] = available_gpus


class BackwardActivationStressTool(StressToolFactory): 
    name = 'BackwardActivationStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the activation\noperation"
                   " in cuDNN library with backward propagation."
                  )

    def get_controller(self):
        n,c,h,w = ps.calculate_nchw_activation_backward(self.gpu_id, self.mem_util)
        return dcb.activation_backward(w,h,c,n)

class ForwardSoftmaxStressTool(StressToolFactory): 
    name = 'ForwardSoftmaxStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the softmax\noperation"
                   " in cuDNN library with forward propagation."
                  )

    def get_controller(self):
        n,c,h,w = ps.calculate_nchw_forward(self.gpu_id, self.mem_util)
        return dcb.softmax_forward(w,h,c,n,"fast")

class BackwardSoftmaxStressTool(StressToolFactory): 
    name = 'BackwardSoftmaxStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the softmax\noperation"
                   " in cuDNN library with backward propagation."
                  )

    def get_controller(self):
        n,c,h,w = ps.calculate_nchw_softmax_backward(self.gpu_id, self.mem_util)
        return dcb.softmax_backward(w,h,c,n,"fast")

    
class ForwardPoolingStressTool(StressToolFactory): 
    name = 'ForwardPoolingStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the pooling\noperation"
                   " in cuDNN library with forward propagation."
                  )

    def get_controller(self):
        win = 3
        pad = 1
        stride = 1
        n,c,h,w = ps.calculate_nchw_pooling(self.gpu_id, self.mem_util,win,pad,stride)

        controller = dcb.pooling_forward(w, h, c, n, win, win, pad, pad, stride, stride, "max")
        return controller

class BackwardPoolingStressTool(StressToolFactory): 
    name = 'BackwardPoolingStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the pooling\noperation"
                   " in cuDNN library with backward propagation."
                  )

    def get_controller(self):
        win = 3
        pad = 1
        stride = 1
        n,c,h,w = ps.calculate_nchw_pooling(self.gpu_id, self.mem_util,win,pad,stride)

        controller = dcb.pooling_backward(w, h, c, n, win, win, pad, pad, stride, stride, "max")
        return controller

class ForwardConvolutionStressTool(StressToolFactory):
    name = 'ForwardConvolutionStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the convolution\noperation"
                   " in cuDNN library with forward propagation."
                  )

    def get_controller(self):
        win = 3
        pad = 1
        stride = 1
        k = 3
        n,c,h,w = ps.calculate_nchw_convolution(self.gpu_id, self.mem_util,k,win,pad,stride)
        
        controller = dcb.convolution_forward(w, h, c, n, k, win, win, pad, pad, stride, stride)
        return controller

class BackwardConvolutionDataStressTool(StressToolFactory):
    name = 'BackwardConvolutionDataStressTool'
    description = ("Stresses the GPU to 100% GPU Utilization using the convolution\noperation"
                   " in cuDNN library with backward propagation."
                  )

    def get_controller(self):
        win = 3
        pad = 1
        stride = 1
        k = 3
        n,c,h,w = ps.calculate_nchw_convolution(self.gpu_id, self.mem_util,k,win,pad,stride)
        
        controller = dcb.convolution_backward_data(w, h, c, n, k, win, win, pad, pad, stride, stride)
        return controller

