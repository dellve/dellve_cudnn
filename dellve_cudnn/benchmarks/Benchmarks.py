import dellve_cudnn_benchmark as dcb
from BenchmarkFactory import BenchmarkFactory
from helper import problem_set as ps


class ForwardActivationBenchmark(BenchmarkFactory):
    name = 'ForwardActivationBenchmark'
    csv_filename = 'activation/basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the activation operation in cuDNN library with forward\n"
                   "propagation.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of each feature map\n"
                   "H = height of each feature map\n"
                   "C = number of feature maps per image\n"
                   "N = number of feature maps"
                  )

    def get_controller(self):
        return dcb.activation_forward

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class BackwardActivationBenchmark(BenchmarkFactory):
    name = 'BackwardActivationBenchmark'
    csv_filename = 'activation/basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the activation operation in cuDNN library with backward\n"
                   "propagation.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of each feature map\n"
                   "H = height of each feature map\n"
                   "C = number of feature maps per image\n"
                   "N = number of feature maps"
                  )


    def get_controller(self):
        return dcb.activation_forward

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class ForwardSoftmaxBenchmark(BenchmarkFactory):
    name = 'ForwardSoftmaxBenchmark'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the softmax operation in cuDNN library with forward\n"
                   "propagation.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of each feature map\n"
                   "H = height of each feature map\n"
                   "C = number of feature maps per image\n"
                   "N = number of feature maps"
                  )

    csv_filename = 'softmax/basic.csv'

    def get_controller(self):
        return dcb.softmax_forward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('fast')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class BackwardSoftmaxBenchmark(BenchmarkFactory):
    name = 'BackwardSoftmaxBenchmark'
    csv_filename = 'softmax/basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the softmax operation in cuDNN library with backward\n"
                   "propagation.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of each feature map\n"
                   "H = height of each feature map\n"
                   "C = number of feature maps per image\n"
                   "N = number of feature maps"
                  )


    def get_controller(self):
        return dcb.softmax_backward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('fast')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class ForwardPoolingBenchmark(BenchmarkFactory):
    # TODO: override config to set algorithm
    name = 'ForwardPoolingBenchmark'
    csv_filename = 'pooling/basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the pooling operation in cuDNN library with forward\n"
                   "propagation.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of input image\n"
                   "H = height of input image\n"
                   "C = number of channels per image\n"
                   "N = number of images\n"
                   "WIN_H = height of pooling window\n"
                   "WIN_W = width of pooling window\n"
                   "PAD_H = height of the zero padding\n"
                   "PAD_W = width of the zero padding\n"
                   "VSTRIDE = pooling vertical stride\n"
                   "HSTRIDE = pooling horizontal stride"
                  )

    def get_controller(self):
        return dcb.pooling_forward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('max')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class BackwardPoolingBenchmark(BenchmarkFactory):
    name = 'BackwardPoolingBenchmark'
    csv_filename = 'pooling/basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the pooling operation in cuDNN library with backward\n"
                   "propagation.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of each feature map\n"
                   "H = height of each feature map\n"
                   "C = number of feature maps per image\n"
                   "N = number of feature maps\n"
                   "WIN_H = height of pooling window\n"
                   "WIN_W = width of pooling window\n"
                   "PAD_H = height of the 0 padding\n"
                   "PAD_W = width of the 0 padding\n"
                   "VSTRIDE = pooling vertical stride\n"
                   "HSTRIDE = pooling horizontal stride"
                  )

    def get_controller(self):
        return dcb.pooling_backward

    def get_problem_set(self):
        result = ps.csv_get_problem_set(self.csv_filename)
        for s in result:
            s.append('max')
        return result

    def get_problem_header(self):
        header = ps.csv_get_header(self.csv_filename)
        header.append('algo')
        return header

class ForwardConvolutionBenchmark(BenchmarkFactory):
    name = 'ForwardConvolutionBenchmark'
    csv_filename = 'convolution/forward_basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the convolution operation in cuDNN library with forward\n"
                   "propagation.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of input image\n"
                   "H = height of input image\n"
                   "C = number of channels per input image\n"
                   "N = number of input images\n"
                   "K = number of filters\n"
                   "R = height of filter\n"
                   "S = width of filter\n"
                   "PAD_H = height of the zero padding\n"
                   "PAD_W = width of the zero padding\n"
                   "VSTRIDE = filter vertical stride\n"
                   "HSTRIDE = filter horizontal stride"
                  )

    def get_controller(self):
        return dcb.convolution_forward

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class BackwardConvolutionDataBenchmark(BenchmarkFactory):
    name = 'BackwardConvolutionDataBenchmark'
    csv_filename = 'convolution/backward_data_basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the convolution operation in cuDNN library with backward\n"
                   "propagation in data.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of input image\n"
                   "H = height of input image\n"
                   "C = number of channels per input image\n"
                   "N = number of input images\n"
                   "K = number of filters\n"
                   "R = height of filter\n"
                   "S = width of filter\n"
                   "PAD_H = height of the zero padding\n"
                   "PAD_W = width of the zero padding\n"
                   "VSTRIDE = filter vertical stride\n"
                   "HSTRIDE = filter horizontal stride"
                  )

    def get_controller(self):
        return dcb.convolution_backward_data

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)

class BackwardConvolutionFilterBenchmark(BenchmarkFactory):
    name = 'BackwardConvolutionFilterBenchmark'
    csv_filename = 'convolution/backward_filter_basic.csv'
    description = ("Benchmarks the GPU for different configurations testing\n"
                   "the convolution operation in cuDNN library with backward\n"
                   "propagation in filter.\n\n"
                   "----------------Parameter Description------------------\n"
                   "W = width of input image\n"
                   "H = height of input image\n"
                   "C = number of channels per input image\n"
                   "N = number of input images\n"
                   "K = number of filters\n"
                   "R = height of filter\n"
                   "S = width of filter\n"
                   "PAD_H = height of the zero padding\n"
                   "PAD_W = width of the zero padding\n"
                   "VSTRIDE = filter vertical stride\n"
                   "HSTRIDE = filter horizontal stride"
                  )

    def get_controller(self):
        return dcb.convolution_backward_filter

    def get_problem_set(self):
        return ps.csv_get_problem_set(self.csv_filename)

    def get_problem_header(self):
        return ps.csv_get_header(self.csv_filename)
