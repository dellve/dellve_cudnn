
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import os.path

cuda_path       = os.environ['CUDA_PATH']
cuda_include    = os.path.join(cuda_path, "include")
cuda_lib64      = os.path.join(cuda_path, "lib64")
cudnn_path      = os.environ['CUDNN_PATH']
cudnn_include   = os.path.join(cudnn_path, "include")
cudnn_lib64     = os.path.join(cudnn_path, "lib64")

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

        The purpose of this class is to postpone importing pybind11
        until it is actually installed, so that the ``get_include()``
        method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension('dellve_cudnn_benchmark',
        sources=[
            'dellve_cudnn_benchmark/src/dellve_cudnn_benchmark.cpp',
            ],
        include_dirs=[
            'dellve_cudnn_benchmark/include/',
            cuda_include,
            cudnn_include,
            get_pybind_include(),
            get_pybind_include(user=True),
            ],
        libraries = [
            'cuda',
            'cudart',
            'cudnn',
            'curand'
            ],
        library_dirs=[
            cuda_lib64,
            cudnn_lib64
            ],
        extra_compile_args=[
            '-std=c++11',
            '-std=c++1y'
            ],
        language='c++'
    ),

    Extension('dellve_cudnn_helper',
        sources=[
            'dellve_cudnn_benchmark/src/dellve_cudnn_helper.cpp',
            ],
        include_dirs=[
            'dellve_cudnn_benchmark/include/',
            cuda_include,
            cudnn_include,
            get_pybind_include(),
            get_pybind_include(user=True),
            ],
        libraries = [
            'cuda',
            'cudart',
            'cudnn',
            'curand'
            ],
        library_dirs=[
            cuda_lib64,
            cudnn_lib64
            ],
        extra_compile_args=[
            '-std=c++11',
            '-std=c++1y'
            ],
        language='c++'
    ),
]

setup (
    name='dellve_cudnn',
    version='0.0.0',
    packages=find_packages(),
    requires=['dellve'],
    install_requires=[
        'dellve',
        'nvidia-ml-py',
        'pybind11',
        'breathe'
    ],
    ext_modules=ext_modules,
    zip_safe=False,
    entry_points='''
    [dellve.benchmarks]
    BackwardACtivationBenchmark=dellve_cudnn:BackwardActivationBenchmark
    BackwardActivationStressTool=dellve_cudnn:BackwardActivationStressTool
    BackwardConvolutionDataBenchmark=dellve_cudnn:BackwardConvolutionDataBenchmark
    BackwardConvolutionDataStressTool=dellve_cudnn:BackwardConvolutionDataStressTool
    BackwardConvolutionFilterBenchmark=dellve_cudnn:BackwardConvolutionFilterBenchmark
    BackwardPoolingBenchmark=dellve_cudnn:BackwardPoolingBenchmark
    BackwardPoolingStressTool=dellve_cudnn:BackwardPoolingStressTool
    BackwardSoftmaxBenchmark=dellve_cudnn:BackwardSoftmaxBenchmark
    BackwardSoftmaxStressTool=dellve_cudnn:BackwardSoftmaxStressTool
    ForwardActivationBenchmark=dellve_cudnn:ForwardActivationBenchmark
    ForwardActivationStressTool=dellve_cudnn:ForwardActivationStressTool
    ForwardConvolutionBenchmark=dellve_cudnn:ForwardConvolutionBenchmark
    ForwardConvolutionStressTool=dellve_cudnn:ForwardConvolutionStressTool
    ForwardPoolingBenchmark=dellve_cudnn:ForwardPoolingBenchmark
    ForwardPoolingStressTool=dellve_cudnn:ForwardPoolingStressTool
    ForwardSoftmaxBenchmark=dellve_cudnn:ForwardSoftmaxBenchmark
    ForwardSoftmaxStressTool=dellve_cudnn:ForwardSoftmaxStressTool
    '''
)
