
from setuptools import setup, find_packages

setup (
    name='StressTools',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['dellve',
                      'nvidia-ml-py'],
    entry_points='''
    [dellve.benchmarks]
    ForwardActivationStressTool=StressTools:ForwardActivationStressTool
    BackwardActivationStressTool=StressTools:BackwardActivationStressTool
    ForwardSoftmaxStressTool=StressTools:ForwardSoftmaxStressTool
    BackwardSoftmaxStressTool=StressTools:BackwardSoftmaxStressTool
    ForwardPoolingStressTool=StressTools:ForwardPoolingStressTool
    BackwardPoolingStressTool=StressTools:BackwardPoolingStressTool
    ForwardConvolutionStressTool=StressTools:ForwardConvolutionStressTool
    BackwardConvolutionDataStressTool=StressTools:BackwardConvolutionDataStressTool
    '''
)
