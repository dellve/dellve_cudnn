Getting Started
===============

Dellve CuDNN is one of the tools that are used as an extension to the main
project `Dellve Deep <https://github.com/dellve/dellve_benchend>`_ on
GitHub. This tool utilizes NVIDIA cuDNN operations as benchmark and stress 
test tools.

You can install Dellve CuDNN from our GitHub repo:

.. code-block:: bash

    pip install git+https://github.com/dellve/dellve_cudnn.git

Install Requirements
====================

Dellve CuDNN requires CUDA and CuDNN before setup. 

.. important::
    These environment variables must be set before installation:
        CUDA_PATH: Path to CUDA library

        CUDNN_PATH: Path to CUDNN library

`CUDA Setup <http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#installing-cuda-development-tools>`_

`CuDNN Setup <https://developer.nvidia.com/cudnn>`_

Dellve CuDNN Framework
======================

This framework uses a cpp to python framework
`PyBind11 <https://github.com/pybind/pybind11>`_ to create functions in cpp
and allows them to be controlled by Python side. The CPP code uses an open
source Python wrapper: `PyCuDNN <https://github.com/komarov-k/pycudnn>`_ 
to make calls to CuDNN simpler.


.. doxygenclass:: DELLve::BenchmarkController
    :project: dellve_cudnn_benchmarks
    :members:

Dellve CuDNN Operations
=======================

Following are all of the operations supported by Dellve CuDNN.

.. doxygenfunction:: DELLve::Activation::forward
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Activation::backward
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Softmax::forward
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Softmax::backward
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Convolution::forward
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Convolution::backwardData
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Convolution::backwardFilter
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Pooling::forward
    :project: dellve_cudnn_benchmarks

.. doxygenfunction:: DELLve::Pooling::backward
    :project: dellve_cudnn_benchmarks

API Reference
=============

.. automodule:: dellve_cudnn.benchmarks.BenchmarkFactory
    :members:
.. automodule:: dellve_cudnn.stresstools.StressFactory
    :members:

Contributing
============

This section is coming soon. Please check back later.

Licensing
=========

Dellve CuDNN is licensed to you under the MIT License:

.. important::
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
