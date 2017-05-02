Dellve cuDNN Tool
=================

Getting Started
---------------

Dellve CuDNN is one of the tools that are used as an extension to the main
project [Dellve Deep](https://github.com/dellve/dellve_benchend) on
GitHub. This tool utilizes NVIDIA cuDNN operations as benchmark and stress 
test tools.

You can install Dellve CuDNN from our GitHub repo:

```bash
    pip install git+https://github.com/dellve/dellve_cudnn.git
```

Install Requirements
--------------------

Dellve CuDNN requires CUDA and CuDNN before setup. 

>These environment variables must be set before installation:
>
>CUDA\_PATH: Path to CUDA library
>
>CUDNN\_PATH: Path to CUDNN library

[CUDA Setup](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#installing-cuda-development-tools)

[CuDNN Setup](https://developer.nvidia.com/cudnn)

Create API Docs
---------------

Once pip install is successful, you can build the api docs located in the docs/ folder.

```bash
    cd docs
    make html
```

This will create a sphinx document in docs/build/html/index.html
