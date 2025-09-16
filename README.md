# Joint Bilateral Upsampling (JBU) for Python using CUDA

This repository is an implementation of the Joint Bilateral Upsampling (JBU) using CUDA. It can be installed as a Python package and used 
in a Python script with [PyTorch](https://pytorch.org/) tensors.

## Installation

### Prerequisites

The NVIDIA CUDA Compiler (NVCC) must be installed in order to build the project. This can be done by installing the CUDA Toolkit via apt or in a Conda env using the following commands:

```bash
# Via apt
sudo apt install nvidia-cuda-toolkit

# Via conda
conda install nvidia::cuda-toolkit
```

### Package installation

You can install this package in an environnement using pip by typing the following command in the main folder of the package:

```bash
pip install --use-pep517 --no-build-isolation .
```

## Usage

You can call the JBU in a Python script as following:

```python
import torch # Should be placed before the import of jbu_cuda. jbu_cuda cannot work without torch import.
import jbu_cuda

# Parameters of the JBU
sigma_s:float = 4
radius:int = 10
sigma_r:float = 0.2
downsampling_factor:int = 14

result:Tensor = jbu_cuda.upsample(guidance:Tensor, low_res:Tensor, downsampling_factor, radius, sigma_s, sigma_r)
```

The provided tensors should contain `float` between `0.0` and `1.0`, with shape `(B,C,h,w)` for the low resolution image and `(B,1,H,W)` for the guidance image.

A example script is provided in the [example/](example/) folder.

