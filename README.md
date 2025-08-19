# Joint Bilateral Upsampling (JBU) for Python using CUDA

This repository is an implementation of the Joint Bilateral Upsampling (JBU) using CUDA. It can be installed as a Python package and used 
in a Python script with [PyTorch](https://pytorch.org/) tensors.

## Installation

You can install this package in an environnement using pip by typing the following command in the folder of the package:

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

result:Tensor = jbu_cuda.upsample(guidance:Tensor, low_res:Tensor, radius, sigma_s, sigma_r)
```

The provided tensors should contain `float` between `0.0` and `1.0`, with shape `(B,C,H,W)` for the low resolution image (after applying a bilinear upsampling) and `(B,1,H,W)` for the guidance image.

A example script is provided in the [example/](example/) folder.

