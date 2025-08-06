import torch
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load


def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = torch.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = torch.exp(-0.5 * torch.square(ax) / sig*sig)
    kernel = torch.outer(gauss, gauss)
    return kernel / torch.sum(kernel)

input_tensor = torch.rand(1, 3, 512, 512).to('cuda')
high_res = torch.rand(1, 1, 512, 512).to('cuda')

sigma_s = 0.2
radius = 3
sigma_r = 0.5

gaussian_kernel = gkern(2*radius+1, sigma_s).to('cuda')

load(
    name="jbu_filter",
    sources=["jbu_filter.cu"],
    verbose=True,
    with_cuda=True,
    is_python_module=False
)
result = torch.ops.jbu_cuda.jbu_filter_global(high_res, input_tensor, radius, gaussian_kernel, sigma_r)
torch.cuda.empty_cache()
print(result)