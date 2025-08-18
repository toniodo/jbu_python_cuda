from setuptools import setup, find_packages  
import pybind11
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(  
    name='jbu_cuda',
    version='1.0',  
    packages=find_packages(),  
    install_requires=['torch'],
    ext_modules=[  
        CUDAExtension(  
            name='jbu_cuda',  
            sources=['jbu_filter.cu'],  
            include_dirs=[pybind11.get_include()],
        ) 
    ],  
    cmdclass={  
        'build_ext': BuildExtension  
    }  
)