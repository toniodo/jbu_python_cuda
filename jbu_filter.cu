#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "cuda_stuff.cuh"

#define TENSORC2IDX(b, c, h, w, n_channel, height, width) (((b*n_channel + c) * height + h) * width + w)
#define IDX2C(i,j,nb_rows) (((j)*(nb_rows))+(i))

namespace jbu_cuda {

__global__ void jbu_filter_kernel(const int64_t numel, const float* high_res, const float* low_res, float* final_tensor, int64_t radius, const float *gaussian_kernel, int64_t batch, int64_t channel, int64_t height, int64_t width, float sigma_range)
{
    int current_id = blockIdx.x * blockDim.x + threadIdx.x;


    if (current_id < numel)
    {
        int w = current_id % width;
        int tmp = current_id / width;
        int h = tmp % height;
        tmp /= height;
        int c = tmp % channel;
        int b = tmp / channel;

        float norm_coeff = 0.0;

        //printf("Current kernel: %d, %d, %d, %d\n", b, c, h, w);

        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                int current_i = h + i;
                if (current_i < 0) {
                    current_i = -current_i; // Reflect back
                } else if (current_i >= height) {
                    current_i = 2 * height - current_i - 1; // Reflect back
                }

                int current_j = w + j;
                if (current_j < 0) {
                    current_j = -current_j; // Reflect back
                } else if (current_j >= width) {
                    current_j = 2 * width - current_j - 1; // Reflect back
                }

                // Compute product of filters
                float local_coeff = gaussian_kernel[IDX2C(i + radius, j + radius, 2*radius+1)] * 
                (1 / (sqrt(2 * M_PI) * sigma_range)) * exp(-(high_res[TENSORC2IDX(b,0,h,w,channel,height,width)] - high_res[TENSORC2IDX(b,0,current_i,current_j,channel,height,width)]) * (high_res[TENSORC2IDX(b,0,h,w,channel,height,width)] - high_res[TENSORC2IDX(b,0,current_i,current_j,channel,height,width)]) / (2 * sigma_range * sigma_range));
                // Update the pixel value
                final_tensor[TENSORC2IDX(b,c,h,w,channel, height,width)] += low_res[TENSORC2IDX(b, c, current_i, current_j, channel, height, width)] * local_coeff;
                norm_coeff += local_coeff;
            }
        }
        // Normalize the pixel value
        if (norm_coeff != 0){
            final_tensor[TENSORC2IDX(b,c,h,w,channel, height,width)] /= norm_coeff;
        }
    }
}

at::Tensor jbu_filter_global(const at::Tensor& high_resolution, const at::Tensor& low_resolution, const int64_t radius, const at::Tensor& gaussian_kernel, const double sigma_range) {
    
    TORCH_CHECK(high_resolution.sizes()[0] == low_resolution.sizes()[0]) // Same batch size
    TORCH_CHECK(high_resolution.sizes()[1] == 1) // Only one channel for the high resolution image
    TORCH_CHECK(high_resolution.sizes()[2] == low_resolution.sizes()[2]) // Same height size
    TORCH_CHECK(high_resolution.sizes()[3] == low_resolution.sizes()[3]) // Same width size
    int batch = high_resolution.sizes()[0];
    int channel = low_resolution.sizes()[1];
    int height = high_resolution.sizes()[2];
    int width = high_resolution.sizes()[3];

    TORCH_INTERNAL_ASSERT(high_resolution.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(low_resolution.device().type() == at::DeviceType::CUDA);
    at::Tensor high_resolution_contig = high_resolution.contiguous();
    at::Tensor low_resolution_contig = low_resolution.contiguous();
    at::Tensor gaussian_kernel_contig = gaussian_kernel.contiguous();
    at::Tensor result = at::zeros({batch, channel, height, width}, high_resolution_contig.options());
    const float* high_resolution_ptr = high_resolution_contig.data_ptr<float>();
    const float* low_resolution_ptr = low_resolution_contig.data_ptr<float>();
    const float* gaussian_kernel_ptr = gaussian_kernel_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    dim3 dimBlock(1024);
    dim3 dimGrid((int)(result.numel() / 1024) + 1);
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "jbu_filter", ([&]{
        jbu_filter_kernel <<< dimGrid, dimBlock >>> (result.numel(), high_resolution_ptr, low_resolution_ptr, result_ptr, radius, gaussian_kernel_ptr, batch, channel, height, width, sigma_range);
    }));

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    return result;
}

// Registers CUDA implementations
TORCH_LIBRARY(jbu_cuda, m) {
   // Note that "float" in the schema corresponds to the C++ double type
   // and the Python float type.
   m.def("jbu_filter_global(Tensor high_res, Tensor low_res, int r, Tensor gaussian_kernel, float sigma_r) -> Tensor");
}

TORCH_LIBRARY_IMPL(jbu_cuda, CUDA, m) {
  m.impl("jbu_filter_global", &jbu_filter_global);
}

}