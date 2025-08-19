#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
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

                float range_distance = high_res[TENSORC2IDX(b, 0, h, w, channel, height, width)] - high_res[TENSORC2IDX(b, 0, current_i, current_j, channel, height, width)];

                // Compute product of filters
                float local_coeff = gaussian_kernel[IDX2C(i + radius, j + radius, 2*radius+1)]*  // Spatial filter
                (1 / (sqrt(2 * M_PI) * sigma_range)) * 
                exp(-range_distance * range_distance / (2 * sigma_range * sigma_range)); // Range filter

                // Update the pixel value
                final_tensor[TENSORC2IDX(b, c, h, w, channel, height, width)] += low_res[TENSORC2IDX(b, c, current_i, current_j, channel, height, width)] * local_coeff;
                norm_coeff += local_coeff;
            }
        }
        // Normalize the pixel value
        if (norm_coeff != 0.0){
            final_tensor[TENSORC2IDX(b, c, h, w, channel, height, width)] /=  norm_coeff;
        }
    }
}

at::Tensor compute_gaussian_kernel(int radius, float sigma) {
    // Calculate the kernel size (diameter of the kernel)
    int kernel_size = 2 * radius + 1;
    
    // Create a 2D tensor to hold the kernel
    auto kernel = at::zeros({kernel_size, kernel_size}, at::kFloat);
    
    // Compute the Gaussian kernel
    float sum = 0.0f;
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            // Calculate distance from the center
            float x_dist = x - radius;
            float y_dist = y - radius;
            
            // Compute Gaussian function
            float value = exp(-(x_dist * x_dist + y_dist * y_dist) / (2 * sigma * sigma));
            
            // Set the value in the kernel
            kernel[x][y] = value;
            
            // Accumulate for normalization
            sum += value;
        }
    }
    kernel = kernel / sum;
    return kernel;
}

at::Tensor upsample(const at::Tensor& high_resolution, const at::Tensor& low_resolution, const int64_t radius, const float sigma_spatial, const float sigma_range) {
    
    TORCH_CHECK(high_resolution.sizes()[0] == low_resolution.sizes()[0]) // Same batch size
    TORCH_CHECK(high_resolution.sizes()[1] == 1) // Only one channel for the high resolution image (guidance)
    TORCH_CHECK(high_resolution.sizes()[2] == low_resolution.sizes()[2]) // Same height size
    TORCH_CHECK(high_resolution.sizes()[3] == low_resolution.sizes()[3]) // Same width size
    TORCH_CHECK(high_resolution.max().item<float>() < 1.0 + 0.0001) // Check inputs are in float 
    TORCH_CHECK(low_resolution.max().item<float>() < 1.0 + 0.0001) // Check inputs are in float

    int batch = high_resolution.sizes()[0];
    int channel = low_resolution.sizes()[1];
    int height = high_resolution.sizes()[2];
    int width = high_resolution.sizes()[3];

    TORCH_INTERNAL_ASSERT(high_resolution.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(low_resolution.device().type() == at::DeviceType::CUDA);

    // Compute gaussian kernel
    at::Tensor kernel = compute_gaussian_kernel(radius, sigma_spatial);
    // Move to device
    at::Tensor gaussian_kernel = kernel.to(at::kCUDA);

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

    AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "jbu_filter", ([&]{
        jbu_filter_kernel <<< dimGrid, dimBlock >>> (result.numel(), high_resolution_ptr, low_resolution_ptr, result_ptr, radius, gaussian_kernel_ptr, batch, channel, height, width, sigma_range);
    }));

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Implementation fo the Joint Bilateral Upsampling (JBU) using CUDA";
    m.def("upsample", &upsample, "Upsample a low resolution image given a guidance image", 
    pybind11::arg("high_res_tensor"), pybind11::arg("low_res_tensor"), pybind11::arg("radius"), pybind11::arg("sigma_spatial"), pybind11::arg("sigma_range"));
}

}