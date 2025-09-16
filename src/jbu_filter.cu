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

// Reflect index into [0, limit-1]
__device__ inline int reflect(int idx, int limit)
{
    while (idx < 0 || idx >= limit) {
        if (idx < 0) idx = -idx;                     // mirror left side
        if (idx >= limit) idx = 2*limit - idx - 1;   // mirror right side
    }
    return idx;
}

__global__ void jbu_filter_kernel(
    const int64_t numel, 
    const float* high_res, // (B, 1, H, W)
    const float* low_res,  // (B, C, H, W)
    float* final_tensor,  // (B, C, H, W)
    int64_t p_s, 
    int64_t radius, 
    const float *gaussian_kernel, // (2*r+1, 2*r+1)
    int64_t batch, // Batch size
    int64_t channel, // Channel size
    int64_t height,  
    int64_t width, 
    int64_t high_channel, // Number of channels in guidance 
    float sigma_range)
{
    int current_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (current_id < numel)
    {
        const int w = current_id % width;
        int tmp = current_id / width;
        const int h = tmp % height;
        tmp /= height;
        const int c = tmp % channel;
        const int b = tmp / channel;

        float norm_coeff = 0.0f;
        float guidance_value = high_res[TENSORC2IDX(b, 0, h, w, high_channel, height, width)];

        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {   
                // Sample position in high-res
                int nh = h + i * p_s;
                int nw = w + j * p_s;
                // Reflect if needed
                nh = reflect(nh, height);
                nw = reflect(nw, width);

                float range_distance = guidance_value - high_res[TENSORC2IDX(b, 0, nh, nw, high_channel, height, width)];

                // Compute product of filters
                float local_coeff = gaussian_kernel[IDX2C(i + radius, j + radius, 2*radius+1)] * // Spatial filter
                expf(-range_distance * range_distance / (2.0f * sigma_range * sigma_range)); // Range filter

                // Update the pixel value
                final_tensor[TENSORC2IDX(b, c, h, w, channel, height, width)] += low_res[TENSORC2IDX(b, c, nh, nw, channel, height, width)] * local_coeff;
                norm_coeff += local_coeff;
            }
        }
        // Normalize the pixel value
        if (norm_coeff >= 1e-5f){
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
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            // Calculate distance from the center
            float x_dist = x - radius;
            float y_dist = y - radius;
            
            // Compute Gaussian function
            float value = exp(-(x_dist * x_dist + y_dist * y_dist) / (2 * sigma * sigma));
            
            // Set the value in the kernel
            kernel[x][y] = value;
        }
    }
    return kernel;
}

at::Tensor upsample(const at::Tensor& guidance, const at::Tensor& low_resolution, const int64_t p_s, const int64_t radius, const float sigma_spatial, const float sigma_range) {
    
    TORCH_CHECK(guidance.sizes()[0] == low_resolution.sizes()[0]) // Same batch size
    TORCH_CHECK(guidance.sizes()[1] == 1) // Only one channel for the high resolution image (guidance)
    TORCH_CHECK((guidance.sizes()[2] / p_s) == low_resolution.sizes()[2]) // Correct height ratio
    TORCH_CHECK((guidance.sizes()[3] / p_s) == low_resolution.sizes()[3]) // Correct wisth ratio
    TORCH_CHECK(guidance.max().item<float>() < 1.0 + 0.0001) // Check guidance are in float 

    int batch = guidance.sizes()[0];
    int channel = low_resolution.sizes()[1];
    int height = guidance.sizes()[2];
    int width = guidance.sizes()[3];

    // Check Tensors are on the GPU
    TORCH_INTERNAL_ASSERT(guidance.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(low_resolution.device().type() == at::DeviceType::CUDA);

    // Compute gaussian kernel
    at::Tensor kernel = compute_gaussian_kernel(radius, sigma_spatial);
    // Flatten and move to device
    at::Tensor gaussian_kernel = kernel.reshape({-1}).to(at::kCUDA).contiguous();

    at::Tensor guidance_contig = guidance.contiguous();
    at::Tensor low_resolution_contig = low_resolution.contiguous();
    at::Tensor result = at::zeros({batch, channel, height, width}, guidance_contig.options());
    const float* guidance_ptr = guidance_contig.data_ptr<float>();
    const float* low_resolution_ptr = low_resolution_contig.data_ptr<float>();
    const float* gaussian_kernel_ptr = gaussian_kernel.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    // bilinearly upsample low_resolution to guidance size
    at::Tensor bili_up = at::upsample_bilinear2d(low_resolution_contig, {height, width}, false);
    // make contiguous and get pointer
    at::Tensor bili_up_contig = bili_up.contiguous();
    const float* bili_up_ptr = bili_up_contig.data_ptr<float>();

    dim3 dimBlock(1024);
    dim3 dimGrid((int)(result.numel() / 1024) + 1);

    AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "jbu_filter", ([&]{
        jbu_filter_kernel <<< dimGrid, dimBlock >>> (result.numel(), guidance_ptr, bili_up_ptr, result_ptr, p_s, radius, gaussian_kernel_ptr, batch, channel, height, width, guidance.sizes()[1], sigma_range);
    }));

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Implementation fo the Joint Bilateral Upsampling (JBU) using CUDA";
    m.def("upsample", &upsample, "Upsample a low resolution image given a guidance image", 
    pybind11::arg("high_res_tensor"), pybind11::arg("low_res_tensor"), pybind11::arg("ratio / patch_size"), pybind11::arg("radius"), pybind11::arg("sigma_spatial"), pybind11::arg("sigma_range"));
}

}