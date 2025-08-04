#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuda_stuff.cuh"
#include "jbu_filter.cuh"
#include "ftensor4d.cuh"
#include "fmatrix.cuh"

__global__ void jbu_filter_kernel(float* high_res, float* low_res, float* final_tensor, int radius, float *gaussian_kernel, int batch, int channel, int height, int width, float sigma_range)
{
    int current_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channel * height * width;

    if (current_id < total)
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
                int current_i = i;
                if (h + i < 0 || h + i > height - 1)
                {
                    current_i = h - i;
                }
                int current_j = j;
                if (w + j || w + j > width - 1)
                {
                    current_j = w - j;
                }

                // Compute product of filters
                float local_coeff = gaussian_kernel[IDX2C(i + radius, j + radius, radius)] * 
                (1 / (sqrt(2 * M_PIf) * sigma_range)) * exp(-(high_res[TENSORC2IDX(b,c,h,w,channel,height,width)] - high_res[TENSORC2IDX(b,c,current_i,current_j,channel,height,width)]) * (high_res[TENSORC2IDX(b,c,h,w,channel,height,width)] - high_res[TENSORC2IDX(b,c,current_i,current_j,channel,height,width)]) / (2 * sigma_range * sigma_range));
                // Update the pixel value
                final_tensor[TENSORC2IDX(b,c,h,w,channel, height,width)] += low_res[TENSORC2IDX(b, c, current_i, current_j, channel, height, width)] * local_coeff;
                norm_coeff += local_coeff;
            }
        }
        // Normalize the pixel value
        final_tensor[TENSORC2IDX(b,c,h,w,channel, height,width)] /= norm_coeff;

    }
}

void jbu_filter_global(ftensor4d high_resolution, ftensor4d low_resolution, ftensor4d final_tensor, int radius, fmatrix gaussian_kernel, float sigma_range) {
    
    ftensor4d_assert_size(high_resolution, low_resolution);
    ftensor4d_assert_size(low_resolution, final_tensor);

    dim3 dimBlock(1024);
    dim3 dimGrid((int)(ftensor4d_elements(final_tensor) / 1024) + 1);

    jbu_filter_kernel <<< dimGrid, dimBlock >>> (high_resolution.data, low_resolution.data, final_tensor.data, radius, gaussian_kernel.data, final_tensor.batch, final_tensor.channel, final_tensor.height, final_tensor.width, sigma_range);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
