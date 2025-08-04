#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_stuff.cuh"
#include "ftensor4d.cuh"

size_t ftensor4d_elements(ftensor4d tensor) {
    return tensor.batch*tensor.channel*tensor.height*tensor.width;
}

size_t ftensor4d_size(ftensor4d tensor) {
    return ftensor4d_elements(tensor) * sizeof(float);
}

void ftensor4d_init(ftensor4d tensor, float f) {
    for (int b = 0; b < tensor.batch; b++){
        for (int c = 0; c < tensor.channel; c++) {
            for (int h = 0; h < tensor.height; h++) {
                for (int w = 0; w < tensor.width; w++) {
                    getft4d(tensor, b, c, h, w) = f;
                }
            }
        }
    }
}


void ftensor4d_assert(ftensor4d tensor) {
    assert(tensor.data);
    assert(tensor.batch);
    assert(tensor.channel);
    assert(tensor.height);
    assert(tensor.width);
}


void ftensor4d_assert_size(ftensor4d tensor_a, ftensor4d tensor_b) {
    assert(tensor_a.batch == tensor_b.batch);
    assert(tensor_a.channel == tensor_b.channel);
    assert(tensor_a.height == tensor_b.height);
    assert(tensor_a.width == tensor_b.width);
}



ftensor4d ftensor4d_create_on_host(size_t batch, size_t channel, size_t height, size_t width) {
    assert(batch>0);
    assert(channel>0);
    assert(height>0);
    assert(width>0);
    ftensor4d tensor;
    tensor.batch = batch;
    tensor.channel = channel;
    tensor.height = height;
    tensor.width = width;
    tensor.data = (float*)malloc(ftensor4d_size(tensor));
    return tensor;
}

ftensor4d ftensor4d_create_on_device(size_t batch, size_t channel, size_t height, size_t width) {
    assert(batch>0);
    assert(channel>0);
    assert(height>0);
    assert(width>0);
    ftensor4d tensor;
    tensor.batch = batch;
    tensor.channel = channel;
    tensor.height = height;
    tensor.width = width;
    gpuErrchk(
        cudaMalloc((void **)&(tensor.data), ftensor4d_size(tensor))
    );
    return tensor;
}

void ftensor4d_data_to_device(ftensor4d tensor_host, ftensor4d tensor_device) {
    ftensor4d_assert(tensor_host);
    ftensor4d_assert(tensor_device);
    assert(tensor_host.batch==tensor_device.batch);
    assert(tensor_host.channel==tensor_device.channel);
    assert(tensor_host.height==tensor_device.height);
    assert(tensor_host.width==tensor_device.width);
    gpuErrchk(
        cudaMemcpy( tensor_device.data, tensor_host.data,
                   ftensor4d_size(tensor_host),
                   cudaMemcpyHostToDevice
                   )
        );
}

void ftensor4d_data_to_host(ftensor4d tensor_host, ftensor4d tensor_device) {
    ftensor4d_assert(tensor_host);
    ftensor4d_assert(tensor_device);
    assert(tensor_host.batch==tensor_device.batch);
    assert(tensor_host.channel==tensor_device.channel);
    assert(tensor_host.height==tensor_device.height);
    assert(tensor_host.width==tensor_device.width);
    gpuErrchk(
        cudaMemcpy( tensor_host.data, tensor_device.data,
                   ftensor4d_size(tensor_device),
                   cudaMemcpyDeviceToHost
                   )
        );
}

void ftensor4d_free_on_host(ftensor4d* tensor) {
    ftensor4d_assert(*tensor);
    free(tensor->data);
    tensor->data=0;
    tensor->batch=0;
    tensor->channel=0;
    tensor->height=0;
    tensor->width=0;
}

void ftensor4d_free_on_device(ftensor4d* tensor) {
    ftensor4d_assert(*tensor);
    gpuErrchk(cudaFree(tensor->data));
    tensor->data=0;
    tensor->batch=0;
    tensor->channel=0;
    tensor->height=0;
    tensor->width=0;
}