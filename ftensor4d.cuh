#ifndef ftensor4d_H
#define ftensor4d_H
#include <stddef.h>

typedef struct {
    float* data;
    size_t batch;
    size_t channel;
    size_t height;
    size_t width;
} ftensor4d;

/* transform tensor index to vector offset
   Since CUDA uses column major,
   nb_rows = number of rows */

#define TENSORC2IDX(b, c, h, w, n_channel, height, width) (((b*n_channel + c) * height + h) * width + w)

/* Access element (b, c, h, w) of 4d tensor */
#define getft4d(tensor,b, c, h, w) (tensor.data[TENSORC2IDX(b,c,h,w,tensor.channel,tensor.height,tensor.width)])


size_t ftensor4d_elements(ftensor4d tensor);
size_t ftensor4d_size(ftensor4d tensor);
void ftensor4d_init(ftensor4d tensor, float f);
/** Assert that the tensor is coherent: all fields nonzero. */
void ftensor4d_assert(ftensor4d tensor);
/* Assert that the two tensors have the same shape */
void ftensor4d_assert_size(ftensor4d tensor_a, ftensor4d tensor_b);

ftensor4d ftensor4d_create_on_host(size_t batch, size_t channel, size_t height, size_t width);
ftensor4d ftensor4d_create_on_device(size_t batch, size_t channel, size_t height, size_t width);

void ftensor4d_data_to_host(ftensor4d tensor_host, ftensor4d tensor_device);
void ftensor4d_data_to_device(ftensor4d tensor_host, ftensor4d tensor_device);

void ftensor4d_free_on_host(ftensor4d* tensor);
void ftensor4d_free_on_device(ftensor4d* tensor);

/** Print the first nb rows of the matrix mat
 *  on the host.
 *  If nb<0, print all rows.
 */
void ftensor4d_host_print(ftensor4d tensor, int nb=-1);

/** Print the first nb rows of the matrix mat
 *  on the device.
 *  If nb<0, print all rows.
 */
void ftensor4d_device_print(ftensor4d tensor, int nb=-1);
#endif