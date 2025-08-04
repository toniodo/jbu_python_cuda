#ifndef fmatrices_H
#define fmatrices_H
#include <stddef.h>

typedef struct {
    float* data;
    size_t cols;
    size_t rows;
} fmatrix;

/* transform matrix index to vector offset
   Since CUDA uses column major,
   nb_rows = number of rows */
#define IDX2C(i,j,nb_rows) (((j)*(nb_rows))+(i))

/* Access element (i,j) of matrix mat */
#define getfm(mat,i,j) (mat.data[IDX2C(i,j,mat.rows)])


size_t fmatrix_elements(fmatrix mat);
size_t fmatrix_size(fmatrix mat);
void fmatrix_init(fmatrix mat, float f);
/** Assert that the matrix is coherent: all fields nonzero. */
void fmatrix_assert();

fmatrix fmatrix_create_on_host(size_t rows, size_t cols);
fmatrix fmatrix_create_on_device(size_t rows, size_t cols);

void fmatrix_data_to_host(fmatrix mat_host, fmatrix mat_device);
void fmatrix_data_to_device(fmatrix mat_host, fmatrix mat_device);

void fmatrix_free_on_host(fmatrix* mat);
void fmatrix_free_on_device(fmatrix* mat);

/** Print the first nb rows of the matrix mat
 *  on the host.
 *  If nb<0, print all rows.
 */
void fmatrix_host_print(fmatrix mat, int nb=-1);

/** Print the first nb rows of the matrix mat
 *  on the device.
 *  If nb<0, print all rows.
 */
void fmatrix_device_print(fmatrix mat, int nb=-1);

#endif