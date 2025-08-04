#include <stdio.h>
#include <stdlib.h>
#include "fmatrix.cuh"
#include "sgemm.cuh"

#define TILE_WIDTH 32
#define SIZE 40

int main(void)
{
  /* Allocate and initialize data on host */
  fmatrix A = fmatrix_create_on_host(TILE_WIDTH * SIZE, TILE_WIDTH * SIZE);
  fmatrix_init(A, 1.0);
  fmatrix B = fmatrix_create_on_host(TILE_WIDTH * SIZE, TILE_WIDTH * SIZE);
  fmatrix_init(B, 2.0);
  fmatrix C = fmatrix_create_on_host(TILE_WIDTH * SIZE, TILE_WIDTH * SIZE);
  fmatrix_init(C, 0.0);

  /* Allocate data on device */
  fmatrix d_A = fmatrix_create_on_device(TILE_WIDTH * SIZE, TILE_WIDTH * SIZE);
  fmatrix d_B = fmatrix_create_on_device(TILE_WIDTH * SIZE, TILE_WIDTH * SIZE);
  fmatrix d_C = fmatrix_create_on_device(TILE_WIDTH * SIZE, TILE_WIDTH * SIZE);

  /* Transfer A and B on device */
  fmatrix_data_to_device(A, d_A);
  fmatrix_data_to_device(B, d_B);
  fmatrix_data_to_device(C, d_C);

  clock_t start, end;
  float cpu_time_used;

  /* Start calculation "cpu", "gpu_basic", "gpu_tiled", "gpu_cublas" */
  /************** "cpu" *******************/
  start = clock();
  mat_mul(A, B, C, "cpu");
  end = clock();
  cpu_time_used = ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;
  printf("Time taken by CPU in milliseconds: %.2f\n", cpu_time_used);


  /* Result correctness */
  {
    float maxError = 0.0f;
    for (int i = 0; i < TILE_WIDTH * SIZE; i++){
      for (int j = 0; j < TILE_WIDTH * SIZE; j++){
        maxError = max(maxError, abs(getfm(C,i,j)- 2*TILE_WIDTH * SIZE));
      }
    }
    printf("Max error: %f\n", maxError);
  }
  fmatrix_init(C, 0.0);

  /************** "gpu_basic" *******************/
  cpu_time_used = 0;
  for (int i = 0; i<5; i++){
    fmatrix_init(C, 0.0);
    fmatrix_data_to_device(C, d_C);
    start = clock();
    mat_mul(d_A, d_B, d_C, "gpu_basic");
    end = clock();
    cpu_time_used += ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;
  }
  printf("GPU basic matrix multiplication in milliseconds : %.2f\n", cpu_time_used/5.);

  /* Retrieve the result */
  fmatrix_data_to_host(C, d_C);
  /* Result correctness */
  {
    float maxError = 0.0f;
    for (int i = 0; i < TILE_WIDTH * SIZE; i++){
      for (int j = 0; j < TILE_WIDTH * SIZE; j++){
        maxError = max(maxError, abs(getfm(C,i,j)- 2*TILE_WIDTH * SIZE));
      }
    }
    printf("Max error: %f\n", maxError);
  }
  fmatrix_init(C, 0.0);
  fmatrix_data_to_device(C, d_C);


 /************** "gpu_tiled" *******************/
 cpu_time_used = 0;
 for (int i = 0; i<5; i++){
   fmatrix_init(C, 0.0);
   fmatrix_data_to_device(C, d_C);
   start = clock();
   mat_mul(d_A, d_B, d_C, "gpu_tiled");
   end = clock();
   cpu_time_used += ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;
 }
  printf("GPU tiled matrix multiplication in milliseconds : %.2f\n", cpu_time_used/5.);

  /* Retrieve the result */
  fmatrix_data_to_host(C, d_C);
  /* Result correctness */
  {
    float maxError = 0.0f;
    for (int i = 0; i < TILE_WIDTH * SIZE; i++){
      for (int j = 0; j < TILE_WIDTH * SIZE; j++){
        maxError = max(maxError, abs(getfm(C,i,j)- 2*TILE_WIDTH * SIZE));
      }
    }
    printf("Max error: %f\n", maxError);
  }
  fmatrix_init(C, 0.0);
  fmatrix_data_to_device(C, d_C);


  /************** "gpu_cublas" *******************/
  for(int warmup = 0; warmup < 5; warmup++){
    mat_mul(d_A, d_B, d_C, "gpu_cublas");
  }
  fmatrix_init(C, 0.0);
  fmatrix_data_to_device(C, d_C);

  start = clock();
  mat_mul(d_A, d_B, d_C, "gpu_cublas");
  end = clock();
  cpu_time_used = ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;
  printf("GPU cuBLAS matrix multiplication in milliseconds : %.2f\n", cpu_time_used);

  /* Retrieve the result */
  fmatrix_data_to_host(C, d_C);
  /* Result correctness */
  {
    float maxError = 0.0f;
    for (int i = 0; i < TILE_WIDTH * SIZE; i++){
      for (int j = 0; j < TILE_WIDTH * SIZE; j++){
        maxError = max(maxError, abs(getfm(C,i,j)- 2*TILE_WIDTH * SIZE));
      }
    }
    printf("Max error: %f\n", maxError);
  }
  fmatrix_init(C, 0.0);
  fmatrix_data_to_device(C, d_C);

  /* Free */
  fmatrix_free_on_host(&A);
  fmatrix_free_on_host(&B);
  fmatrix_free_on_host(&C);
  fmatrix_free_on_device(&d_A);
  fmatrix_free_on_device(&d_B);
  fmatrix_free_on_device(&d_C);
}