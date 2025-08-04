#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "cuda_stuff.cuh"
#include "sgemm.cuh"
#include "fmatrix.cuh"

#define THREADS_PER_BLOCK 128
#define TILE_WIDTH 32

using namespace std;

static cublasHandle_t handle;
static int cublas_init = 0;

/* basic matrix multiplication C = alpha*A*B + beta*C on host as reference for the speedup */
void matrixMultiplication_basic_host(float alpha, fmatrix A, fmatrix B, float beta, fmatrix C) {
  float tmp = 0;
  for (int i = 0; i<A.rows; i++){
    for (int j = 0; j<B.cols; j++){
      for (int k = 0; k<A.cols; k++){
        tmp += alpha * getfm(A,i, k) * getfm(B, k, j);
      }
      getfm(C, i, j) = beta * getfm(C, i, j) + tmp;
      tmp = 0;
    }
  }
}

/* TODO : 3 different versions of matrix multiplication C = alpha*A*B + beta*C on device */
__global__
void matmul_basic_kernel(float alpha, float *A, float *B, float beta, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB) {
  /* TODO */
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nb_LigneA && j < nb_LigneB){
    float temp_c = beta * C[IDX2C(i, j, nb_LigneA)];
      for (int k = 0; k < nb_ColA; k++){
        temp_c += alpha*
        A[IDX2C(i, k, TILE_WIDTH)]*
        B[IDX2C(k, j, TILE_WIDTH)];
      }

    C[IDX2C(i, j, nb_LigneA)] += temp_c;
  }

}
void matrixMultiplication_basic(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C) {
  // TODO - declaration of dimGrid and dimBlock
  /* Initialization of the variables nbthreadbyblock and nbblockbygrid */
  int nbthreadbyblock = THREADS_PER_BLOCK;
  int nbblockbygrid = d_C.rows * d_C.cols / nbthreadbyblock + 1;
  dim3 dimBlock((int)sqrt(nbthreadbyblock), (int)sqrt(nbthreadbyblock));
  dim3 dimGrid(nbblockbygrid, nbblockbygrid);

  matmul_basic_kernel <<< dimGrid, dimBlock >>> (alpha, d_A.data, d_B.data, beta, d_C.data, d_A.cols, d_B.cols, d_A.rows, d_B.rows);
}

/**********************/
__global__
void matmul_tiled_kernel(float alpha, float *A, float *B, float beta, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB){
  /* TODO */
  // Initialize shared memory
  __shared__ float shared_A[TILE_WIDTH*TILE_WIDTH];
  __shared__ float shared_B[TILE_WIDTH*TILE_WIDTH];

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nb_LigneA && j < nb_LigneB){
    float temp_c = beta * C[IDX2C(i, j, nb_LigneA)];
    for (int m=0; m < nb_ColA / TILE_WIDTH ; m++){
      // (a) Load one data from A and B on shared memory
      shared_A[IDX2C(threadIdx.y, threadIdx.x, TILE_WIDTH)] = A[IDX2C(i, j%TILE_WIDTH + m * TILE_WIDTH, nb_LigneA)];
      shared_B[IDX2C(threadIdx.y, threadIdx.x, TILE_WIDTH)] = B[IDX2C(i%TILE_WIDTH + m * TILE_WIDTH, j, nb_LigneB)];

      // (b) Synchronization
      __syncthreads();

      // (c) Partial result computation
      for (int k = 0; k < TILE_WIDTH; k++){
        temp_c += alpha*
        shared_A[IDX2C(threadIdx.y, k, TILE_WIDTH)]*
        shared_B[IDX2C(k, threadIdx.x, TILE_WIDTH)];
      }

      // (d) Synchronization
      __syncthreads();
    }
    C[IDX2C(i, j, nb_LigneA)] += temp_c;
  }
}



void matrixMultiplication_tiled(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C){
  // TODO - declaration of dimGrid and dimBlock
  // The dimension of the blocks depends on the tile width
  int nbblockbygrid = d_C.rows / TILE_WIDTH + 1;
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(nbblockbygrid, nbblockbygrid);

  matmul_tiled_kernel <<< dimGrid, dimBlock >>> (alpha, d_A.data, d_B.data, beta, d_C.data, d_A.cols, d_B.cols, d_A.rows, d_B.rows);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

/**********************/
void matrixMultiplication_cublas(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C){
  /* TODO */
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d_A.rows, d_B.cols, d_A.cols, &alpha, d_A.data, d_A.rows, d_B.data, d_A.cols, &beta, d_C.data, d_A.rows);

  cublasDestroy(handle);
}



/*MAIN SGEMM*/
void gen_mat_mul(float alpha, fmatrix A, fmatrix B, float beta, fmatrix C, std::string arg){
    if (arg == "cpu"){
        matrixMultiplication_basic_host(alpha, A, B, beta, C);
    } else {
      /* kernel function*/
      if (arg == "gpu_basic"){
          matrixMultiplication_basic(alpha, A, B, beta, C);

      } else if (arg == "gpu_tiled"){
          matrixMultiplication_tiled(alpha, A, B, beta, C);

      } else if (arg == "gpu_cublas"){
         matrixMultiplication_cublas(alpha, A, B, beta, C);

      } else{
          printf("Matrix Multiplication argument is Wrong");
          exit(0);
      }
      // wait for everything to finish
      device_synchronize();
    }
}

void mat_mul(fmatrix A, fmatrix B, fmatrix C, std::string arg){
 gen_mat_mul(1.0, A, B, 0.0, C, arg);
}
