#ifndef sgemm_H
#define sgemm_H

#include <string>
#include "fmatrix.cuh"

void mat_mul(fmatrix A, fmatrix B, fmatrix C, std::string arg);

#endif