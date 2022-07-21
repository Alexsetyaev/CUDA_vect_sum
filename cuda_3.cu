#include <math.h>
#include <stdlib.h>
#include <iostream>

__global__ void vecAdd(double* A, double* B, double* C, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    if (idx % 2 == 0) {
      for (int i = 0; i < idx; ++i) {
        C[idx] += A[idx] + B[idx];
      }
    } else {
      C[idx] = 0.0;
      while (abs(C[idx]) < 1) {
        C[idx] += A[idx] * B[idx];
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Wrong arguments" << std::endl;
    return 1;
  }
  int n = atoi(argv[1]);
  double *h_a, *h_b, *h_c;
  size_t bytes = n * sizeof(double);

  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);

  for (int i = 0; i < n; ++i) {
    h_a[i] = sin(i) * sin(i);
    h_b[i] = cos(i) * cos(i);
  }

  double *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int block_size, grid_size;
  block_size = 1024;
  grid_size = (n - 1) / block_size + 1;

  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);

  cudaEventRecord(start_gpu);

  vecAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

  cudaDeviceSynchronize();
  cudaEventRecord(stop_gpu);

  float delta = 0.0;
  cudaEventElapsedTime(&delta, start_gpu, stop_gpu);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    std::cout << h_c[i] << std::endl;
  }
  
  for (int i = 0; i < n; ++i){
          if (h_c[i] != h_a[i] + h_b[i]){
                  std::cout << "Not equal" << std::endl;
                  break;
          }
          if (i == n-1){
                  std::cout << "Equal" << std::endl;
          }
  }

  std::cout << "Elapsed time" << delta << std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
