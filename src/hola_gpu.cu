#include <cstdio>
#include <cuda_runtime.h>

__global__ void cuadrados(int *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = i * i;
}

int main() {
    const int n = 10;
    int h_a[n];                         // h_ = host (CPU)
    int *d_a;                           // d_ = device (GPU)
    cudaMalloc(&d_a, n * sizeof(int));

    cuadrados<<<1, 16>>>(d_a, n);       // 1 bloque, 16 hilos

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%d ", h_a[i]);
    printf("\n");
    cudaFree(d_a);
}
