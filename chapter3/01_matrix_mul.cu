#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width)
{
    // Kernel code
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < Width && col < Width)
    {
        // Pvalue stores the P element that is computed by the thread
        float Pvalue = 0;
        for (int k = 0; k < Width; k++)
        {
            float Mdelement = Md[row * Width + k];
            float Ndelement = Nd[k * Width + col];
            Pvalue += Mdelement * Ndelement;
        }
        // Write the matrix to device memory; each thread writes one element
        Pd[row * Wdith + col] = Pvalue;
    }
}

void MatrixMultiplication(float *M, float *N, float *P, int Width)
{
    // Define the pointer variables that point to the global device memory region for M, N, and P matrices
    float *Md, Nd, Pd;
    int size = Width * Width * sizeof(float); // Calculate the space required for the matrices in bytes

    // Part1: Allocate device memory for matrices M, N, and P
    cudaMalloc((void **)&Md, size);
    // Transfer the M matrix from host to device memory
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&Nd, size);
    // Transfer the N matrix from host to device memory
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

    // Allocate device memory for the P matrix
    cudaMalloc((void **)&Pd, size);

    // Part2: Kernel Invocation
    // Define the block size and grid size for the kernel launch
    dim3 dimBlock(Width, Width);
    dim3 dimGrid(1, 1);

    // Launch the kernel function to perform matrix multiplication on the device
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width);

    // Part3: Copy the result matrix Pd from device to host memory
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    // Free the device memory allocated for matrices M, N, and P
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}