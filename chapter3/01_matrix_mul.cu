#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width)
{
    // Kernel code
    // CUDA uses cartersian coordinates to identify threads in a block and blocks in a grid.
    // Each thread has a unique thread index within its block, and each block has a unique block index within the grid.
    // The combination of these indices allows each thread to compute its own unique element in the output matrix.
    // (x, y, z) coordinates of the thread
    // x-axis horizontal. Moving along the x-axis corresponds to moving across columns
    // y-axis vertical. Moving along the y-axis corresponds to moving across rows
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Pvalue stores the P element that is computed by the thread
    float Pvalue = 0;
    for (int k = 0; k < Width; k++)
    {
        float Mdelement = Md[Row * Width + k];
        float Ndelement = Nd[k * Width + Col];
        Pvalue += Mdelement * Ndelement;
    }
    // Write the matrix to device memory; each thread writes one element
    Pd[Row * Wdith + Col] = Pvalue;
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
    // The dimensiones of a block are defined by TILE_WIDTH, which is a constant that specifies the number of threads in each block along the x and y dimensions.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);

    // Launch the kernel function to perform matrix multiplication on the device
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width);

    // Part3: Copy the result matrix Pd from device to host memory
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    // Free the device memory allocated for matrices M, N, and P
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}