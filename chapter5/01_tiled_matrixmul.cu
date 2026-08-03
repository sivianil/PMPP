#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int m, int n, int o)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o)
    {
        float Pvalue = 0;
        for (int k = 0; k < n; k++)
        {
            Pvalue += Md[row * n + k] * Nd[k * o + col];
        }
        Pd[row * o + col] = Pvalue;
    }
}

__global__ void TiledMatrixMulKernel(float *Md, float *Nd, float *Pd, int m, int n, int o)
{
    // Declare shared memory variables
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // Get the block indexes, thread indexes across x and y dimensions
    int bx = blockIndex.bx;
    int by = blockIndex.by;
    int tx = threadIndex.tx;
    int ty = threadIndex.ty;

    // Identify the row & column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    // Iterate over the Md, Nd tiles to compute the Pd element
    // No of iterations = No of Phases = Width / TILE_WIDTH
    for (int m = 0; m < n / TILE_WIDTH; m++)
    {
        if (Row < m && (m * TILE_WIDTH + tx) < n)
        {
            // Threads in collaboration to load Md & Nd elements by accessing global memory once into shared memory
            Mds[ty][tx] = Md[Row * n + (m * TILE_WIDTH + tx)];
        }
        else
        {
            Mds[ty][tx] = 0.0f; // Padding with 0.0f for threads that are out of bounds
        }

        if ((m * TILE_WIDTH + ty) < n && Col < o)
        {
            Nds[ty][tx] = Nd[(m * TILE_WIDTH + ty) * o + Col];
        }
        else
        {
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads(); // to make sure that all threads in a block have finished the loading Md & Nd tiles unto Mds, Nds

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
            __syncthreads(); // All threads of the block have completed using Mds, Nds contents before loops into the next iteration and loads the next Md, Nd tiles
        }
    }

    Pd[Row * o + Col] = Pvalue;
}

void MatrixMul(float *M, float *N, float *P, int m, int n, int o)
{
    float *Md, *Nd, *Pd;

    cudaMalloc((void **)&Md, m * n * sizeof(float));
    cudaMalloc((void **)&Nd, n * o * sizeof(float));
    cudaMalloc((void **)&Pd, m * o * sizeof(float));

    cudaMemcpy(Md, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, m, n, o);

    cudaMemcpy(P, Pd, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}

void MatrixMulTiling(float *M, float *N, float *P, int m, int n, int o)
{

    float *Md, *Nd, *Pd;

    cudaMalloc((void **)&Md, m * n * sizeof(float));
    cudaMalloc((void **)&Nd, n * o * sizeof(float));
    cudaMalloc((void **)&Pd, m * o * sizeof(float));

    cudaMemcpy(Md, M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, n * o * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, m, n, o);

    cudaMemcpy(P, Pd, m * o * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}