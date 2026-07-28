#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width)
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
    for (int m = 0; m < Width / TILE_WIDTH; m++)
    {
        // Threads in collaboration to load Md & Nd elements by accessing global memory once into shared memory
        Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];
        Mds[ty][tx] = Md[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads(); // to make sure that all threads in a block have finished the loading Md & Nd tiles unto Mds, Nds
    }

    for (int k = 0; k < Width; k++)
    {
        Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads(); // All threads of the block have completed using Mds, Nds contents before loops into the next iteration and loads the next Md, Nd tiles
    }

    Pd[Row * Width + Col] = Pvalue;
}