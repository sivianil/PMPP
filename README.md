# PMPP: Programming Massively Parallel Processors

My learning journey and code implementations from the book *Programming Massively Parallel Processors: A Hands-on Approach*. This repository is structured chapter-by-chapter, containing my custom CUDA C++ kernels, exercise solutions, and study notes.

## 🚀 Objective
To build a practical understanding of GPU architecture, memory hierarchies, and parallel computing patterns by writing and optimizing custom CUDA kernels.

## 📂 Repository Structure & Progress

- [ ] **Chapter 2: Data Parallel Computing**
  - Vector Addition
- [ ] **Chapter 3: Scalable Parallel Execution**
  - Multidimensional Grids and Matrix Multiplication
- [ ] **Chapter 4: Memory and Data Locality**
  - Tiled Matrix Multiplication
- [ ] **Chapter 5: Performance Considerations**
  - Thread Coalescing and Bank Conflicts
- [ ] **Chapter 6: Numerical Considerations**
  - IEEE Format and Precision
- [ ] **Chapter 7: Parallel Patterns - Convolution**
  - 1D and 2D Convolution kernels
- [ ] **Chapter 8: Parallel Patterns - Prefix Sum (Scan)**
  - Kogge-Stone and Brent-Kung algorithms
- [ ] **Chapter 9: Parallel Patterns - Histogram Computation**
  - Atomic Operations and Privatization
- [ ] **Chapter 10: Parallel Patterns - Sparse Matrix Computation**
  - CSR format and SpMV 

*(Add or modify chapters based on the specific edition of the book you are using).*

## 🛠️ Prerequisites

To compile and run the code in this repository, you will need:
* An NVIDIA GPU with CUDA compute capability
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed
* A C++ compiler (e.g., `gcc`, `MSVC`) compatible with your `nvcc` version

## 💻 Build Instructions

Each chapter contains its own `.cu` files. You can compile them individually using the NVIDIA CUDA Compiler (`nvcc`). 

Navigate to a specific chapter's directory and compile the code:

```bash
cd Chapter_03_Scalable_Parallel_Execution
nvcc 01_matrixmul.cu -o matrixmul
