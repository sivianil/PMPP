# Introduction to CUDA - Study Notes Template

## 📚 Book Reference

- **Book**: Programming Massively Parallel Processors (PMPP)
- **Chapter**: Introduction to CUDA
- **Date Studied**: **\*\***\_\_\_\_**\*\***
- **Review Date**: **\*\***\_\_\_\_**\*\***

---

## 1️⃣ CUDA Overview & History

### 1.1 What is CUDA?

- **Definition**:
- **Purpose**:
- ## **Key Characteristics**:
  -
  -

### 1.2 GPU vs CPU

| Aspect     | CPU | GPU |
| ---------- | --- | --- |
| Purpose    |     |     |
| Memory     |     |     |
| Throughput |     |     |
| Latency    |     |     |
| Energy     |     |     |

### 1.3 Why CUDA?

- ## **Advantages**:
  -
  -
- ## **Use Cases**:

---

## 2️⃣ CUDA Architecture Fundamentals

### 2.1 GPU Hardware Model

- **What Is a GPU?**:
  - A modern GPU (e.g H100 or B200) is a bunch of compute cores specialised in matrix multiplication (called Streaming Multiprocessors or SMs) connected to a stick of fast memory (called HBM).
  - [Abstract Layout of H100 or B200 GPU](gpu-diagram.png)

- **Device Components**:
  - SM (Streaming Multiprocessor): Each SM has a dedicated matrix multiplication core (called `Tensor Core`), a vector arithmetic unit (called `Warp Scheduler`), and a fast on-chip cache (called `SMEM/L1 Cache`).
  - A modern GPU has more than 100 SMs. H132 has 132 and B200 has 148 SMs.
  - Each SM is broken up into 4 sub-partitions (also called SM subpartitions). Each containing a Tensor Core, 16k 32-bit registers, and a SIMD/SIMT vector arithmetic unit called `Warp Scheduler`, whose lanes (ALUs) Nvidia calls `CUDA Cores`.
  - **CUDA Cores**: Set of ALUs within a each partition that do SIMD/SIMT vector arithmetic. Each ALU can only perform 1 arithmetic op each cycle e.g fp32.add.
    - Each sub-partition consists of 32 fp-32 cores and a smaller number of int32 and fp64 cores that all execute the same instruction for each cycle.
    - CUDA cores responsible for RELUs, point-wise vector operations and reductions (sums).
  - **Tensor Core (TC)**:The core component of each partition which performs matrix multiplication and makes up the vast majority of the GPU FLOPs/s. On an H100 990 bf16 TFLOPs/s from TC compared to just 66 TFLOPs/s from CUDA Cores.

### 2.2 Thread Hierarchy

- **Thread**:
  - Definition: Threads do the work in parallel. All threads execute the same instructions but with different inputs (SIMT). 
  - Characteristics: 

- **Block**:
  - Definition: Blocks are group of threads. Each block runs on a SM w/ its own shared memory. 
  - Max Threads per Block: Each block can accommodate unto 512threads. 
  - Synchronization:
    
- **Warps**:
  - Definition: Scheduling unit of what is happening inside of the GPU. Threads always execute in "warp" of 32 consecutively numbered threads each. It decreases the overhead of scheduler deciding which threads to run.
    
- **Grid**:
  - Definition: 
  - Multiple Grids:
  - Grid Dimensions:

### 2.3 Memory Hierarchy

```
[Fastest]
  └─ Registers (per thread)
  └─ Shared Memory (per block)
  └─ L1 Cache
  └─ L2 Cache
  └─ Global Memory
  └─ Constant Memory
  └─ Texture Memory
[Slowest]
```

- **Streaming MultiProcessor** (SM):
- **Registers**: Variables reside in the registers can be accessed at very high speed in a parallel manner. Allocated to individual threads, each thread can only access its own register.
  - In `NVIDIA Blackwell B200`, each SM contains a register file with 64k 32-bit registers. Each bit is 4 bytes (32 bits), yielding a total register file size of `256 KB` per SM.

| Memory Type | Scope  | Size | Speed | Cache |
| ----------- | ------ | ---- | ----- | ----- |
| Registers   | Thread |      |       |       |
| Shared      | Block  |      |       |       |
| Global      | All    |      |       |       |
| Constant    | All    |      |       |       |

---

## 3️⃣ CUDA Programming Model

### 3.1 Kernel Basics

- **What is a Kernel?**: An user-defined function that the GPU can run. To do this, add a specifier `__global__` to the function. It tells the CUDA C++ compiler, that this function runs on the GPU and can be called from the CPU code.
- **Kernel Launch Syntax**:
  ```cuda
  kernelName<<<gridDim, blockDim>>>(arguments);
  ```

### 3.2 Built-in Variables

- **Thread Index**:
  - `threadIdx.x`, `threadIdx.y`, `threadIdx.z`:
  - Usage:

- **Block Index**:
  - `blockIdx.x`, `blockIdx.y`, `blockIdx.z`:
  - Usage:

- **Block/Grid Dimensions**:
  - `blockDim`, `gridDim`:
  - Usage:

### 3.3 Thread Index Computation

- **1D Case**:

  ```cuda
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  ```

- **2D Case**:
  ```cuda
  int index = threadIdx.x + blockIdx.x * blockDim.x +
              (threadIdx.y + blockIdx.y * blockDim.y) * width;
  ```

---

## 4️⃣ Host vs Device

### 4.1 Host (CPU)

- **Role**: Host is a traditional CPU in personal computers. It runs the OS, handles I/O operations, and executes the main, sequential parts of a program. 
- **Responsibilities**: Directs the overall flow of the application. The host allocates memory on the device, transfers data to it, initiates the workload, and eventually retrieves the result. CPUs optimise for latency (each thread finishes quickly).
- **Examples**:

### 4.2 Device (GPU)

- **Role**: The device acts as a highly specialized, massively parallel co-processor equipped with a large number of arithmetic execution units. It only executes functions called `kernels` specifically handed to it by the host.
- **Responsibilities**: It takes heavy, compute intensive tasks - that require performing the same mathematical operations on massive amounts of data simulataneously and crunches them across thousands of tiny-processing cores. GPUs optimise for throughput (total processed data).
- **Kernel Execution**: CUDA extends C function call syntax with kernel execution configuration parameters surrounded by <<< and >>>. The execution parameters are defined by dimensions of the grid and dimensions of the block. 
```
  // Execution configuration setup
  dim3 dimBlock(Width, Width)
  dim3 dimGrid(1, 1)

  //Kernel Invocation
  kernelFunc<<<dimGrid, dimBlock>>>(arg1, arg2, arg3...)
```


### 4.3 Host-Device Communication

- **Data Transfer Methods**:
  - `cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice)`: The host copies the raw data from its own system RAM over to the device memory.
  - `cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToHost)`: The host copies the final results back from the device to its own system RAM.

---

## 5️⃣ Basic CUDA Program Structure

### 5.1 Typical Workflow

```
1. Allocate device memory (cudaMalloc)
2. Copy input data to device (cudaMemcpy - H2D)
3. Launch kernel
4. Copy result back to host (cudaMemcpy - D2H)
5. Free device memory (cudaFree)
6. Process/verify results
```

### 5.2 Key CUDA Runtime Functions

| Function      | Purpose | Notes |
| ------------- | ------- | ----- |
| `cudaMalloc`  |         |       |
| `cudaFree`    |         |       |
| `cudaMemcpy`  |         |       |
| `cudaMemset`  |         |       |
| `cudaError_t` |         |       |

---

## 6️⃣ First CUDA Program: Vector Addition

### 6.1 Problem Statement

- Input:
- Output:
- Operations:

### 6.2 Kernel Implementation

```cuda
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

**Key Concepts**:

- `__global__`:
- Boundary Check (`if (i < n)`):
- Index Computation:

### 6.3 Host Code

```cuda
// Allocation
// Copy to device
// Launch
// Copy back
// Cleanup
```

---

## 7️⃣ CUDA Execution Model

### 7.1 Warp

- **Definition**:
- **Size**: (typically 32 threads)
- **Importance**:

### 7.2 Synchronization

- **`__syncthreads()`**:
  - Purpose:
  - Scope:
  - When Used:

### 7.3 Occupancy

- **Definition**:
- ## **Factors Affecting**:
  -

---

## 8️⃣ Common Patterns & Best Practices

### 8.1 Grid/Block Configuration

- ## **Considerations**:
  -
  -

- **Typical Configurations**:
  - 1D:
  - 2D:

### 8.2 Memory Access Patterns

- **Coalesced Access**:
  - Definition:
  - Benefits:

- **Bank Conflicts**:
  - Definition:
  - Avoidance:

### 8.3 Performance Tips

- ## **Do**:
  -
  -

- ## **Don't**:
  -
  -

---

## 9️⃣ Error Handling

### 9.1 Common Errors

| Error                     | Cause | Solution |
| ------------------------- | ----- | -------- |
| Memory allocation failure |       |          |
| Out-of-bounds access      |       |          |
| Synchronization issues    |       |          |

### 9.2 Debugging Approaches

- **`cudaGetLastError()`**:
- **`cudaDeviceSynchronize()`**:
- **NVIDIA Compute Sanitizer**:

---

## 🔟 Important Formulas & Calculations

### 10.1 Thread Index (1D)

```
globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x
```

### 10.2 Thread Index (2D)

```
globalIdx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y
          + threadIdx.y * blockDim.x + threadIdx.x
```

### 10.3 Occupancy

```
Occupancy = (Active Warps per SM) / (Maximum Warps per SM)
```

---

## 1️⃣1️⃣ Key Takeaways

### Summary Points

-
-
-
-

### Questions for Review

1.
2.
3.
4.

---

## 1️⃣2️⃣ Code Examples & Experiments

### Example 1: Vector Addition

- **File**: `ch01_vector_add.cu`
- **Key Concepts Illustrated**:
- **Observations**:

### Example 2: (Additional Examples)

- **File**:
- **Key Concepts Illustrated**:
- **Observations**:

---

## 1️⃣3️⃣ Additional Resources

### Official Documentation

- CUDA Programming Guide:
- CUDA Runtime API:

### Recommended Readings

-
-

### Related Code/Repositories

-
- ***

## 📝 Personal Notes & Reflections

### Concepts I Found Challenging

1.
2.

### Concepts I Grasped Well

1.
2.

### Topics for Further Study

-
-

### Practice Problems

- [ ] Implement vector addition from scratch
- [ ] Implement matrix transpose
- [ ] Experiment with different block/grid sizes
- [ ]

---

## 📅 Study Log

| Date | Duration | Topics Covered | Notes |
| ---- | -------- | -------------- | ----- |
|      |          |                |       |
|      |          |                |       |

---

**Last Updated**: **\*\***\_\_\_\_**\*\***
