## Thread Execution
- threadIdx.x -> tx; threadIdx.x -> ty;
- blockIdx.x -> bx; blockIdx.y -> by;
- CUDA Kernel launch generates a grid of threads organised in two-level hierarchy.
   - ---- Top level Blocks (1D or 2D)
   - ---- Bottom level Each block consists of threads (1D, 2D or 3D arrays)
- Like blocks, threads too execute in an arbitrary order.
- `Synchronisation barrier` used to make sure that all threads have completed their assigned task in the phase before any other thread begins next phase.
- Due to hardware cost considerations, CUDA devices bundle multiple threads for execution. Each block gets partitioned into `warps`.
- These warps helps to reduce costs and optimisation in memory serving accesses. Each warp consists of 32 threads.
- Assume thread block organised into a one-dimensional array represented by threadIdx.x. The indices within a warp are in consecutive locations and in increasing order.
- ```
     warp0 starts with thread 0 and ends with thread 31
     warp1 starts with thread 32 and ends with thread 63
     warp2 starts with thread 64 and ends with thread 95 and so on.....
     .
     .
     .
     .
     warpn starts with thread 32 * n and ends with 32 * (n + 1) - 1
  ```
- ```
    if (block_size % 32 != 0) {
        print("Last warp padded with extra threads to fill up the warp size")
  ```
- For 2-dimensional blocks, dimensions are projected into linear order before each block partitioning into warps. Threads with `ty=1` will be placed after `ty=0`. Similarly threads whose `ty=2` will be placed after those with `ty=1`.
- To get a clear glance, let's have a look at how threads in a 2D block placed in a linear order. All threads with `ty=0` and `tx=0,1,2,3` as moving horizontally increases tx index by 1 followed by threads with `ty=1` placed with their increasing `tx` values. 

   <img width="470" height="262" alt="Screenshot 2026-08-05 at 19 36 56" src="https://github.com/user-attachments/assets/a718f0b2-3ab2-48e8-879b-a35e3808ac59" />

- ```
      col = bx * blockDim.x + tx;
      row = by * blockDim.y + ty;
      // consider a 2D block with 8 x 8 threads.
      bx = 0; by = 0;
      // col indices ranges from 0 to blockDim.x - 1 i.e 0 to 7
      // row indices ranges from 0 to blockDim.y - 1 i.e 0 to 7
      // Wrap0 starts with T(0,0) and ends with T(3,7). In the table T(3,7) represented as T(7,3)
      // row, col notation in CUDA device represents as col, row
      // Wrap 1 starts with T(4, 0) and ends with T(7,7)
  ```
    
| Left   | Center | Right  | Left   | Center | Right  | Left   | Center  |
| :---   | :----: | ----:  | :---   | :----: | ----:  | :---   | :----:  |
| T(0,0) | T(1,0) | T(2,0) | T(3,0) | T(4,0) | T(5,0) | T(6,0) | T(7,0)  |
| T(0,1) | T(1,1) | T(2,1) | T(3,1) | T(4,1) | T(5,1) | T(6,1) | T(7,1)  |
| T(0,2) | T(1,2) | T(2,2) | T(3,2) | T(4,2) | T(5,2) | T(6,2) | T(7,2)  |
| T(0,3) | T(1,3) | T(2,3) | T(3,3) | T(4,3) | T(5,3) | T(6,3) | T(7,3)  |
| T(0,4) | T(1,4) | T(2,4) | T(3,4) | T(4,4) | T(5,4) | T(6,4) | T(7,4)  |
| T(0,5) | T(1,5) | T(2,5) | T(3,5) | T(4,5) | T(5,5) | T(6,5) | T(7,5)  |
| T(0,6) | T(1,6) | T(2,6) | T(3,6) | T(4,6) | T(5,6) | T(6,6) | T(7,6)  |
| T(0,7) | T(1,7) | T(2,7) | T(3,7) | T(4,7) | T(5,7) | T(6,7) | T(7,7)  |

-    
