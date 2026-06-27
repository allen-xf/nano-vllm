// db_reg_tiling_gemm.cu
// =============================================
// True Double Buffering with cp.async (Ampere SM80+)
// DMA 硬件搬数据，线程不等待，真正做到搬算重叠
// =============================================
//
// cp.async 原理：
//   普通 load:  线程发请求 → 卡住等 ~400 cycles → 数据到寄存器 → 写 shared
//   cp.async:   线程提交请求给 DMA → 立刻返回继续执行 → DMA 后台搬 global→shared
//
// 执行流水线：
//   [cp.async tile1→buf1] [COMPUTE buf0] [wait + sync] [cp.async tile2→buf0] [COMPUTE buf1] ...
//    └── DMA 后台搬运 ──┘  └── 线程计算 ─┘              └── DMA 后台 ──────┘  └── 线程计算 ─┘
//    真正并行！线程和 DMA 各干各的

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 16      // 双 buffer 共 32KB shared memory
#define TM 8
#define TN 8

// =============================================
// cp.async PTX 封装
// =============================================

// 异步复制 4 bytes (1 float) 从 global → shared，线程不阻塞
__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        : : "r"(smem_addr), "l"(gmem_ptr)
    );
}

// 提交一组异步复制（标记为一个 group）
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// 等待直到最多还有 N 个 group 未完成（N=0 表示全部完成）
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// =============================================
// Kernel
// =============================================
__global__ void db_reg_tiling_gemm(const float *A, const float *B, float *C,
                                    int M, int K, int N) {
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;  // 256

    const int thread_row = tid / (BN / TN);  // 0..15
    const int thread_col = tid % (BN / TN);  // 0..15

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float reg_C[TM][TN] = {};
    float reg_A[TM];
    float reg_B[TN];

    // ===== 异步加载 tile 到指定 buffer（线程不阻塞）=====
    #define ASYNC_LOAD_TILE(buf, k_offset) \
        for (int i = tid; i < BM * BK; i += num_threads) { \
            int r = i / BK, c = i % BK; \
            int gr = block_row + r, gc = (k_offset) + c; \
            if (gr < M && gc < K) \
                cp_async_4B(&As[buf][r][c], &A[gr * K + gc]); \
            else \
                As[buf][r][c] = 0.0f; \
        } \
        for (int i = tid; i < BK * BN; i += num_threads) { \
            int r = i / BN, c = i % BN; \
            int gr = (k_offset) + r, gc = block_col + c; \
            if (gr < K && gc < N) \
                cp_async_4B(&Bs[buf][r][c], &B[gr * N + gc]); \
            else \
                Bs[buf][r][c] = 0.0f; \
        } \
        cp_async_commit();

    // ===== 计算指定 buffer =====
    #define COMPUTE(buf) \
        for (int bk = 0; bk < BK; bk++) { \
            for (int tm = 0; tm < TM; tm++) \
                reg_A[tm] = As[buf][thread_row * TM + tm][bk]; \
            for (int tn = 0; tn < TN; tn++) \
                reg_B[tn] = Bs[buf][bk][thread_col * TN + tn]; \
            for (int tm = 0; tm < TM; tm++) \
                for (int tn = 0; tn < TN; tn++) \
                    reg_C[tm][tn] += reg_A[tm] * reg_B[tn]; \
        }

    // ===== Step 1: 异步预加载第一个 tile =====
    // 线程提交搬运请求后立刻返回（DMA 后台执行）
    ASYNC_LOAD_TILE(0, 0);
    cp_async_wait_all();  // 第一个 tile 必须等到位才能开始算
    __syncthreads();

    // ===== Step 2: 主循环 —— 真正的搬算重叠 =====
    int buf = 0;
    for (int k_step = BK; k_step < K; k_step += BK) {
        // ★ 异步提交下一个 tile（DMA 后台搬运，线程不等！）
        ASYNC_LOAD_TILE(1 - buf, k_step);

        // ★ 线程立刻开始计算当前 buffer（DMA 在后台同时工作！）
        COMPUTE(buf);

        // 等 DMA 搬完 + 确保所有线程计算读完当前 buffer
        cp_async_wait_all();
        __syncthreads();

        buf = 1 - buf;
    }

    // ===== Step 3: 计算最后一个 tile =====
    COMPUTE(buf);

    // ===== Step 4: 写回 C =====
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int gr = block_row + thread_row * TM + tm;
            int gc = block_col + thread_col * TN + tn;
            if (gr < M && gc < N) {
                C[gr * N + gc] = reg_C[tm][tn];
            }
        }
    }

    #undef ASYNC_LOAD_TILE
    #undef COMPUTE
}

// =============================================
// Host
// =============================================
void run_db_reg_tiling_gemm(int M, int K, int N) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);  // 256 threads
    dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    db_reg_tiling_gemm<<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("C[0:4][0:4] =\n");
    for (int i = 0; i < 4 && i < M; i++) {
        for (int j = 0; j < 4 && j < N; j++)
            printf("%8.1f ", h_C[i * N + j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

int main() {
    int M = 2048, K = 16384, N = 2048;
    printf("Double Buffer + cp.async GEMM (BM=%d, BN=%d, BK=%d, TM=%d, TN=%d): (%d x %d) x (%d x %d)\n",
           BM, BN, BK, TM, TN, M, K, K, N);
    run_db_reg_tiling_gemm(M, K, N);
    return 0;
}
