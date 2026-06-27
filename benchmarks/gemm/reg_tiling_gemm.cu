// reg_tiling_gemm.cu
// =============================================
// Register Tiling GEMM
// 每个线程计算 TM×TN = 8×8 = 64 个输出元素
// 计算密度 ≈ 64 FLOP/Byte，接近 A10 平衡点
// =============================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Block tile: 每个 block 负责 C 的 BM×BN 区域
#define BM 128
#define BN 128
// K tile: 每次从 K 维加载的宽度（对齐 warp size=32，确保全局内存 coalesced 访问）
#define BK 32
// Thread tile: 每个线程负责 C 的 TM×TN 区域
#define TM 8
#define TN 8

// block 线程数 = (BM/TM) × (BN/TN) = 16 × 16 = 256
// 每线程 64 个累加器 → 充分利用寄存器
//
// 计算密度分析：
//   每个 tile step 加载: (BM×BK + BK×BN) × 4B = (4096+4096) × 4 = 32768 Bytes
//   每个 tile step 计算: BM × BN × BK × 2 = 128×128×32×2 = 1048576 FLOPs
//   计算密度 = 1048576 / 32768 = 32 FLOP/Byte
//   (与 BK 无关，由 BM×BN/(2(BM+BN)) = 128/4 = 32 决定)
//
// 对齐设计：
//   BK=32 对齐 warp size，加载 A 时一个 warp 恰好覆盖一行 32 个连续 float
//   → 单次 128B 事务，完美 coalesced

__global__ void reg_tiling_gemm(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    // --- Shared Memory ---
    __shared__ float As[BM][BK];   // 128×32 = 16 KB
    __shared__ float Bs[BK][BN];   // 32×128 = 16 KB  (共 32 KB)

    // 线程 ID（展平）
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;  // 256

    // 当前线程在 "thread tile grid" 中的逻辑位置
    // thread tile grid: (BM/TM) × (BN/TN) = 16 × 16
    const int thread_row = tid / (BN / TN);  // 0..15
    const int thread_col = tid % (BN / TN);  // 0..15

    // Block 负责 C 的全局起始行列
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // ★ 寄存器累加器：每线程 TM×TN = 64 个
    float reg_C[TM][TN] = {};

    // 寄存器缓存：从 shared memory 读取的 A/B 片段
    float reg_A[TM];
    float reg_B[TN];

    // --- 沿 K 维度分块迭代 ---
    for (int k_step = 0; k_step < K; k_step += BK) {

        // ========== 协作加载 A tile 到 shared memory (coalesced) ==========
        // As[BM][BK] = 128×32 = 4096 元素，256 线程 → 每线程搬 16 个
        // 关键：as_col = i % BK，BK=32 对齐 warp，同一 warp 内 tid 连续
        // → 访问 A[same_row][k_step+0..31]，32 个连续 float = 128B coalesced
        for (int i = tid; i < BM * BK; i += num_threads) {
            int as_row = i / BK;
            int as_col = i % BK;
            int global_row = block_row + as_row;
            int global_col = k_step + as_col;
            As[as_row][as_col] = (global_row < M && global_col < K)
                                 ? A[global_row * K + global_col] : 0.0f;
        }

        // ========== 协作加载 B tile 到 shared memory (coalesced) ==========
        // Bs[BK][BN] = 32×128 = 4096 元素，256 线程 → 每线程搬 16 个
        // bs_col = i % BN，BN=128 > warp，同一 warp 访问同一行连续 32 float
        for (int i = tid; i < BK * BN; i += num_threads) {
            int bs_row = i / BN;
            int bs_col = i % BN;
            int global_row = k_step + bs_row;
            int global_col = block_col + bs_col;
            Bs[bs_row][bs_col] = (global_row < K && global_col < N)
                                 ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        // ========== 计算：外积累加 ==========
        // 对 BK 中的每一层 k，做 TM×TN 的外积
        for (int bk = 0; bk < BK; bk++) {
            // 从 As 取 TM 个元素（当前线程负责的 A 行片段）
            for (int tm = 0; tm < TM; tm++) {
                reg_A[tm] = As[thread_row * TM + tm][bk];
            }
            // 从 Bs 取 TN 个元素（当前线程负责的 B 列片段）
            for (int tn = 0; tn < TN; tn++) {
                reg_B[tn] = Bs[bk][thread_col * TN + tn];
            }
            // ★ 外积：TM × TN = 64 次 FMA
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    reg_C[tm][tn] += reg_A[tm] * reg_B[tn];
                }
            }
        }

        __syncthreads();
    }

    // ========== 写回 C ==========
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int global_row = block_row + thread_row * TM + tm;
            int global_col = block_col + thread_col * TN + tn;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = reg_C[tm][tn];
            }
        }
    }
}

// =============================================
// Host 代码
// =============================================
void run_reg_tiling_gemm(int M, int K, int N) {
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

    // Grid: 每个 block 算 BM×BN = 128×128 的 C 子块
    dim3 block_size(16, 16);  // 256 threads
    dim3 grid_size((N + BN - 1) / BN,
                   (M + BM - 1) / BM);

    reg_tiling_gemm<<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);

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
    printf("Register Tiling GEMM (BM=%d, BN=%d, BK=%d, TM=%d, TN=%d): (%d x %d) x (%d x %d)\n",
           BM, BN, BK, TM, TN, M, K, K, N);
    run_reg_tiling_gemm(M, K, N);
    return 0;
}
