// tiling_gemm.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 32

// =============================================
// Kernel: Tiling GEMM
// 每个 block 计算 C 的一个 TILE x TILE 块
// =============================================
__global__ void tiling_gemm(const float *A, const float *B, float *C,
                            int M, int K, int N) {
    // --- 分配 shared memory ---
    __shared__ float As[TILE][TILE];  // A 的 tile
    __shared__ float Bs[TILE][TILE];  // B 的 tile

    // 当前线程在 block 内的位置
    int tx = threadIdx.x;  // 列
    int ty = threadIdx.y;  // 行

    // 当前线程负责的 C 元素的全局位置
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // --- 沿 K 维度分段遍历 ---
    for (int k_step = 0; k_step < K; k_step += TILE) {
        // 协作加载：每个线程搬一个元素到 shared memory
        // 256 个线程，刚好搬 16x16 = 256 个元素
        if (row < M && (k_step + tx) < K) {
            As[ty][tx] = A[row * K + k_step + tx];
        } else {
            As[ty][tx] = 0.0f;  // 越界补零
        }

        if ((k_step + ty) < K && col < N) {
            Bs[ty][tx] = B[(k_step + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 同步：确保 tile 加载完毕
        __syncthreads();

        // 在 shared memory 中做小矩阵乘法
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
            //     ↑ 同一行线程共享 As 同一行（A 的复用）
            //              ↑ 同一列线程共享 Bs 同一列（B 的复用）
        }

        // 同步：确保所有线程用完当前 tile 后再加载下一个
        __syncthreads();
    }

    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================
// Host 代码
// =============================================
void run_tiling_gemm(int M, int K, int N) {
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

    dim3 block_size(TILE, TILE);
    dim3 grid_size((N + TILE - 1) / TILE,
                   (M + TILE - 1) / TILE);

    tiling_gemm<<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);

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
    printf("Tiling GEMM (TILE=%d): (%d x %d) x (%d x %d)\n", TILE, M, K, K, N);
    run_tiling_gemm(M, K, N);
    return 0;
}