// naive_gemm.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// =============================================
// Kernel: 朴素 GEMM
// 每个线程计算 C 的一个元素
// =============================================
__global__ void naive_gemm(const float *A, const float *B, float *C,
                           int M, int K, int N) {
    // 当前线程负责 C 的哪个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            //      A[row][k]        B[k][col]
        }
        C[row * N + col] = sum;
    }
}

// =============================================
// Host 代码
// =============================================
void run_naive_gemm(int M, int K, int N) {
    // --- 1. 在 CPU 上分配并初始化矩阵 ---
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // 简单初始化
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10);

    // --- 2. 在 GPU 上分配显存 ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // --- 3. 拷贝数据 CPU → GPU ---
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // --- 4. 配置并启动 kernel ---
    dim3 block_size(16, 16);  // 每个 block 16x16 = 256 个线程
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);

    naive_gemm<<<grid_size, block_size>>>(d_A, d_B, d_C, M, K, N);

    // --- 5. 拷回结果 GPU → CPU ---
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // --- 6. 验证（打印左上角 4x4）---
    printf("C[0:4][0:4] =\n");
    for (int i = 0; i < 4 && i < M; i++) {
        for (int j = 0; j < 4 && j < N; j++)
            printf("%8.1f ", h_C[i * N + j]);
        printf("\n");
    }

    // --- 7. 清理 ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

int main() {
    int M = 2048, K = 16384, N = 2048;
    printf("Naive GEMM: (%d x %d) x (%d x %d)\n", M, K, K, N);
    run_naive_gemm(M, K, N);
    return 0;
}