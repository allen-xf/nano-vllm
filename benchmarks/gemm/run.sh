#!/bin/bash
# GEMM Benchmark: 编译 + nsys profiling + 分析
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  编译 naive_gemm"
echo "=========================================="
nvcc -O2 -o naive_gemm naive_gemm.cu
echo "Done."

echo ""
echo "=========================================="
echo "  编译 tiling_gemm"
echo "=========================================="
nvcc -O2 -o tiling_gemm tiling_gemm.cu
echo "Done."

echo ""
echo "=========================================="
echo "  编译 reg_tiling_gemm"
echo "=========================================="
nvcc -O2 -o reg_tiling_gemm reg_tiling_gemm.cu
echo "Done."

echo ""
echo "=========================================="
echo "  编译 db_reg_tiling_gemm"
echo "=========================================="
nvcc -O2 -o db_reg_tiling_gemm db_reg_tiling_gemm.cu
echo "Done."

echo ""
echo "=========================================="
echo "  Profiling naive_gemm"
echo "=========================================="
nsys profile --force-overwrite=true -o naive_report ./naive_gemm

echo ""
echo "=========================================="
echo "  Profiling tiling_gemm"
echo "=========================================="
nsys profile --force-overwrite=true -o tiling_report ./tiling_gemm

echo ""
echo "=========================================="
echo "  分析结果: naive_gemm"
echo "=========================================="
nsys stats --report cuda_gpu_trace --force-export=true naive_report.nsys-rep

echo ""
echo "=========================================="
echo "  Profiling reg_tiling_gemm"
echo "=========================================="
nsys profile --force-overwrite=true -o reg_tiling_report ./reg_tiling_gemm

echo ""
echo "=========================================="
echo "  Profiling db_reg_tiling_gemm"
echo "=========================================="
nsys profile --force-overwrite=true -o db_reg_tiling_report ./db_reg_tiling_gemm

echo ""
echo "=========================================="
echo "  分析结果: tiling_gemm"
echo "=========================================="
nsys stats --report cuda_gpu_trace --force-export=true tiling_report.nsys-rep

echo ""
echo "=========================================="
echo "  分析结果: reg_tiling_gemm"
echo "=========================================="
nsys stats --report cuda_gpu_trace --force-export=true reg_tiling_report.nsys-rep

echo ""
echo "=========================================="
echo "  分析结果: db_reg_tiling_gemm"
echo "=========================================="
nsys stats --report cuda_gpu_trace --force-export=true db_reg_tiling_report.nsys-rep
