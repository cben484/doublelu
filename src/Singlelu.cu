#include "../include/utils_check_fuctions.cuh"
#include "cublas_v2.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

typedef struct {
  int width;
  int height;
  int stride;
  float *elements;
} blockMatrix;

int curandSgenerate(float *matrx, int m, int n, unsigned long long seed);
int curandDgenerate(double *matrx, int m, int n, unsigned long long seed);
int init_eyes(double *A, size_t m, size_t n);
int init_zero(double *A, size_t m, size_t n);
double *getSubMatrix(double *matrix, size_t lda, size_t rows, size_t rowe,
                     size_t cols, size_t cole);
template <typename T> int print_matrix_colmajor(T *matrix, int m, int n);
template <typename T> int print_matrix_rowmajor(T *matrix, int m, int n);
int swapRows(double *matrix, int Cols, int row1, int row2);

int main(int argc, char *argv[]) {

  // CHECK_Device(&argv[0]);

  cusolverDnHandle_t cusolver_handle;
  CHECK_Cusolver(cusolverDnCreate(&cusolver_handle));

  double *DA;
  double *DWorkspace;
  int DLwork;
  int *DdevIpiv;
  int *DdevInfo;
  int INPUTN = std::stoi(argv[1]);

  int m = std::stoi(argv[1]);
  int n = std::stoi(argv[2]);
  int blocksize = std::stoi(argv[3]);

  double *L;
  double *U;
  double *P;

  L = (double *)malloc(sizeof(double) * m * n);
  U = (double *)malloc(sizeof(double) * m * n);
  P = (double *)malloc(sizeof(double) * m * n);

  init_eyes(P, m, n);
  init_zero(L, m, n);
  init_zero(U, m, n);

  // 总的A，最大的A
  std::cout << "开始 " << argv[1] << " " << "x" << " " << argv[1]
            << " 规模的 Double LU 分解" << std::endl;
  CHECK_Runtime(cudaMalloc((void **)&DA, sizeof(double) * m * n));

  curandDgenerate(DA, m, n, 4321ULL);

  cudaEvent_t start, stop;
  if (cudaEventCreate(&start) != cudaSuccess) {
    printf("Failed to create start event\n");
    return EXIT_SUCCESS;
  }

  if (cudaEventCreate(&stop) != cudaSuccess) {
    printf("Failed to create stop event\n");
    CHECK_Runtime(cudaEventDestroy(start));
    return EXIT_SUCCESS;
  }

  // 分块的A
  for (int blockStart{0U}; blockStart < n; blockStart += n) {
    int blockEnd = std::min(blockStart + blocksize, n);

    double *host_DA;
    host_DA = (double *)malloc(sizeof(double) * m * n);
    CHECK_Runtime(cudaMemcpy(host_DA, DA, sizeof(double) * m * n,
                             cudaMemcpyDeviceToHost));

    // 查看整体的A
    print_matrix_rowmajor(host_DA, m, n);

    double *host_blockA;
    host_blockA = getSubMatrix(host_DA, n, blockStart, m, blockStart, blockEnd);

    // 查看分块的A
    printf("\n");
    print_matrix_rowmajor(host_blockA, m, n);
    double *blockA;
    CHECK_Runtime(
        cudaMalloc((void **)&blockA, sizeof(double) * (m - blockStart) *
                                         (blockEnd - blockStart)));
    CHECK_Runtime(
        cudaMemcpy(blockA, host_blockA,
                   sizeof(double) * (m - blockStart) * (blockEnd - blockStart),
                   cudaMemcpyHostToDevice));

    std::cout << "开始 " << m - blockStart << " " << "x" << " "
              << blockEnd - blockStart
              << " 规模的 Double LU Single bolck 循环分解" << std::endl;

    CHECK_Cusolver(cusolverDnDgetrf_bufferSize(cusolver_handle, m - blockStart,
                                               blockEnd - blockStart, blockA,
                                               blockEnd - blockStart, &DLwork));

    CHECK_Runtime(cudaMalloc((void **)&DWorkspace, sizeof(double) * DLwork));
    CHECK_Runtime(cudaMalloc((void **)&DdevInfo, sizeof(int)));
    CHECK_Runtime(cudaMalloc((void **)&DdevIpiv, sizeof(int) * m));

    std::cout << "查看m - blockStart " << m - blockStart << std::endl;
    std::cout << "查看blockEnd - blockStart " << blockEnd - blockStart
              << std::endl;

    std::cout << "flag*****\n" << std::endl;
    CHECK_Cusolver(cusolverDnDgetrf(
        cusolver_handle, m - blockStart, blockEnd - blockStart, blockA,
        blockEnd - blockStart, DWorkspace, DdevIpiv, DdevInfo));

    std::cout << "flag*****\n" << std::endl;

    int *host_DdevInfo;
    host_DdevInfo = (int *)malloc(sizeof(int));
    CHECK_Runtime(cudaMemcpy(host_DdevInfo, DdevInfo, sizeof(int),
                             cudaMemcpyDeviceToHost));
    std::cout << "flag*****\n" << std::endl;
    std::cout << *host_DdevInfo << std::endl;

    // 反正要用结果矩阵对lu操作，索性先memcpy
    CHECK_Runtime(
        cudaMemcpy(host_blockA, blockA,
                   sizeof(double) * (m - blockStart) * (blockEnd - blockStart),
                   cudaMemcpyDeviceToHost));

    std::cout << "看一下这个getrf的结果" << std::endl;
    print_matrix_rowmajor(host_blockA, m, n);

    // 看一下放入之前的U
    std::cout << std::endl;
    std::cout << "查看一下放入之前的U" << std::endl;
    print_matrix_rowmajor(U, m, n);

    // 直接将host_blockA的上三角部分放入U
    for (size_t i{0}; i < (m - blockStart); ++i) {
      for (size_t j{i}; j < (blockEnd - blockStart); ++j) {
        U[j + i * (blockEnd - blockStart)] =
            host_blockA[j + i * (blockEnd - blockStart)];
      }
    }

    // 查看一下放入的U
    std::cout << std::endl;
    std::cout << "查看一下放入的U" << std::endl;
    print_matrix_rowmajor(U, m, n);

    // 提取P从DdveIpiv
    int *host_DdevIpiv;
    host_DdevIpiv = (int *)malloc(sizeof(int) * m);
    CHECK_Runtime(cudaMemcpy(host_DdevIpiv, DdevIpiv, sizeof(int) * m,
                             cudaMemcpyDeviceToHost));
    std::cout << std::endl;
    for (int i = 0; i < m; ++i) {
      std::cout << host_DdevIpiv[i] << std::endl;
    }

    double *p;
    p = (double *)malloc(sizeof(double) * (m - blockStart) * (m - blockStart));

    init_eyes(p, m - blockStart, m - blockStart);

    printf("小p初始化\n");
    print_matrix_rowmajor(p, m - blockStart, m - blockStart);
    printf("\n");

    for (int i{0}; i < (m - blockStart); ++i) {
      if ((host_DdevIpiv[i] != i + 1) && (host_DdevIpiv[i] != 0)) {
        swapRows(p, (m - blockStart), i, (host_DdevIpiv[i] - 1));
      }
    }
    printf("查看所得p\n");
    print_matrix_rowmajor(p, m - blockStart, m - blockStart);
  }

  return EXIT_SUCCESS;
}

// 取子矩阵
double *getSubMatrix(double *matrix, size_t lda, size_t rows, size_t rowe,
                     size_t cols, size_t cole) {
  double *submatrix;
  submatrix = (double *)malloc(sizeof(double) * (rowe - rows) * (cole - cols));
  for (size_t i{rows}; i < (rowe); ++i) {
    for (size_t j{cols}; j < (cole); ++j) {
      submatrix[(i - rows) * lda + (j - cols)] =
          matrix[(i - rows) * lda + (j - cols)];
    }
  }

  return submatrix;
}

// 生成Double
int curandDgenerate(double *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniformDouble(gen, matrx, Sum);

  return EXIT_SUCCESS;
}
// 生成Single
int curandSgenerate(float *matrx, size_t m, size_t n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, matrx, Sum);

  return EXIT_SUCCESS;
}

// 初始化成ZERO矩阵
int init_zero(double *A, size_t m, size_t n) {
  for (size_t t{0}; t < m * n; ++t) {
    A[t] = 0.0;
  }
  return EXIT_SUCCESS;
}

// 初始化成eyes矩阵
int init_eyes(double *A, size_t m, size_t n) {
  init_zero(A, m, n);
  for (int i = 0; i < n; ++i) {
    A[i * m + i] += 1.0;
  }
  return EXIT_SUCCESS;
}

template <typename T> int print_matrix_colmajor(T *matrix, int m, int n) {
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // printf(" %lf", matrix[j * m + i]);
      std::cout << " " << matrix[j * m + i];
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}

template <typename T> int print_matrix_rowmajor(T *matrix, int m, int n) {
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // printf(" %lf", matrix[j * m + i]);
      std::cout << " " << matrix[i * n + j];
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}

// 行交换函数
int swapRows(double *matrix, int Cols, int row1, int row2) {
  for (int col{0}; col < Cols; ++col) {
    std::swap(matrix[row1 * Cols + col], matrix[row2 * Cols + col]);
  }
  return EXIT_SUCCESS;
}

void extract() {
  // 貌似不用提取......
  //  CHECK_Runtime(
  //      cudaMemcpy(host_blockA, blockA,
  //                 sizeof(double) * (m - blockStart) * (blockEnd -
  //                 blockStart), cudaMemcpyDeviceToHost));

  // double *l;
  // double *u;
  // l = (double *)malloc(sizeof(double) * (m - blockStart) *
  //                      (blockEnd - blockStart));

  // u = (double *)malloc(sizeof(double) * (blockEnd - blockStart) *
  //                      (blockEnd - blockStart));

  // CHECK_Runtime(
  //     cudaMalloc((void **)&u, sizeof(double) * (blockEnd - blockStart) *
  //                                 (blockEnd - blockStart)));
  // // 从DA中提取L和U
  // // 下三角
  // // 提取L，因为cusolver是col-major，所以其实是取上三角
  // printf("flag1\n");

  // for (int j = 0; j < (blockEnd - blockStart); ++j) { // 遍历列
  //   // 复制下三角部分的元素，包括对角线
  //   for (int i = j; i < (n - blockStart); ++i) { // 从对角线开始遍历行
  //     l[i + j * (blockEnd - blockStart)] =
  //         host_blockA[i + j * (blockEnd - blockStart)];
  //     l[i * n - blockStart + i] = 1.0;
  //   }

  //   // 可选地，将上三角部分（对角线之上）的元素设为零
  //   for (int i = 0; i < j; ++i) { // 遍历对角线之上的行
  //     l[i + j * n] = 0.0;
  //   }
  // }
  // // 看全部的提取的l
  // print_matrix_rowmajor(l, (m), (n));
  // printf("\n");
  // // 只看和blockA部分相对应的l
  // print_matrix_rowmajor(l, (m - blockStart), (blockEnd - blockStart));
  // printf("flag\n");

  // // 上三角
  // // 提取U，因为cusolver是col-major，所以其实是取下三角
  // for (int j = 0; j < n; ++j) { // 遍历列
  //   // 复制上三角部分的元素，包括对角线
  //   printf("1\n");
  //   for (int i = 0; i <= j; ++i) { // 遍历行直到对角线
  //     u[i + j * n] = host_blockA[i + j * n];
  //   }
  //   printf("2\n");

  //   // 可选地，将下三角部分（对角线之下）的元素设为零
  //   for (int i = j + 1; i < n; ++i) { // 遍历对角线之下的行
  //     printf("3\n");

  //     u[i + j * n] = 0.0;
  //   }
  // }
  // printf("123\n");
  // print_matrix_colmajor(u, (blockEnd - blockStart), (blockEnd -
  // blockStart));
}