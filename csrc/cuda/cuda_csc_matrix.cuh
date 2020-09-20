#pragma once

#include "../utils.h"
#include <cuda_runtime.h>

template <typename scalar_t>
struct cuda_csc_matrix {
  int nrow;
  int ncol;
  int nnz;
  int* colptr;
  int* row;
  scalar_t* val;
  cuda_csc_matrix(int nrow, int ncol, int nnz);
  cuda_csc_matrix(struct csr_matrix<scalar_t> R):
  ~cuda_csc_matrix();
};

template <typename scalar_t>
cuda_csc_matrix::cuda_csc_matrix(int nrow, int ncol, int nnz):
  nrow(nrow), ncol(ncol), nnz(nnz)
{
  cudaError_t cudaError;
  cudaError = cudaMalloc(&colptr, ncol * sizeof(int));
  TORCH_CHECK(cudaError == cudaSuccess, "cudaMalloc error: " # cudaError);
  cudaMalloc(&row, nnz * sizeof(int));
  TORCH_CHECK(cudaError == cudaSuccess, "cudaMalloc error: " # cudaError);
  cudaMalloc(&val, nnz * sizeof(scalar_t));
  TORCH_CHECK(cudaError == cudaSuccess, "cudaMalloc error: " # cudaError);
}

template <typename scalar_t>
cuda_csc_matrix::cuda_csc_matrix(struct csr_matrix<scalar_t> R): cuda_csc_matrix(R->nrow, R->ncol, R->nnz) {
  cusparseHandle_t cusparseHandle = at::cuda::getCurrentCUDASparseHandle();
  size_t bufferSize;
  void * buffer;
  cusparseStatus_t cusparseStatus;
  cudaError_t cudaError;
  cusparseStatus = cusparseCsr2cscEx2_bufferSize(handle, nrow, ncol, nnz,
                                                 R->val, R->rowptr, R->col, &val,
                                                 &colptr, &row, scalar_t,
                                                 CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG2, &bufferSize);
  TORCH_CHECK(cusparseStatus == CUSPARSE_STATUS_SUCCESS; "cusparseCsr2cscEx2_bufferSize error: " # cusparseStatus);
  cudaError = cudaMalloc(&buffer, bufferSize);
  TORCH_CHECK(cudaError == cudaSuccess, "cudaMalloc error: " # cudaError);
  cusparseStatus = cusparseCsr2cscEx2(handle, nrow, ncol, nnz,
                                      R->val, R->rowptr, R->col, &val,
                                      &colptr, &row, scalar_t,
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG2, buffer);
  cudaFree(buffer);
  TORCH_CHECK(cusparseStatus == CUSPARSE_STATUS_SUCCESS; "cusparseCsr2cscEx2 error: " # cusparseStatus);
}


template <typename scalar_t> cuda_csc_matrix::~cuda_csc_matrix(){
  cudaFree(colptr);
  cudaFree(row);
  cudaFree(val);
  return;
}
