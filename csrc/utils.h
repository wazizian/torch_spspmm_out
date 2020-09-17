#pragma once

#include <torch/extension.h>

// Code from Pytorch's "Custom C++ and CUDA Extensions" tutorial by Peter Goldsborough
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contigious")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == torch::kInt, #x " must be int")
#define CHECK_CUDA(x) CHECK_CONTIGUOUS(x); TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");
#define CHECK_INPUT(b) TORCH_CHECK(b, "Input mismatch: " #b " failed")

template <typename scalar_t>
struct csr_matrix {
  int nrow;
  int ncol;
  int nnz;
  int* rowptr;
  int* col;
  scalar_t* val;
};
