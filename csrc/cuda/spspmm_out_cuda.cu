#include "../utils.h"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
void spspmm_out_cuda(struct csr_matrix<scalar_t>* A, struct csr_matrix<scalar_t>* B, struct csr_matrix<scalar_t>* C){
  struct csc_matrix<scalar_t> cscB(B->nrow, B->ncol, B->nnz);

  
}
