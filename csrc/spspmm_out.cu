#include <torch/extension.h>
#include <torch/script.h>
#include "utils.h"

#include "cpu/spspmm_out_cpu.h"

#ifdef __CUDACC__
#include "cuda/spspmm_out_cuda.h"
#endif

#ifdef __CUSP__
#include "cusp/spspmm_out_cusp.h"
#endif

template <typename scalar_t>
void check_and_make_csr_matrix(torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor val, int ncol, bool is_cuda,
                               struct csr_matrix<scalar_t> * m){
  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(val.dim() == 1);
  CHECK_INPUT(col.size(0) == val.size(0));
  CHECK_INT(rowptr);
  CHECK_INT(col);
  // val.scalar_type() == scalar_t
  if (is_cuda) {
    CHECK_CUDA(rowptr);
    CHECK_CUDA(col);
    CHECK_CUDA(val);
  }
  m->nnz = col.size(0);
  m->nrow = rowptr.size(0) - 1;
  m->ncol = ncol;
  m->rowptr = rowptr.data_ptr<int>();
  m->col = col.data_ptr<int>();
  m->val = val.data_ptr<scalar_t>();
  return;
} 

void spspmm_out(torch::Tensor rowptrA, torch::Tensor colA, torch::Tensor valA,
                torch::Tensor rowptrB, torch::Tensor colB, torch::Tensor valB, 
                torch::Tensor rowptrC, torch::Tensor colC, torch::Tensor valC,
                int64_t ncolB) {
  auto scalar_type = valA.scalar_type();
  bool is_cuda = false;
  int ncolA = rowptrB.size(0) - 1;
  int ncolC = ncolB;
  if (rowptrA.device().is_cuda()){
#ifdef __CUDACC__
    is_cuda = true;
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "spspmm_out", [&] {
      struct csr_matrix<scalar_t> A, B, C;
      check_and_make_csr_matrix<scalar_t>(rowptrA, colA, valA, ncolA, is_cuda, &A);
      check_and_make_csr_matrix<scalar_t>(rowptrB, colB, valB, ncolB, is_cuda, &B);
      check_and_make_csr_matrix<scalar_t>(rowptrC, colC, valC, ncolC, is_cuda, &C);
#ifdef __CUSP__
      spspmm_out_cusp<scalar_t>(is_cuda, A, B, C);
#else
      if (is_cuda){
        AT_ERROR("TODO");
      } else {
        spspmm_out_cpu<scalar_t>(A, B, C);        
      }
#endif
  });
}

static auto registry = torch::RegisterOperators("torch_spspmm_out::spspmm_out", &spspmm_out);
// Another way would be to bind spspmm_out with pybind
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
//   m.def("spspmm_out", &spspmm_out, "SpSpMM out");
// }
