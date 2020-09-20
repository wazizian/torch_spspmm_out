#pragma once

#include "../utils.h"
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <thrust/device_ptr.h>

template <typename T>
class host_ptr {
    T* ptr;
  public:
    host_ptr(T* ptr): ptr(ptr) {};
    T& operator*() { return *ptr; };
    host_ptr<T> operator+(int i) {
      return host_ptr<T>(ptr + i);
    };
};

template <typename scalar_t, template<class> typename ptr_t>
using ValueView = cusp::array1d_view<ptr_t<scalar_t>>;

template <template<class> typename ptr>
using IndexView = cusp::array1d_view<ptr_t<int>>;

template <typename scalar_t, template<class> typename ptr_t>
using CSRMatrixView = cusp::csr_matrix_view<IndexView<scalar_t, ptr_t>, IndexView<scalar_t, ptr_t>, ValueView<ptr_t>>;

template <typename scalar_t, template<class> typename ptr_t>
CSRMatrixView<scalar_t, ptr_t> make_csr_matrix_view(struct csr_matrix<scalar_t> A){

  ptr_t<int> rowptr_ptr(A.rowptr);
  ptr_t<int> col_ptr(A.col);
  ptr_t<scalar_t> val_ptr(A.val);

  IndexView<ptr_t> rowptr_view =
    cusp::make_array1d_view(rowptr_ptr, rowptr_ptr + A.nrow + 1);
  IndexView<ptr_t> col_view =
    cusp::make_array1d_view(col_ptr, col_ptr + A.nnz);
  ValueView<scalar_t, ptr_t> val_view =
    cusp::make_array1d_view(val_ptr, val_ptr + A.nnz);

  return cusp::make_csr_matrix_view(A.nrow, A.ncol, A.nnz, rowptr_view,
                                    col_view, val_view);
}

template <typename scalar_t, template<class> typename ptr_t>
void _spspmm_out_cusp(struct csr_matrix<scalar_t> A, struct csr_matrix<scalar_t> B,
                      struct csr_matrix<scalar_t> C){
  
  auto A_view = make_csr_matrix_view<scalar_t, ptr_t>(A);
  auto B_view = make_csr_matrix_view<scalar_t, ptr_t>(B);
  auto C_view = make_csr_matrix_view<scalar_t, ptr_t>(C);
  
  thrust::identity<float>   identity;
  thrust::multiplies<float> combine;
  thrust::plus<float>       reduce;
  
  cusp::generalized_spgemm(A_view, B_view, C_view, identity, combine, reduce);
  return;
}

template <typename scalar_t>
void spspmm_out_cusp(bool is_cuda, struct csr_matrix<scalar_t> A,
                     struct csr_matrix<scalar_t> B, struct csr_matrix<scalar_t> C){
#ifdef __CUDACC__
  if (is_cuda)
    _spspmm_out_cusp<scalar_t, thrust::device_ptr>(A, B, C);
  else
#endif
    _spspmm_out_cusp<scalar_t,           host_ptr>(A, B, C);
  return;
}

