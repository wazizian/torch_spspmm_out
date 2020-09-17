#include "../utils.h"
#include "spspmm_out_cpu.h"

template <typename scalar_t>
void spspmm_out_cpu(struct csr_matrix<scalar_t> A, struct csr_matrix<scalar_t> B,
               struct csr_matrix<scalar_t> C){
    int colA, indexB, indexC;
    for (auto rowA = 0; rowA < A.nrow; rowA++) {
      for (auto indexA = A.rowptr[rowA]; indexA < A.rowptr[rowA + 1]; indexA++) {
        colA = A.col[indexA];
        indexB = B.rowptr[colA];
        indexC = C.rowptr[rowA];
        while (indexB < B.rowptr[colA + 1] && indexC < C.rowptr[rowA + 1]) {
          if (B.col[indexB] < C.col[indexC])
            indexB++;
          else if (B.col[indexB] > C.col[indexC])
            indexC++;
          else {
            C.val[indexC] += A.val[indexA] * B.val[indexB];
            indexB++;
            indexC++;
          }
        }
      }
    }
  }


