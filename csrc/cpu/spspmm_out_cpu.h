#pragma once

#include "../utils.h"

template <typename scalar_t>
void spspmm_out_cpu(struct csr_matrix<scalar_t> A, struct csr_matrix<scalar_t> B,
               struct csr_matrix<scalar_t> C);

#include "spspmm_out_cpu.tpp"
