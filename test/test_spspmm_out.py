from itertools import product
import pytest
import torch
from torch_spspmm_out import spspmm_out

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices += [torch.device("cuda")]
float_types = [torch.float, torch.double]

torch.manual_seed(0)

N = 500
ATOL = 1e-7 * N

def masked_dense_to_csr(tens, mask):
    """ Converts a dense matrix with a mask to a sparse tensor in CSR """
    nnz_rows = torch.sum(mask, axis=1)
    rowptr = torch.empty(tens.size(0) + 1, dtype=torch.int)
    rowptr[0] = 0
    rowptr[1:] = torch.cumsum(nnz_rows, 0)
    nnz = rowptr[-1]
    nnz_indices = torch.nonzero(mask, as_tuple=True)
    col = torch.flatten(nnz_indices[1]).to(torch.int)
    val = torch.flatten(tens[nnz_indices])
    assert col.size() == (nnz,)
    assert val.size() == (nnz,)
    return rowptr, col, val

@pytest.fixture
def mask():
    """ Returns a random mask """
    probs = 0.5*torch.ones(N, N)
    return torch.bernoulli(probs)

@pytest.fixture
def tens():
    """ Returns a random dense tensor """
    return torch.normal(0, 1, size=(N, N))

maskA, maskB, maskC = mask, mask, mask
tensA, tensB, tensC = tens, tens, tens

@pytest.mark.parametrize('float_dtype, device', product(float_types, devices))
def test_spspmm_out(tensA, maskA, tensB, maskB, tensC, maskC, float_dtype, device):
    ncolB = tensB.size(1)
    csrA = masked_dense_to_csr(tensA, maskA)
    csrB = masked_dense_to_csr(tensB, maskB)
    csrC = masked_dense_to_csr(tensC, maskC)
    tensA *= maskA
    tensB *= maskB
    tensC *= maskC
    real_result = torch.matmul(tensA, tensB) + tensC
    real_csr = masked_dense_to_csr(real_result, maskC)
    spspmm_out(*csrA, *csrB, *csrC, ncolB)
    assert torch.equal(csrC[0], real_csr[0])
    assert torch.equal(csrC[1], real_csr[1])
    assert torch.allclose(csrC[2], real_csr[2], atol=ATOL), "inf norm: {}".format(
            torch.norm(csrC[2] - real_csr[2], p=float('inf')))
