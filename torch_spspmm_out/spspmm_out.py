import torch

#TODO: wrap spspmm_out in torch.autograd.Function
spspmm_out = torch.ops.torch_spspmm_out.spspmm_out
