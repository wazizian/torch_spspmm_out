import glob
import os
import torch

# Find .so library containing spspmm_out
init_dir = os.path.dirname(__file__)
libs = glob.glob(os.path.join(init_dir, '*.so'))
assert len(libs) == 1
torch.ops.load_library(libs[0])
