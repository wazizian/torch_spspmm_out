from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

cuda_support = torch.cuda.is_available()

def get_extensions():
    extra_link_args = []
    extension_dir = os.path.join(os.path.dirname(__file__), "csrc")
    if cuda_support:
        Extension = CUDAExtension
        extra_link_args += ["-lcusparse"]
    else:
        Extension = CppExtension
    extension = Extension(name="torch_spspmm_out._spspmm_out",
                          sources=[
                              "csrc/spspmm_out.cpp",
                              ],
                          include_dirs=[extension_dir],
                          extra_link_args=extra_link_args,
                          )
    return [extension]

setup(
        name="torch_spspmm_out",
        install_requires=["torch"],
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False, no_python_abi_suffix=True)}
        )



