from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

cuda_support = torch.cuda.is_available()

cusp_support = False
if cuda_support and os.getenv("CUSP_INCLUDE_PATH", default="") != "":
    cusp_support = True

def get_extensions():
    extra_link_args = []
    define_macros = []
    extension_dir = os.path.join(os.path.dirname(__file__), "csrc")
    include_dirs = [extension_dir]

    if cuda_support:
        Extension = CUDAExtension
    else:
        print("cpp")
        Extension = CppExtension
    if cusp_support:
        define_macros += [("__CUSP__", None)]
        include_dirs += [os.getenv("CUSP_INCLUDE_PATH", "")]

    extension = Extension(name="torch_spspmm_out._spspmm_out",
                          sources=[
                              "csrc/spspmm_out.cu",
                              ],
                          include_dirs=include_dirs,
                          define_macros=define_macros,
                          extra_link_args=extra_link_args,
                          )
    return [extension]

setup(
        name="torch_spspmm_out",
        install_requires=["torch"],
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False, no_python_abi_suffix=True)}
        )



