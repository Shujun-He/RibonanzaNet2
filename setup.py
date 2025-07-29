#!/usr/bin/env python3

import os
import sys

# Check for required dependencies before importing
def check_dependencies():
    """Check if required dependencies are available"""
    # Skip dependency check if running in build isolation (PEP517)
    if os.environ.get('PEP517_BUILD_BACKEND') or '--no-deps' in sys.argv:
        print("🔄 Skipping dependency check (build isolation)")
        return
    
    # Skip dependency check for certain build commands
    build_commands = {'build', 'build_ext', 'bdist_wheel', 'sdist', 'egg_info'}
    if any(cmd in sys.argv for cmd in build_commands):
        print("🔄 Skipping dependency check (build command)")
        return
    
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pybind11
    except ImportError:
        missing_deps.append("pybind11")
    
    if missing_deps:
        print("\n❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n🔧 Please install missing dependencies:")
        print("  pip install " + " ".join(missing_deps))
        print("\n🚀 Or use the Makefile:")
        print("  make fix-deps")
        print("  make build")
        # Don't exit during build, just warn
        if not any(cmd in sys.argv for cmd in ['install', 'develop']):
            exit(1)

# Check dependencies first (but allow builds to proceed)
check_dependencies()

# Safe imports with fallbacks
try:
    import torch
    torch_version = torch.__version__
    print(f"🔍 PyTorch version: {torch_version}")
    
    # Check PyTorch version compatibility
    torch_major, torch_minor = map(int, torch_version.split('.')[:2])
    if torch_major < 1 or (torch_major == 1 and torch_minor < 12):
        print(f"\n❌ PyTorch version {torch_version} is not supported")
        print("🔧 Please upgrade PyTorch:")
        print("  pip install torch>=1.12.0")
        if 'build' not in sys.argv:
            exit(1)
except ImportError:
    torch = None
    print("⚠️ PyTorch not available during build setup")

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_cmake_dir
    import pybind11
except ImportError:
    print("⚠️ pybind11 not available, using fallback")
    Pybind11Extension = None
    build_ext = None

try:
    from setuptools import setup, Extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
except ImportError as e:
    print(f"⚠️ Setup tools import error: {e}")
    # Provide minimal fallbacks
    from setuptools import setup, Extension
    BuildExtension = None
    CUDAExtension = None
    CppExtension = None

# Determine if CUDA is available
def cuda_is_available():
    try:
        if torch is None:
            return False
        return torch.cuda.is_available()
    except:
        return False

# Get CUDA compute capability with enhanced detection
def get_cuda_compute_capability():
    if not cuda_is_available():
        return []
    
    try:
        if torch is None:
            return []
        
        # Try to get actual GPU capability
        capability = torch.cuda.get_device_capability()
        arch_code = f"{capability[0]}{capability[1]}"
        
        # Generate optimized flags for detected GPU
        gpu_flags = [
            f"-gencode=arch=compute_{arch_code},code=sm_{arch_code}",  # Exact match
            f"-gencode=arch=compute_{arch_code},code=compute_{arch_code}",  # PTX for compatibility
        ]
        
        print(f"🎯 Detected GPU compute capability: {capability[0]}.{capability[1]}")
        return gpu_flags
        
    except Exception as e:
        print(f"⚠️ Could not detect GPU capability: {e}")
        # Default to comprehensive compute capabilities for maximum compatibility
        return [
            "-gencode=arch=compute_70,code=sm_70",   # V100, Tesla V100
            "-gencode=arch=compute_75,code=sm_75",   # T4, RTX 20xx, Quadro RTX
            "-gencode=arch=compute_80,code=sm_80",   # A100, A40
            "-gencode=arch=compute_86,code=sm_86",   # RTX 30xx series
            "-gencode=arch=compute_87,code=sm_87",   # Jetson AGX Orin
            "-gencode=arch=compute_89,code=sm_89",   # RTX 40xx series, L40S
            "-gencode=arch=compute_90,code=sm_90",   # H100
            "-gencode=arch=compute_90,code=compute_90", # PTX for future compatibility
        ]

def main():
    # Check if required tools are available
    if BuildExtension is None or CUDAExtension is None:
        print("⚠️ PyTorch extensions not available, falling back to basic setup")
        # Minimal setup without extensions
        setup(
            name='triangle_kernels',
            version='1.0.0',
            description='Triangle Kernels (basic setup)',
            python_requires='>=3.8',
            install_requires=[
                'torch>=1.12.0',
                'numpy>=1.20.0',
                'pybind11>=2.10.0',
            ],
        )
        return

    # Enhanced C++ compilation flags for maximum optimization
    cxx_flags = [
        "-O3",                              # Maximum optimization
        "-std=c++17",                       # C++17 standard
        "-fPIC",                            # Position independent code
        "-march=native",                    # Optimize for current CPU architecture
        "-mtune=native",                    # Tune for current CPU
        "-ffast-math",                      # Fast math operations
        "-funroll-loops",                   # Unroll loops for performance
        "-fomit-frame-pointer",             # Omit frame pointer for speed
        "-finline-functions",               # Inline function calls
        "-fno-signed-zeros",                # Optimize floating point
        "-fno-trapping-math",               # No math traps
        "-frename-registers",               # Register renaming optimization
        "-ftree-vectorize",                 # Tree vectorization
        "-mfpmath=sse",                     # Use SSE for floating point math
        "-msse4.2",                         # Enable SSE 4.2 instructions
        "-mavx2",                           # Enable AVX2 instructions if available
        "-DEIGEN_USE_MKL_ALL",              # Use Intel MKL if available
        "-DWITH_CUDA" if cuda_is_available() else "-DCPU_ONLY",
        "-DTORCH_EXTENSION_NAME=triangle_kernels_cuda",
        "-DNDEBUG",                         # Disable debug assertions
    ]
    
    # Enhanced NVCC flags for maximum CUDA optimization
    nvcc_flags = [
        "-O3",                              # Maximum optimization
        "--use_fast_math",                  # Fast math operations
        "--ftz=true",                       # Flush denormals to zero
        "--prec-div=false",                 # Fast division
        "--prec-sqrt=false",                # Fast square root
        "--fmad=true",                      # Fused multiply-add
        "-std=c++17",                       # C++17 standard
        "-Xcompiler=-fPIC",                 # Position independent code
        "-Xcompiler=-O3",                   # Max optimization for host code
        "-Xcompiler=-march=native",         # Optimize host code for current CPU
        "-Xcompiler=-mtune=native",         # Tune host code for current CPU
        "-Xcompiler=-ffast-math",           # Fast math for host code
        "-Xcompiler=-funroll-loops",        # Unroll loops in host code
        "--expt-relaxed-constexpr",         # Relaxed constexpr
        "--expt-extended-lambda",           # Extended lambda support
        "--extended-lambda",                # Lambda in device code
        "--restrict",                       # Use restrict keyword optimization
        "--maxrregcount=64",                # Limit register usage for occupancy
        "-lineinfo",                        # Line info for profiling
        "--ptxas-options=-v",               # Verbose PTX assembler
        "--ptxas-options=-O3",              # PTX optimization
        "--ptxas-options=--warn-on-spills", # Warn on register spills
        "-Xcudafe", "--diag_suppress=esa_on_defaulted_function_ignored",
        "-DTORCH_API_INCLUDE_EXTENSION_H",  # Latest PyTorch API
        "-DWITH_CUDA",                      # Enable CUDA macros
        "-DTORCH_EXTENSION_NAME=triangle_kernels_cuda",
        "-DCUDA_HAS_FP16=1",               # Enable FP16 support
        "-DNDEBUG",                         # Disable debug assertions
        "-DEIGEN_USE_GPU",                  # Enable GPU Eigen
        "-DEIGEN_USE_CUDA",                 # Enable CUDA Eigen
    ] + get_cuda_compute_capability()
    
    # Include directories with enhanced paths
    include_dirs = [
        "csrc",
    ]
    
    # Add PyTorch includes if available
    if torch is not None:
        try:
            torch_includes = torch.utils.cpp_extension.include_paths()
            include_dirs.extend(torch_includes)
            print(f"🔗 Added PyTorch include paths: {len(torch_includes)} directories")
        except Exception as e:
            print(f"⚠️ Could not get PyTorch includes: {e}")
    
    # Enhanced library directories and libraries
    library_dirs = []
    libraries = []
    
    if cuda_is_available():
        # CUDA libraries with enhanced optimization
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home:
            include_dirs.append(os.path.join(cuda_home, 'include'))
            lib64_path = os.path.join(cuda_home, 'lib64')
            lib_path = os.path.join(cuda_home, 'lib')
            if os.path.exists(lib64_path):
                library_dirs.append(lib64_path)
            elif os.path.exists(lib_path):
                library_dirs.append(lib_path)
        
        # Enhanced CUDA libraries for maximum performance
        libraries.extend([
            'cublas',           # Basic Linear Algebra Subprograms
            'cublasLt',         # cuBLAS Light (optimized GEMM)
            'curand',           # Random number generation
            'cufft',            # Fast Fourier Transform
            'cusparse',         # Sparse matrix operations
        ])
        
        # Source files for CUDA extension
        sources = [
            "csrc/triangle_kernels.cpp",
            "csrc/triangle_kernels_cuda.cu",
        ]
        
        # Enhanced define macros for maximum optimization
        define_macros = [
            ('WITH_CUDA', None),
            ('THRUST_IGNORE_CUB_VERSION_CHECK', None),
            ('TORCH_EXTENSION_NAME', 'triangle_kernels_cuda'),
            ('CUDA_HAS_FP16', '1'),
            ('NDEBUG', None),                    # Disable debug assertions
            ('EIGEN_USE_GPU', None),             # Enable GPU Eigen
            ('EIGEN_USE_CUDA', None),            # Enable CUDA Eigen
            ('EIGEN_USE_MKL_ALL', None),         # Use Intel MKL if available
            ('AT_USE_JITERATOR', None),          # Use PyTorch JITerator
            ('USE_CUDA', None),                  # General CUDA flag
            ('__CUDA_NO_HALF_OPERATORS__', None), # Avoid half precision issues
            ('__CUDA_NO_HALF_CONVERSIONS__', None),
            ('__CUDA_NO_BFLOAT16_CONVERSIONS__', None),
        ]
        
        # Create CUDA extension with enhanced settings
        extension = CUDAExtension(
            name='triangle_kernels_cuda',
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': nvcc_flags
            },
            define_macros=define_macros,
            extra_link_args=[
                '-Wl,-rpath,$ORIGIN',           # RPATH for library loading
                '-Wl,--as-needed',              # Only link needed libraries
                '-Wl,-O3',                      # Link-time optimization
                '-flto',                        # Link Time Optimization
            ] if not sys.platform.startswith('darwin') else [
                '-Wl,-rpath,@loader_path',      # macOS equivalent of RPATH
            ],
            optional=False,                     # Fail if can't build CUDA version
        )
        
        ext_modules = [extension]
        cmdclass = {'build_ext': BuildExtension.with_options(use_ninja=True)}
        
        print("🔥 Building with CUDA support")
        print(f"🎯 CUDA compute capabilities: {get_cuda_compute_capability()}")
        print(f"🚀 Enhanced optimization flags: {len(nvcc_flags)} NVCC flags")
        
    else:
        # Enhanced CPU-only extension
        sources = [
            "csrc/triangle_kernels.cpp",
        ]
        
        # Enhanced CPU-only define macros
        define_macros = [
            ('CPU_ONLY', None),
            ('TORCH_EXTENSION_NAME', 'triangle_kernels_cpu'),
            ('NDEBUG', None),
            ('EIGEN_USE_MKL_ALL', None),
            ('AT_USE_JITERATOR', None),
        ]
        
        extension = CppExtension(
            name='triangle_kernels_cpu',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=cxx_flags,
            define_macros=define_macros,
            extra_link_args=[
                '-Wl,-rpath,$ORIGIN',
                '-Wl,--as-needed',
                '-Wl,-O3',
                '-flto',
            ] if not sys.platform.startswith('darwin') else [
                '-Wl,-rpath,@loader_path',
            ],
        )
        
        ext_modules = [extension]
        cmdclass = {'build_ext': BuildExtension.with_options(use_ninja=True)}
        
        print("💻 Building with CPU-only support")
        print(f"🚀 Enhanced optimization flags: {len(cxx_flags)} C++ flags")
    
    # Enhanced package metadata
    setup(
        name='triangle_kernels',
        version='1.0.0',
        description='Optimized C++/CUDA kernels for Triangle Multiplicative Module',
        long_description="""
        High-performance C++/CUDA implementation of Triangle Multiplicative Module operations.
        
        Features:
        - Optimized CUDA kernels for layer normalization
        - Custom standard attention BMM (no window constraints)
        - Fused operations to reduce memory bandwidth
        - cuBLAS integration for maximum GEMM performance
        - Enhanced compiler optimizations (-O3, -march=native, --use_fast_math)
        - Memory pool optimization and VRAM management
        - CPU fallback implementation with vectorization
        - Full PyTorch autodiff support
        - Triton-equivalent performance with faithful memory patterns
        """,
        author='RibonanzaNet2 Team',
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        zip_safe=False,
        python_requires='>=3.8',
        py_modules=['triangle_kernels_wrapper', 'autodiff_optimized_triangle'],
        install_requires=[
            'torch>=1.12.0',
            'numpy>=1.20.0',
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: C++',
            'Programming Language :: CUDA',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    )

if __name__ == '__main__':
    main() 