# Building with CUDA

CUDA requires an NVIDIA GPU supported by the installed toolkit and driver. CUDA is not available on Apple M1. The CUDA Toolkit must provide `nvcc` and its development libraries; a driver alone is insufficient for building.

## Dependencies

- Git, CMake, Rust/Cargo, C and C++ compilers, and initialized GGML submodule (see [BUILDING.md](BUILDING.md)).
- NVIDIA CUDA Toolkit with `nvcc` and development libraries.
- Compatible NVIDIA GPU and NVIDIA driver.

## Common systems

### Ubuntu/Debian

Install a supported NVIDIA driver and the CUDA Toolkit using NVIDIA's official CUDA repository/instructions for your distribution and release. The package is commonly named `cuda-toolkit`; avoid relying on distribution packages if their toolkit version is incompatible with your driver or GGML. Check that `nvcc --version` works and that the toolkit is discoverable by CMake (typically via `CUDA_HOME` or `CUDAToolkit_ROOT`).

### Fedora

Install the supported NVIDIA driver and CUDA Toolkit from NVIDIA's official repository/instructions for your Fedora release. Ensure `nvcc` is on `PATH` and CMake can locate the toolkit.

### Windows

Install the NVIDIA driver and CUDA Toolkit for Windows using NVIDIA's official installer. Install a compatible Visual Studio C++ build toolchain as required by the toolkit. Ensure `nvcc` is available in the developer environment.

### macOS

CUDA is not supported on modern macOS and is unavailable on Apple Silicon. Use Metal on supported Apple hardware instead.

## Build

From the repository root:

```sh
cargo run -p graph-example --features cuda
```
