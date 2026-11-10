# Building GGML

## Clone

Clone this repository and fetch the pinned GGML submodule:

```sh
git clone --recurse-submodules <repository-url>
cd ollm
```

If you already cloned without submodules:

```sh
git submodule update --init --recursive
```

All builds require Git, CMake, a C/C++ compiler toolchain, and Rust/Cargo. Cargo invokes the `ggml-sys` build script, which configures and builds GGML from the submodule. The default Cargo feature set builds the CPU backend only; enable one accelerator feature to build that backend alongside CPU support. All Cargo commands below run from the repository root. GGML's optional tests and examples are disabled.

## CPU only

```sh
cargo run -p graph-example
```

This is the default; all accelerator backends are disabled.

## Accelerator dependencies

| Backend | Cargo feature | Additional dependencies |
| --- | --- | --- |
| Metal | `metal` | macOS, Xcode Command Line Tools (Clang and Apple SDKs); Metal, MetalKit, MetalPerformanceShaders, Foundation and Accelerate frameworks are provided by macOS. See [BUILDING_metal.md](BUILDING_metal.md) |
| OpenCL | `opencl` | OpenCL development headers and loader/ICD; a vendor OpenCL implementation and compatible device/driver for actual acceleration. See [BUILDING_opencl.md](BUILDING_opencl.md) |
| Vulkan | `vulkan` | Vulkan SDK with headers, loader, shader compiler (including `glslc`), and SPIRV-Headers development files; compatible Vulkan device/driver. See [BUILDING_vulkan.md](BUILDING_vulkan.md) |
| CUDA | `cuda` | NVIDIA CUDA Toolkit including `nvcc`, plus a compatible NVIDIA GPU and driver. See [BUILDING_cuda.md](BUILDING_cuda.md) |

## Vulkan Example
```sh
cargo run -p graph-example --features vulkan
```

Without an accelerator feature, CPU is selected. With one enabled, the example initializes that backend, allocates graph tensors there, executes the graph on it, then reads the scalar result back. `Graph::backend()` reports the selected backend.
