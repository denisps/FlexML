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

You need Git, CMake, and a C/C++ compiler. Cargo invokes the `ggml-sys` build script, which configures and builds GGML from the submodule. The default Cargo feature set builds the CPU backend only; enable one accelerator feature to build that backend alongside CPU support. All Cargo commands below run from the repository root.

GGML's optional tests and examples are disabled for these builds.

## CPU only

```sh
cargo run -p graph-example
```

This is the default; all accelerator backends are disabled.

## Metal (macOS)

Requires macOS and Xcode Command Line Tools. Metal support is available on Apple platforms; it is not supported on Linux or Windows.

```sh
cargo run -p graph-example --features metal
```

Cargo configures GGML with Metal and Accelerate enabled.

## OpenCL

Requires OpenCL development headers and a working OpenCL SDK/ICD for the target device. Install the platform-specific SDK before configuring.

```sh
cargo run -p graph-example --features opencl
```

## Vulkan

Requires the Vulkan SDK (including headers, loader, and shader compiler) to be installed and discoverable by CMake.

```sh
cargo run -p graph-example --features vulkan
```

## CUDA

Requires a compatible NVIDIA GPU and NVIDIA CUDA Toolkit (including `nvcc`) installed and discoverable by CMake. CUDA is not available on Apple M1.

```sh
cargo run -p graph-example --features cuda
```

## Backend selection notes

The backend Cargo features are mutually exclusive; Cargo reports a build-script error if more than one accelerator feature is enabled. CPU support is always built, and without an accelerator feature it is selected by default. With an accelerator feature, the example initializes that backend, allocates the graph tensors there, executes the graph on it, then reads the scalar result back. `Graph::backend()` reports the selected backend.
