# Building with OpenCL

OpenCL needs both development files for compiling GGML and a working vendor implementation/ICD for runtime access to a device. Package names differ across distributions and vendors; installing headers alone does not provide GPU acceleration.

## Dependencies

- Git, CMake, Rust/Cargo, C and C++ compilers, and initialized GGML submodule (see [BUILDING.md](BUILDING.md)).
- OpenCL headers and loader/development package.
- A compatible OpenCL ICD/driver from the device vendor (for example, the appropriate Intel, AMD, or NVIDIA driver/runtime).

## Common systems

### Ubuntu/Debian

```sh
sudo apt update
sudo apt install ocl-icd-opencl-dev opencl-headers
```

Install the OpenCL runtime/ICD for your GPU separately using the GPU vendor's supported instructions. For CPU-only OpenCL testing, a CPU ICD such as POCL may be installed with `sudo apt install pocl-opencl-icd`.

### Fedora

```sh
sudo dnf install ocl-icd-devel opencl-headers
```

Install the device vendor's OpenCL runtime/ICD separately.

### Arch Linux

```sh
sudo pacman -S ocl-icd opencl-headers
```

Install the suitable vendor ICD package separately.

### Windows

Install the OpenCL SDK/development headers and loader using the vendor-supported SDK or a maintained OpenCL SDK distribution, and install the GPU vendor's compatible driver. Ensure CMake can find the OpenCL headers and library (for example, through the SDK's CMake package or `OpenCL_ROOT`).

### macOS

Apple no longer supplies OpenCL for new development and availability varies by macOS version. This backend should generally be used on systems with a supported OpenCL implementation and SDK; install the implementation and headers from its provider. Metal is the recommended native Apple backend.

## Build

From the repository root:

```sh
cargo run -p graph-example --features opencl
```
