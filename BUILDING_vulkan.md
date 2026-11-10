# Building with Vulkan

GGML's Vulkan backend requires Vulkan development headers, loader, the shader compiler `glslc`, and SPIRV-Headers (including its CMake package configuration). A compatible Vulkan device and driver are needed for runtime acceleration.

## Dependencies

- Git, CMake, Rust/Cargo, C and C++ compilers, and initialized GGML submodule (see [BUILDING.md](BUILDING.md)).
- Vulkan SDK/development package providing headers, loader, and `glslc`.
- SPIRV-Headers development package, including `SPIRV-HeadersConfig.cmake` (or `spirv-headers-config.cmake`), which CMake's Vulkan backend configuration requires.
- Compatible Vulkan-capable GPU and driver (or another Vulkan implementation).

## Common systems

### Ubuntu/Debian

```sh
sudo apt update
sudo apt install libvulkan-dev vulkan-tools glslc spirv-headers
```

Install a Vulkan-capable driver for the GPU. On Ubuntu/Debian, driver packages depend on the vendor and hardware.

### Fedora

```sh
sudo dnf install vulkan-loader-devel vulkan-headers vulkan-tools glslc spirv-headers-devel
```

Install the suitable vendor Vulkan driver package separately.

### Arch Linux

```sh
sudo pacman -S vulkan-headers vulkan-icd-loader vulkan-tools shaderc spirv-headers
```

Install the appropriate vendor ICD package (for example, an AMD, Intel, or NVIDIA Vulkan driver).

### Windows

Install the LunarG Vulkan SDK for Windows, including Vulkan headers, loader, shader tools, and SPIRV-Headers, and install a Vulkan-capable GPU driver. Use the SDK environment setup so CMake can locate all SDK components.

### macOS

Vulkan is not a native Apple graphics API. To use Vulkan, install the LunarG Vulkan SDK, including SPIRV-Headers, and a supported Vulkan portability implementation such as MoltenVK. Device/runtime support depends on the implementation and hardware; Metal is normally preferable on macOS.

## Build

From the repository root:

```sh
cargo run -p graph-example --features vulkan
```
