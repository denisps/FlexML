# Building with Metal

Metal is supported only on macOS. The GGML build links Apple's Metal, MetalKit, MetalPerformanceShaders, Foundation, Accelerate, and C++ frameworks; they are supplied by the macOS SDK.

## Dependencies

- macOS with a Metal-capable Apple device.
- Xcode Command Line Tools (provides Clang, SDK headers/frameworks, and build utilities).
- Git, CMake, Rust/Cargo, and the repository's initialized GGML submodule (see [BUILDING.md](BUILDING.md)).

Install the command-line tools with:

```sh
xcode-select --install
```

Alternatively install full Xcode from the Mac App Store and select it with `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` if needed. No separate Metal SDK download is required. Metal is not available on Linux or Windows.

## Build

From the repository root:

```sh
cargo run -p graph-example --features metal
```
