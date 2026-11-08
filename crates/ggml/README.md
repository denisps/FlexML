# GGML Rust crate

The safe API selects CPU by default. Set GGML's process-global log filter before constructing a graph; available levels are `Off`, `Warn`, and `Debug`:

```rust
ggml::set_log_level(ggml::LogLevel::Off);
```

The logger callback is installed once and later level changes use an atomic filter. Enable one accelerator backend Cargo feature to construct and execute graphs on that backend:

```sh
cargo run -p graph-example
cargo run -p graph-example --features metal
cargo test -p ggml --features metal
```

The crate test builds a graph, executes it on the selected backend, and checks its result. The example also exposes `opencl`, `vulkan`, and `cuda` features. At the workspace root, pass these example feature aliases directly (for example, `--features metal`); when depending on `ggml` from another crate, use its namespaced dependency feature (for example, `ggml/metal`). Backend SDKs must be installed for those builds.

# Updating the GGML submodule

GGML lives in `crates/ggml/sys/ggml` as a Git submodule pointing to <https://github.com/ggml-org/ggml.git>.

After cloning this repository, initialize the pinned GGML checkout with:

```sh
git submodule update --init --recursive
```

To move to the latest commit on GGML's default branch:

```sh
git submodule update --remote --merge crates/ggml/sys/ggml
```

Then commit the changed submodule pointer in this repository:

```sh
git add crates/ggml/sys/ggml
git commit -m "Update GGML submodule"
```

The parent repository records the exact GGML commit, so other checkouts continue to use the pinned revision until they explicitly update the submodule. After updating GGML, review and regenerate `sys/src/bindings.rs` if the C API has changed. From `crates/ggml/sys`, use:

```sh
bindgen wrapper.h \
  --allowlist-function 'ggml_(init|free|new_tensor_1d|add|mul|new_graph|build_forward_expand|graph_compute_with_ctx|get_f32_1d|set_f32_1d|backend_.*)' \
  --allowlist-type 'ggml_(init_params|context|tensor|cgraph|backend.*|status|type)' \
  --output src/bindings.rs -- -I ggml/include -I ggml/src
```
