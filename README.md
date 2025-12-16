# Build and Run

This project is built using Rust and exposed to Python via `maturin`.

## Requirements
- Rust toolchain
- Python
- `maturin` (`pip install maturin`)

## Build and Install

From the root folder of the Rust project, run:

```bash
maturin build
pip install ./target/...
