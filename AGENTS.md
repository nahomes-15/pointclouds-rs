# AGENTS.md

## Cursor Cloud specific instructions

This is a Rust point cloud processing library (`pointclouds-rs`) with Python bindings via PyO3. No external services, databases, or Docker needed — it is a pure computational library.

### Quick reference

- **Rust lint**: `cargo fmt --all -- --check` and `cargo clippy --workspace -- -D warnings`
- **Rust tests** (176 tests): `cargo test --workspace`
- **Python bindings build**: `source .venv/bin/activate && maturin develop --release --manifest-path crates/python/Cargo.toml`
- **Python tests** (36 tests): `source .venv/bin/activate && pytest tests/test_python.py -v`
- **Demo pipelines**: `source .venv/bin/activate && python examples/python/aerial_lidar.py --quick`

See `README.md` "Building from source" section for full instructions.

### Non-obvious caveats

- A Python virtualenv at `.venv` is required for `maturin develop`. Always activate it before running Python commands.
- The Rust toolchain must be set to `stable` (not pinned to an older version). A transitive dependency (`fixed`) requires `edition2024`, which needs Rust 1.85+. Run `rustup default stable` if the default is pinned to an older version.
- `maturin develop` installs the package in editable mode into the venv. After any Rust code changes in `crates/python/` or upstream crates, re-run `maturin develop --release --manifest-path crates/python/Cargo.toml` to rebuild the native extension.
- Python packages installed via `pip install` to user-level (`~/.local/bin`) may not be on PATH. Always use the `.venv` to avoid this issue.
