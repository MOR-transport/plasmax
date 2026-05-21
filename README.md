# Plasma simulation (Python)

JAX-based Vlasov–Poisson solver.

## Requirements

- Python **3.12+**
- **Linux + NVIDIA GPU:** install a recent [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) (JAX’s CUDA 12 wheels expect driver **≥ 525** on Linux; CUDA 13 needs **≥ 580**—see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html)).
- Dependencies are listed in `pyproject.toml` (JAX, Matplotlib, PyYAML).

## Install in a virtual environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (cmd)
pip install -U pip
pip install -e .
```

- **Linux:** `pip install -e .` pulls **`jax[cuda12]`** (CUDA/cuDNN via PyPI), so JAX can use an NVIDIA GPU when one is visible. If no GPU is present, JAX still runs on the CPU.
- **macOS / Windows:** the same command installs **CPU-only** JAX (no NVIDIA GPU wheels for those platforms in this layout).

**CUDA 13 (Linux, optional):** if your stack matches JAX’s CUDA 13 wheels:

```bash
pip install -e ".[cuda13]"
```

Editable install (`-e .`) registers the `plasma-sim` console script and keeps `params/` on the import path for the wheel layout.

**Troubleshooting GPU:** JAX may fall back to CPU if drivers or libraries are wrong. Check `python -c "import jax; print(jax.devices())"`. If it picks CPU while you expect GPU, avoid pointing `LD_LIBRARY_PATH` at a conflicting system CUDA (see JAX docs).

## Run

With the venv activated:

```bash
plasmax-sim                                  # default: params/landau_damping.yaml
plasmax-sim --params params/two_stream.yaml  # explicit config
```

Or without installing the script:

```bash
python main.py
python main.py --params params/two_stream.yaml
```

## Diagnostics and plotting

Runtime diagnostics can be visualized using `plasmax-diags` post-processing tool, which is documented in [src/diagnostics/README.md](src/diagnostics/README.md).

## Testing

To ensure the stability and accuracy of the simulations (such as the IO restart workflow), we use `pytest`.
See the [tests/README.md](tests/README.md) for instructions on how to install the test dependencies and run the test suite.
