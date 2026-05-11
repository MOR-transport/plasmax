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
plasma-sim                          # default: params/landau_damping.yaml
plasma-sim params/two_stream.yaml  # explicit config
```

Or without installing the script:

```bash
python main.py
python main.py params/two_stream.yaml
```

## Diagonstics and Plotting

After running a simulation, you can automatically generate plots (png and tikz formats) from the `diagnostics.csv` data using the `plasmax-diags` CLI tool.

- **Basic usage (plots all quantities by default):**

```bash
plasmax-diags --params params/two_stream.yaml
```

- **Plots specific quantities :**
You can select specific physical quantities using their respective flags (--epot, --ekin, --etot, --mass, --momentum, --l2norm).
eg :

```bash
plasmax-diags --params params/two_stream.yaml --epot --mass
```

- **Compare multiple runs:**
Pass multiple YAML parameter files to plot data from different simulations on the same figure. The output file will automatically combine the case names
eg :

```bash
plasmax-diags --params params/two_stream1.yaml params/two_stream2.yaml --epot
```

- **Format and Output options:**
By default, the tool produces both .png and .tex (TikZ) files in the corresponding results/<case_name> directory. You can override this behavior using the --format and --output-dir arguments.
eg :

```bash
plasmax-diags --params params/two_stream.yaml --format png
```