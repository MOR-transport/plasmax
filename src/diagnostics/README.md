# Diagnostics and post-processing plots

Simulation output is organized under a case root directory (default `results/<case_name>/`, overridable via `io.save_dir` in the params YAML):

| Path | Contents |
|------|----------|
| `<save_dir>/data/` | `diagnostics.csv`, restart checkpoints (`f_final.npz`, …) |
| `<save_dir>/plots/` | In-run figures from `plasma-sim` and post-processed curves from `plasmax-diags` |

## `measure` — runtime diagnostics (`measure.py`)

During each time step, `plasma-sim` calls `measure(cfg, f, Efield, it, t)` to append one row to `cfg.paths.data_dir / "diagnostics.csv"`.

Recorded quantities:

| Column | Description |
|--------|-------------|
| `iter` | Global iteration index |
| `time` | Simulation time |
| `ekin` | Kinetic energy |
| `epot` | Electric potential energy |
| `etot` | Total energy (`ekin` + `epot`) |
| `l2norm` | \(L_2\) norm of \(f\) |
| `mass` | Integrated mass |
| `momentum` | Integrated momentum |

`save_distribution(f, filepath)` writes the distribution function to a `.npy` file (used where checkpoints are saved as NumPy arrays).

## `plasmax-diags` — post-processing plots (`diags.py`)

After a run, generate time-series plots from `diagnostics.csv` using the `plasmax-diags` CLI (entry point: `src.diagnostics.diags:main`).

### Basic usage

Plots all quantities by default:

```bash
plasmax-diags --params params/two_stream.yaml
```

### Specific quantities

Use `--epot`, `--ekin`, `--etot`, `--mass`, `--momentum`, or `--l2norm`:

```bash
plasmax-diags --params params/two_stream.yaml --epot --mass
```

### Compare multiple runs

Pass several YAML files to overlay cases on the same figure (output filename combines case names):

```bash
plasmax-diags --params params/two_stream.yaml params/landau_damping.yaml --epot
```

### Format and output directory

By default, figures are written as `.png` and `.tex` (TikZ) under `<save_dir>/plots/` from the params file. Override with `--format` and `--output-dir`:

```bash
plasmax-diags --params params/two_stream.yaml --format png
plasmax-diags --params params/two_stream.yaml --output-dir /tmp/figures
```
