# Segmented Simulation & Data Compression

This module introduces the ability to run PlasmaX simulations in segments and dynamically compress the saved restart files using either Proper Orthogonal Decomposition (SVD) or Implicit Neural Representations (INR).

## Features

- **Segmented Runs:** Simulation stops and restarts automatically at intervals defined by `io.dt_save` in the `.yaml` config file.
- **POD Compression:** Compresses the Vlasov distribution function $f(x,v)$ at each save point using SVD.
- **INR Compression:** Compresses the data by training a Multi-Layer Perceptron (MLP) to learn the continuous representation of $f(x,v)$. Features mini-batch training and "warm-starting" (transferring weights between time segments) for fast convergence
- **Spectrum Analysis (POD):** Automatically computes the relative Frobenius error caused by truncation and saves the normalized singular value spectrum plot at each compression step.
- - **Error Tracking:** Stores the history of Frobenius errors in `pod_frobenius_errors.csv` and allows direct multi-run error comparison over time.

## How to Run

### 1. Run a segmented simulation

You can specify the compression method directly in your .yaml file under the io: block (compression_method: POD or INR), or override it dynamically using the --compression console argument.

- **Run with POD** (example with truncation rank=32):

```bash
plasmax-segmented --params params/two_stream.yaml --compression POD --rank 32
```

Outputs will be saved in: `results/case_two_stream/segmented/POD/r32/`

- **Run with INR:**

```bash
plasmax-segmented --params params/two_stream.yaml --compression INR
```

Outputs and network weights (.pkl) will be saved in: `results/case_two_stream/segmented/INR/inr_compress/`

### 2. Compare different compression ranks

You can easily plot the conserved quantities (Energy, Mass) and the truncation errors to observe the impact of different POD ranks or INR architectures against your baseline.

**Example:** Comparing baseline with different POD ranks:

```bash
plasmax-diags --params results/case_two_stream/baseline/data/diagnostics.csv \
results/case_two_stream/segmented/POD/r2/data/diagnostics.csv \
results/case_two_stream/segmented/POD/r8/data/diagnostics.csv \
results/case_two_stream/segmented/POD/r32/data/diagnostics.csv \
--etot --epot --mass --frob-pod --format png
```

**Note:** The script features intelligent routing. The comparison plots (e.g., `etot_comparison.png`, `mass_comparison.png`) will be automatically saved in specialized sub-directories such as `results/case_two_stream/comparisons/baseline_vs_POD/` or `baseline_vs_INR/` based on the data provided
