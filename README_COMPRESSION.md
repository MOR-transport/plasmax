# Segmented Simulation & Data Compression

This module introduces the ability to run PlasmaX simulations in segments and dynamically compress the saved restart files using either Proper Orthogonal Decomposition (SVD) or Implicit Neural Representations (INR).

## Features

* **Segmented Runs:** Simulation stops and restarts automatically at intervals defined by `io.dt_save` in the `.yaml` config file.
* **POD Compression:** Compresses the Vlasov distribution function $f(x,v)$ at each save point using SVD.
* **INR Compression:** Compresses the data by training a neural network to learn the continuous representation of $f(x,v)$. Features curriculum learning ("warm-starting" weights between time segments) and a hybrid ADAM + L-BFGS optimization routine for high-precision convergence.
* **Spectrum Analysis (POD):** Automatically computes the relative Frobenius error caused by truncation and saves the normalized singular value spectrum plot at each compression step.
* **Error & Convergence Tracking:** Stores the history of Frobenius errors and network losses in `.csv` files, allowing for direct multi-run error comparisons over time.
* **Performance Profiling:** Tracks and logs the CPU time spent on physical simulation versus data compression/training, enabling direct computational cost benchmarking via stacked bar charts.

## How to Run

### 1. Run a segmented simulation

You can specify the compression method directly in your `.yaml` file under the `io:` block (`compression_method: POD` or `INR`), or override it dynamically using the `--compression` console argument.

* **Run with POD** (example with truncation rank=32):

```bash
plasmax-segmented --params params/two_stream.yaml --compression POD --rank 32

```

*Outputs will be saved in:* `results/case_two_stream/segmented/POD/r32/`

* **Run with INR** (example with SIREN architecture):

```bash
plasmax-segmented --params params/two_stream.yaml --compression INR --arch siren

```

*Outputs and network weights (`.pkl`) will be saved in:* `results/case_two_stream/segmented/INR/inr_siren/`

---

### 2. Generate Diagnostics and Comparison Plots

You can plot conserved quantities (Energy, Mass), truncation errors, convergence losses, and CPU times to observe the impact of different compression strategies against your baseline. The script features intelligent routing: plots are automatically saved in specialized sub-directories (e.g., `baseline_vs_POD/`, `baseline_vs_INR/`, or `mixed_comparisons/`) based on the provided inputs.

#### A. Evaluating POD Ranks

Compare the baseline with multiple POD truncation ranks:

```bash
plasmax-diags \
  --params results/case_two_stream/baseline/data/diagnostics.csv \
           results/case_two_stream/segmented/POD/r2/data/diagnostics.csv \
           results/case_two_stream/segmented/POD/r8/data/diagnostics.csv \
           results/case_two_stream/segmented/POD/r32/data/diagnostics.csv \
  --etot --epot --mass --frob-pod --format png

```

#### B. Benchmarking INR Architectures

Compare the reconstruction accuracy, final losses, and CPU times across various neural network architectures (MLP, SIREN, Fourier MLP, and their periodic counterparts):

```bash
plasmax-diags \
  --params results/case_two_stream/baseline/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_mlp_16/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_siren/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_siren_deep_128/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_fourier_mlp/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_fourier_mlp_deep_128/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_periodic_mlp_16/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_periodic_siren/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_periodic_siren_deep_128/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_periodic_fourier_mlp/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_periodic_fourier_mlp_deep_128/data/diagnostics.csv \
  --etot --epot --mass --frob-inr --inr-loss --cpu-time --format png

```

#### C. Cross-Method Comparison (POD vs INR)

Evaluate the global Frobenius error and computational cost between classical SVD and specific neural representations:

```bash
plasmax-diags \
  --params results/case_two_stream/baseline/data/diagnostics.csv \
           results/case_two_stream/segmented/INR/inr_siren/data/diagnostics.csv \
           results/case_two_stream/segmented/POD/r2/data/diagnostics.csv \
           results/case_two_stream/segmented/POD/r8/data/diagnostics.csv \
           results/case_two_stream/segmented/POD/r32/data/diagnostics.csv \
  --frob --cpu-time --format png

```