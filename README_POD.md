# Segmented Simulation & POD Compression

This module introduces the ability to run PlasmaX simulations in segments and dynamically compress the saved restart files using Proper Orthogonal Decomposition (SVD).

## Features

- **Segmented Runs:** Simulation stops and restarts automatically at intervals defined by `io.dt_save` in the `.yaml` config file.
- **POD Compression:** Compresses the Vlasov distribution function $f(x,v)$ at each save point.
- **Spectrum Analysis:** Automatically computes the relative Frobenius error caused by truncation and saves the normalized singular value spectrum plot at each compression step.
- - **Error Tracking & Comparison:** Stores the history of Frobenius errors in `pod_frobenius_errors.csv` and allows direct multi-run error comparison over time.

## How to Run

### 1. Run a segmented simulation with POD

Make sure your `.yaml` file contains `compression_method: POD` in the `io:` block. Then run :

```bash
#example with truncation rank=32
plasmax-segmented --params params/two_stream.yaml --tend 40 --rank 32
```

**Note:** The script automatically creates an isolated directory `( results/two_stream_segmented_POD_r32)` to avoid overwriting baseline data.

### 2. Compare different compression ranks

You can easily plot the conserved quantities (Energy, Mass) and the truncation errors to observe the impact of different POD ranks:

```bash
plasmax-diags --params results/two_stream/data/diagnostics.csv results/two_stream_segmented_POD_r32/data/diagnostics.csv results/two_stream_segmented_POD_r8/data/diagnostics.csv results/two_stream_segmented_POD_r2/data/diagnostics.csv --etot --epot --mass --frob --format png
```

**Note:** The comparison plots (`etot_comparison.png`, `etot_comparison.png`, `mass_comparison.png`, and `frob_error_comparison.png`) will be automatically saved in a centralized directory at `results/comparisons/`
