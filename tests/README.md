# PlasmaX Testing

This directory contains the unit tests for the PlasmaX project using `pytest`.

## Tests

* **`test_segmented.py`**: Verifies that running the simulation continuously yields the exact same results (diagnostics data) as running it in smaller, restarted segments. This ensures no data or precision is lost during the save/load workflow.

* **`test_two_stream.py`**: Runs a short two-stream case (`two_stream_test.yaml`, 10 time steps) and checks the last row of `diagnostics.csv` against reference values in `two_stream_diagnostics_reference.yaml`.

## Installation

To run the tests, you need to install the optional test dependencies. From the root of the repository, run:

```bash
pip install -e ".[test]"
```

## Running the Tests

To execute the tests with verbose output, run the following command from the repository root:

```bash
pytest tests/ -v
```

(You can also run a specific test file by targeting it directly, e.g., pytest tests/test_segmented.py -v) .
