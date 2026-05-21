"""Regression test: two-stream run over 10 steps vs stored diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from src.config import Paths, load_config
from src.sim import run_time_loop

TESTS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TESTS_DIR / "two_stream_test.yaml"
REFERENCE_PATH = TESTS_DIR / "two_stream_diagnostics_reference.yaml"

FLOAT_FIELDS = ("time", "ekin", "epot", "etot", "l2norm", "mass", "momentum")
RTOL = 1e-10
ATOL = 1e-12


def load_last_diagnostics_row(csv_path: Path) -> dict[str, float | int]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"No data in {csv_path}"
    last = rows[-1]
    return {
        "iter": int(last["iter"]),
        **{key: float(last[key]) for key in FLOAT_FIELDS},
    }


@pytest.fixture
def reference_row() -> dict[str, float | int]:
    with open(REFERENCE_PATH, encoding="utf-8") as f:
        ref = yaml.safe_load(f)
    return {
        "iter": int(ref["iter"]),
        **{key: float(ref[key]) for key in FLOAT_FIELDS},
    }


def test_two_stream_diagnostics_last_row(tmp_path: Path, reference_row: dict[str, float | int]):
    """Run two-stream for 10 iterations and match final diagnostics.csv row."""
    cfg = load_config(CONFIG_PATH)
    cfg.time.plot_freq = 0
    cfg.paths = Paths.from_case("two_stream_test", tmp_path / "two_stream_test")
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)

    run_time_loop(cfg)

    diag_path = cfg.paths.data_dir / "diagnostics.csv"
    assert diag_path.is_file(), "diagnostics.csv was not written"

    last = load_last_diagnostics_row(diag_path)
    assert last["iter"] == reference_row["iter"]
    assert last == pytest.approx(reference_row, rel=RTOL, abs=ATOL)
