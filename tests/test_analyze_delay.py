from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.analyze_delay import (
    estimate_delay,
    _save_correlation_plot,
    _save_waveform_plot,
    _fft_correlate,
)


def test_estimate_delay_identifies_offset(tmp_path) -> None:
    tx = np.array([1 + 1j, 2 + 0j, -1 - 1j], dtype=np.complex64)
    rx = np.concatenate(
        [np.zeros(5, dtype=np.complex64), tx, np.zeros(2, dtype=np.complex64)]
    )

    result = estimate_delay(tx, rx, samp_rate=1.0)

    assert result.sample_delay == 5
    assert np.isclose(result.time_delay, 5.0)
    assert result.peak_magnitude > 0.0
    assert result.normalized_peak is not None
    assert np.isclose(result.normalized_peak, 1.0)

    plot_path = tmp_path / "corr.png"
    _save_correlation_plot(result, plot_path)
    assert plot_path.exists()


def test_waveform_plot_creates_file(tmp_path) -> None:
    samples = np.exp(1j * np.linspace(0, 2 * np.pi, 32, dtype=np.float32))
    plot_path = tmp_path / "wave.png"
    _save_waveform_plot(samples, plot_path, sample_count=16)
    assert plot_path.exists()


def test_fft_correlate_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=128).astype(np.float32) + 1j * rng.normal(size=128).astype(np.float32)
    b = rng.normal(size=64).astype(np.float32) + 1j * rng.normal(size=64).astype(np.float32)
    expected = np.correlate(a, b, mode="full")
    actual = _fft_correlate(a, b)
    assert np.allclose(expected, actual, atol=1e-6)
