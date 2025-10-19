from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.tx_flowgraph import (
    TxSettings,
    _build_arg_parser,
)


def test_arg_parser_requires_waveform() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--waveform-file",
            "outputs/test.npy",
            "--repeat",
            "--gain",
            "5.0",
        ]
    )
    assert args.waveform_file == Path("outputs/test.npy")
    assert args.repeat is True
    assert np.isclose(args.gain, 5.0)


def test_load_waveform_caches_data(tmp_path) -> None:
    waveform_path = Path(tmp_path) / "wave.npy"
    samples = (np.ones(16, dtype=np.complex64) + 1j).astype(np.complex64)
    np.save(waveform_path, samples)

    settings = TxSettings(waveform_path=waveform_path)
    first = settings.load_waveform()
    second = settings.load_waveform()

    assert np.array_equal(first, samples)
    assert first is second
