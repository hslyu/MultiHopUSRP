import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import align_delay


class _DummyMeta:
    def __init__(self, sec: int, usec: int):
        self.sec = sec
        self.usec = usec


def test_find_packet_for_sample_simple() -> None:
    packets = [
        align_delay.PacketInfo(0.0, 0, 64, 8, 0),
        align_delay.PacketInfo(1.0, 1, 64, 8, 8),
    ]
    packet = align_delay.find_packet_for_sample(packets, 10)
    assert packet.sequence == 1


def test_align_delays_with_stub(tmp_path, monkeypatch) -> None:
    tx_samples = np.array([1 + 0j, 0.5 + 0.1j, -0.3 + 0.2j], dtype=np.complex64)
    rx_samples = np.concatenate(
        [
            np.zeros(4, dtype=np.complex64),
            tx_samples,
            np.zeros(2, dtype=np.complex64),
        ]
    )

    tx_wave_path = tmp_path / "tx.npy"
    np.save(tx_wave_path, tx_samples)

    rx_cap_path = tmp_path / "rx.c32"
    rx_samples.tofile(rx_cap_path)

    header_seq0 = struct.pack("!I", 0) + b"\x00" * 12
    header_seq1 = struct.pack("!I", 1) + b"\x00" * 12
    sample_bytes = 8

    tx_payload = header_seq0 + b"\x00" * (len(tx_samples) * sample_bytes)

    rx_payload1 = header_seq0 + b"\x00" * (4 * sample_bytes)
    rx_payload2 = header_seq1 + b"\x00" * (len(tx_samples) * sample_bytes)

    tx_pcap_path = tmp_path / "tx.pcap"
    rx_pcap_path = tmp_path / "rx.pcap"

    records = {
        str(tx_pcap_path): [(tx_payload, _DummyMeta(1, 0))],
        str(rx_pcap_path): [
            (rx_payload1, _DummyMeta(1, 0)),
            (rx_payload2, _DummyMeta(1, 500000)),
        ],
    }

    def fake_raw_pcap_reader(path: str):
        for payload, meta in records[path]:
            yield payload, meta

    monkeypatch.setattr(align_delay, "RawPcapReader", fake_raw_pcap_reader)

    summary = align_delay.align_delays(
        tx_pcap=tx_pcap_path,
        rx_pcap=rx_pcap_path,
        tx_waveform=tx_wave_path,
        rx_capture=rx_cap_path,
        samp_rate=1.0,
        sample_bytes=8,
        channel_count=1,
    )

    assert summary["sample_delay"] == pytest.approx(4.0)
    assert summary["ota_delay_from_timestamps"] == pytest.approx(0.5)
