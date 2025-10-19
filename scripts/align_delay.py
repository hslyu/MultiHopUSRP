#!/usr/bin/env python3
"""
Correlate TX/RX captures and align UHD packets with IQ samples.

This helper loads:
  * TX waveform (`.npy`)
  * RX capture (`.c32` recorded by rx_flowgraph.py)
  * TX and RX VITA-49 traffic captured with dumpcap (`.pcap`)

It returns the sample delay reported by scripts/analyze_delay.py and matches
the corresponding UHD packet timestamps so you can compare over-the-air delay
from both the sample domain and NIC timestamps.

Usage example:
    python scripts/align_delay.py \
        --tx-pcap outputs/tx_capture.pcap \
        --rx-pcap outputs/rx_capture.pcap \
        --tx-waveform outputs/prach_long.npy \
        --rx-capture captures/prach_rx.c32 \
        --samp-rate 1e6 \
        --timestamp-mode adapter_unsynced

`dumpcap` should have been run with hardware timestamping enabled
(`--time-stamp-type adapter_unsynced` for Mellanox ConnectX cards).
"""
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from scripts.analyze_delay import estimate_delay

try:
    from scapy.all import RawPcapReader  # type: ignore
except ImportError:  # pragma: no cover - scapy is optional
    RawPcapReader = None  # type: ignore


VITA_HEADER_BYTES = 16


@dataclass
class PacketInfo:
    """Metadata extracted per VITA-49 UDP packet."""

    timestamp: float  # Seconds (float) from PCAP record
    sequence: int  # 12-bit sequence number
    payload_len: int  # Payload bytes (excluding header)
    samples: int  # Number of IQ samples carried
    sample_start: int  # First sample index represented by this packet


def _require_scapy() -> None:
    if RawPcapReader is None:
        raise RuntimeError(
            "This script requires scapy. Install it via `pip install scapy` "
            "or rerun dumpcap with `-P`/`--pcapng` and use another parser."
        )


def _bytes_per_sample(sample_bytes: int, channel_count: int) -> int:
    value = sample_bytes * channel_count
    if value <= 0:
        raise ValueError("Bytes per sample must be positive.")
    return value


def iter_packets(
    pcap_path: Path,
    *,
    bytes_per_sample_total: int,
) -> Iterator[PacketInfo]:
    """
    Yield PacketInfo entries for each VITA-49 frame in the PCAP.

    Only little-endian PCAP records are supported (the default produced by dumpcap).
    """
    _require_scapy()

    cumulative_samples = 0
    for payload, metadata in RawPcapReader(str(pcap_path)):
        if len(payload) <= VITA_HEADER_BYTES:
            continue

        timestamp = metadata.sec + metadata.usec / 1e6
        seq = struct.unpack("!I", payload[0:4])[0] & 0x0FFF
        payload_len = len(payload) - VITA_HEADER_BYTES

        if payload_len % bytes_per_sample_total != 0:
            raise ValueError(
                f"Payload length {payload_len} is not divisible by sample size "
                f"{bytes_per_sample_total}."
            )

        samples = payload_len // bytes_per_sample_total
        yield PacketInfo(
            timestamp=timestamp,
            sequence=seq,
            payload_len=payload_len,
            samples=samples,
            sample_start=cumulative_samples,
        )
        cumulative_samples += samples


def find_packet_for_sample(
    packets: Iterable[PacketInfo], sample_index: int
) -> PacketInfo:
    """
    Return the packet containing the specified sample index.
    """
    for packet in packets:
        start = packet.sample_start
        end = start + packet.samples
        if start <= sample_index < end:
            return packet
    raise ValueError(f"Sample index {sample_index} not found in capture.")


def align_delays(
    *,
    tx_pcap: Path,
    rx_pcap: Path,
    tx_waveform: Path,
    rx_capture: Path,
    samp_rate: float,
    sample_bytes: int,
    channel_count: int,
) -> dict[str, float]:
    """
    Compute sample-based delay and align to PCAP timestamps.
    """
    bytes_per_sample_total = _bytes_per_sample(sample_bytes, channel_count)

    tx_packets = list(
        iter_packets(tx_pcap, bytes_per_sample_total=bytes_per_sample_total)
    )
    if not tx_packets:
        raise ValueError(f"No VITA packets found in {tx_pcap}.")

    rx_packets = list(
        iter_packets(rx_pcap, bytes_per_sample_total=bytes_per_sample_total)
    )
    if not rx_packets:
        raise ValueError(f"No VITA packets found in {rx_pcap}.")

    tx_samples = np.load(tx_waveform)
    if tx_samples.ndim != 1 or not np.iscomplexobj(tx_samples):
        raise ValueError("TX waveform must be a 1-D complex array.")

    rx_samples = np.fromfile(rx_capture, dtype=np.complex64)
    if rx_samples.size == 0:
        raise ValueError("RX capture file contains no samples.")

    result = estimate_delay(
        tx_samples.astype(np.complex64, copy=False),
        rx_samples,
        samp_rate=samp_rate,
    )

    sample_delay = result.sample_delay
    tx_packet = tx_packets[0]  # TX sample index 0 corresponds to first packet
    rx_packet = find_packet_for_sample(rx_packets, sample_delay)

    tx_timestamp = tx_packet.timestamp
    rx_timestamp = rx_packet.timestamp + (
        (sample_delay - rx_packet.sample_start) / samp_rate
    )

    return {
        "sample_delay": float(sample_delay),
        "time_delay_samples": float(result.time_delay),
        "tx_timestamp": tx_timestamp,
        "rx_timestamp": rx_timestamp,
        "ota_delay_from_timestamps": rx_timestamp - tx_timestamp,
        "tx_packet_sequence": float(tx_packet.sequence),
        "rx_packet_sequence": float(rx_packet.sequence),
        "rx_packet_sample_start": float(rx_packet.sample_start),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align TX/RX captures with PCAP timestamps."
    )
    parser.add_argument("--tx-pcap", type=Path, required=True)
    parser.add_argument("--rx-pcap", type=Path, required=True)
    parser.add_argument("--tx-waveform", type=Path, required=True)
    parser.add_argument("--rx-capture", type=Path, required=True)
    parser.add_argument(
        "--samp-rate",
        type=float,
        required=True,
        help="Sample rate used during capture (Hz).",
    )
    parser.add_argument(
        "--sample-bytes",
        type=int,
        default=8,
        help="Bytes per complex sample (default: 8 for complex32).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of simultaneous channels captured (default: 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print alignment results as JSON.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = align_delays(
        tx_pcap=args.tx_pcap,
        rx_pcap=args.rx_pcap,
        tx_waveform=args.tx_waveform,
        rx_capture=args.rx_capture,
        samp_rate=args.samp_rate,
        sample_bytes=args.sample_bytes,
        channel_count=args.channels,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
