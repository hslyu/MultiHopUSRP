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
        --timestamp-mode adapter_unsynced \
        --plot outputs/alignment_corr.png

`dumpcap` should have been run with hardware timestamping enabled
(`--time-stamp-type adapter_unsynced` for Mellanox ConnectX cards).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_delay import DelayResult, estimate_delay

try:  # pragma: no cover - import is environment-specific
    from scapy.utils import RawPcapNgReader, RawPcapReader  # type: ignore

    _SCAPY_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # noqa: F401 - keep original exception for later
    RawPcapReader = None  # type: ignore
    RawPcapNgReader = None  # type: ignore
    _SCAPY_IMPORT_ERROR = exc


VITA_HEADER_BYTES = 16
plt = None


def _iter_pcap_records(pcap_path: Path):
    """Return an iterator over (frame, metadata) pairs for the capture file."""
    _require_scapy()
    try:
        with pcap_path.open("rb") as handle:
            magic = handle.read(4)
    except FileNotFoundError:
        if RawPcapReader is None:
            raise RuntimeError(
                f"Failed to open {pcap_path} and no RawPcapReader is available."
            )
        # Allow tests to monkeypatch RawPcapReader without creating real files.
        return RawPcapReader(str(pcap_path))  # type: ignore[call-arg]
    except OSError as exc:  # pragma: no cover - filesystem error
        raise RuntimeError(f"Failed to open {pcap_path}: {exc}") from exc

    if RawPcapNgReader is not None and magic == b"\x0a\x0d\x0d\x0a":
        return RawPcapNgReader(str(pcap_path))  # type: ignore[call-arg]

    if RawPcapReader is None:
        raise RuntimeError("scapy RawPcapReader unavailable; cannot parse PCAP.")

    return RawPcapReader(str(pcap_path))  # type: ignore[call-arg]


def _metadata_to_timestamp(metadata: object) -> float:
    """Return timestamp in seconds from capture metadata."""
    if hasattr(metadata, "tshigh") and hasattr(metadata, "tslow"):
        ts_res = getattr(metadata, "tsresol", None)
        if not ts_res:
            ts_res = 1_000_000
        raw = (int(getattr(metadata, "tshigh")) << 32) | int(getattr(metadata, "tslow"))
        return raw / float(ts_res)
    if hasattr(metadata, "sec") and hasattr(metadata, "usec"):
        ts_res = 1_000_000
        raw = int(getattr(metadata, "sec")) * ts_res + int(getattr(metadata, "usec"))
        return raw / float(ts_res)
    if hasattr(metadata, "ts_sec") and hasattr(metadata, "ts_usec"):
        ts_res = 1_000_000
        raw = int(getattr(metadata, "ts_sec")) * ts_res + int(
            getattr(metadata, "ts_usec")
        )
        return raw / float(ts_res)
    raise AttributeError("Unsupported metadata format returned by RawPcapReader.")


@dataclass
class PacketInfo:
    """Metadata extracted per VITA-49 UDP packet."""

    timestamp: float  # Seconds (float) from PCAP record
    sequence: int  # 12-bit sequence number
    payload_len: int  # Payload bytes (excluding header)
    samples: int  # Number of IQ samples carried
    sample_start: int  # First sample index represented by this packet


@dataclass
class AlignmentArtifacts:
    """Bundle of alignment outputs and intermediate artifacts."""

    summary: dict[str, float]
    delay_result: DelayResult
    rx_reference_timestamp: float


def _ensure_matplotlib() -> None:
    """Lazily import matplotlib when plotting is requested."""
    global plt  # type: ignore[global-variable-undefined]
    if plt is not None:
        return

    cache_dir = Path(__file__).resolve().parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Plotting requires matplotlib. Install it via `pip install matplotlib`."
        ) from exc

    plt = _plt


def _require_scapy() -> None:
    if RawPcapReader is None and RawPcapNgReader is None:
        message = (
            "This script requires scapy. Install it via `pip install scapy` "
            "or rerun dumpcap with `-P`/`--pcapng` and use another parser."
        )
        if _SCAPY_IMPORT_ERROR is not None:
            message += f" (import failed with: {_SCAPY_IMPORT_ERROR})"
        raise RuntimeError(message)


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
    for frame, metadata in _iter_pcap_records(pcap_path):
        if len(frame) <= 42 + VITA_HEADER_BYTES:
            continue

        eth_len = 14
        ip_header_len = (frame[eth_len] & 0x0F) * 4
        udp_offset = eth_len + ip_header_len
        udp_header_len = 8
        vita_offset = udp_offset + udp_header_len

        payload = frame[vita_offset:]
        if len(payload) <= VITA_HEADER_BYTES:
            continue

        timestamp = _metadata_to_timestamp(metadata)

        seq = struct.unpack("!I", payload[0:4])[0] & 0x0FFF
        payload_len = len(payload) - VITA_HEADER_BYTES

        if payload_len % bytes_per_sample_total != 0:
            # Non-IQ payload (e.g., status frame); skip it.
            continue

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


def _compute_alignment(
    *,
    tx_pcap: Path,
    rx_pcap: Path,
    tx_waveform: Path,
    rx_capture: Path,
    samp_rate: float,
    sample_bytes: int,
    channel_count: int,
) -> AlignmentArtifacts:
    """Internal helper that performs alignment and returns full artifacts."""
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

    sample_delay = int(result.sample_delay)
    tx_packet = tx_packets[0]  # TX sample index 0 corresponds to first packet
    rx_packet = find_packet_for_sample(rx_packets, sample_delay)

    tx_timestamp = tx_packet.timestamp
    sample_offset = sample_delay - rx_packet.sample_start
    rx_timestamp = rx_packet.timestamp + (sample_offset / samp_rate)

    peak_time_seconds = float(result.time_delay)
    rx_stream_start_timestamp = rx_timestamp - peak_time_seconds
    rx_first_packet_timestamp = rx_packets[0].timestamp
    ota_delay_seconds = rx_timestamp - tx_timestamp

    summary: dict[str, float] = {
        "rx_start_timestamp": rx_stream_start_timestamp,
        "tx_start_timestamp": tx_timestamp,
        "rx_side_tx_timestamp": rx_timestamp,
        "sample_delay (sample)": float(sample_delay),
        "time_delay (s)": peak_time_seconds,
        "ota_delay_from_tx_to_rx": ota_delay_seconds,
        "rx_first_packet_timestamp": rx_first_packet_timestamp,
        "rx_packet_sample_start": float(rx_packet.sample_start),
        "tx_packet_sequence": float(tx_packet.sequence),
        "rx_packet_sequence": float(rx_packet.sequence),
    }

    rx_reference_timestamp = rx_stream_start_timestamp

    return AlignmentArtifacts(
        summary=summary,
        delay_result=result,
        rx_reference_timestamp=rx_reference_timestamp,
    )


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
    artifacts = _compute_alignment(
        tx_pcap=tx_pcap,
        rx_pcap=rx_pcap,
        tx_waveform=tx_waveform,
        rx_capture=rx_capture,
        samp_rate=samp_rate,
        sample_bytes=sample_bytes,
        channel_count=channel_count,
    )
    return artifacts.summary


def _save_alignment_plot(
    artifacts: AlignmentArtifacts,
    *,
    samp_rate: float,
    tx_timestamp: float,
    output: Path,
) -> Path:
    """Render the correlation magnitude with peak and TX markers."""
    _ensure_matplotlib()
    global plt  # type: ignore[global-variable-undefined]

    delay_result = artifacts.delay_result
    time_axis = delay_result.lags / float(samp_rate)
    magnitude = np.abs(delay_result.correlation)
    peak_time = artifacts.summary["time_delay (s)"]

    tx_relative = tx_timestamp - artifacts.rx_reference_timestamp

    # Focus on the correlation peak region. Compute a radius that scales with the
    # observed delay as well as the overall capture span so the plot stays legible
    # for both small and large offsets.
    total_span = float(time_axis[-1] - time_axis[0]) if time_axis.size > 1 else 0.0
    base_radius = max(total_span * 0.01, 5e-4)
    scaled_radius = max(base_radius, min(abs(peak_time) * 0.25, 0.5))
    tx_delta = abs(peak_time - tx_relative)
    radius = max(scaled_radius, tx_delta + base_radius)
    x_min = peak_time - radius
    x_max = peak_time + radius
    # Ensure the TX timestamp marker is visible.
    x_min = min(x_min, tx_relative - base_radius)
    x_max = max(x_max, tx_relative + base_radius)

    mask = (time_axis >= x_min) & (time_axis <= x_max)
    if np.count_nonzero(mask) < 2:
        filtered_time = time_axis
        filtered_magnitude = magnitude
    else:
        filtered_time = time_axis[mask]
        filtered_magnitude = magnitude[mask]

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)  # type: ignore[arg-type]
    ax.plot(filtered_time, filtered_magnitude, label="|corr|")
    ax.axvline(tx_relative, color="tab:blue", linestyle="-.", label="TX VITA timestamp")
    ax.axvline(peak_time, color="tab:red", linestyle="--", label="Correlation peak")

    ax.set_xlabel("Time relative to RX capture start [s]")
    ax.set_ylabel("|Correlation|")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    ax.set_xlim(x_min, x_max)

    fig.savefig(output)
    plt.close(fig)  # type: ignore[union-attr]
    return output


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
        "--plot",
        type=Path,
        help="Optional path to save the correlation magnitude plot with peak/TX markers.",
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

    artifacts = _compute_alignment(
        tx_pcap=args.tx_pcap,
        rx_pcap=args.rx_pcap,
        tx_waveform=args.tx_waveform,
        rx_capture=args.rx_capture,
        samp_rate=args.samp_rate,
        sample_bytes=args.sample_bytes,
        channel_count=args.channels,
    )

    summary = dict(artifacts.summary)
    if args.plot is not None:
        saved_plot = _save_alignment_plot(
            artifacts,
            samp_rate=args.samp_rate,
            tx_timestamp=summary["tx_start_timestamp"],
            output=args.plot,
        )
        if args.json:
            summary["correlation_plot"] = str(saved_plot)
        else:
            print(f"Correlation plot saved to {saved_plot} \n")

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            if key == "time_delay_samples":
                continue
            if key in {
                "tx_start_timestamp",
                "rx_side_tx_timestamp",
                "rx_start_timestamp",
                "rx_first_packet_timestamp",
            }:
                ts = datetime.fromtimestamp(value, tz=ZoneInfo("America/Chicago"))
                # formatted = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
                formatted = ts.strftime("%H:%M:%S.%f")
                print(f"{key}: {formatted}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
