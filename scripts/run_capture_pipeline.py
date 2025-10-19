#!/usr/bin/env python3
"""
Automate TX/RX flowgraph runs while recording NIC traffic with dumpcap.

This helper starts dumpcap with appropriate filters, runs the chosen flowgraph,
and then gracefully stops dumpcap so that UHD VITA-49 packets (with optional
hardware timestamps) are preserved in a .pcap file for post-processing.

Example (TX capture):
    python scripts/run_capture_pipeline.py tx \
        --interface ens4f0np0 \
        --pcap outputs/tx_capture.pcap \
        --udp-port 49152 \
        --waveform-file outputs/prach_long.npy \
        --tx-args --gain 20 --freq 2.45e9

Example (RX capture):
    python scripts/run_capture_pipeline.py rx \
        --interface ens4f1np1 \
        --pcap outputs/rx_capture.pcap \
        --udp-port 49153 \
        --duration 10 \
        --output captures/prach_rx.c32 \
        --rx-args --gain 30
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def _build_dumpcap_cmd(
    *,
    dumpcap_path: str,
    interface: str,
    pcap_path: Path,
    udp_port: int,
    use_sudo: bool,
    ring_buffer_size_mb: float,
    extra_flags: Sequence[str],
    timestamp_mode: str | None,
) -> list[str]:
    """Construct the dumpcap command with sensible defaults."""
    cmd: list[str] = []

    if use_sudo:
        cmd.extend(["sudo", "-E"])

    cmd.extend(
        [
            dumpcap_path,
            "-i",
            interface,
            "-f",
            f"udp port {udp_port}",
            "-w",
            str(pcap_path),
            "-s",
            "0",
            "-q",
        ]
    )

    if timestamp_mode:
        cmd.extend(["--time-stamp-type", timestamp_mode])

    if ring_buffer_size_mb > 0:
        cmd.extend(["-B", str(ring_buffer_size_mb)])

    cmd.extend(extra_flags)
    return cmd


def _start_dumpcap(cmd: Sequence[str]) -> subprocess.Popen:
    """Launch dumpcap and return the process handle."""
    print(f"Starting dumpcap: {' '.join(shlex.quote(part) for part in cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def _stop_dumpcap(process: subprocess.Popen) -> None:
    """Terminate dumpcap gracefully."""
    if process.poll() is not None:
        return

    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def _run_subprocess(cmd: Sequence[str]) -> None:
    """Execute a subprocess and stream its output."""
    print(f"Running: {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, check=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_remaining_args(args: Sequence[str]) -> list[str]:
    """Return args if provided, otherwise an empty list."""
    return list(args) if args else []


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interface",
        required=True,
        help="Network interface connected to the USRP (e.g., ens4f0np0).",
    )
    parser.add_argument(
        "--pcap",
        type=Path,
        required=True,
        help="Destination .pcap file for captured packets.",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=49152,
        help="UDP port carrying UHD VITA-49 traffic (default: 49152).",
    )
    parser.add_argument(
        "--dumpcap-path",
        default="dumpcap",
        help="Path to dumpcap (default: dumpcap in PATH).",
    )
    parser.add_argument(
        "--use-sudo",
        action="store_true",
        help="Prefix dumpcap with sudo -E (requires password unless passwordless sudo is configured).",
    )
    parser.add_argument(
        "--ring-buffer-mb",
        type=float,
        default=0.0,
        help="Optional dumpcap ring buffer size in MB.",
    )
    parser.add_argument(
        "--extra-dumpcap-flags",
        nargs="*",
        default=[],
        help="Additional flags forwarded to dumpcap.",
    )
    parser.add_argument(
        "--timestamp-mode",
        default=None,
        help="Optional dumpcap timestamp type (e.g., 'adapter_unsynced').",
    )


def _tx_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "tx_flowgraph.py"),
    ]
    cmd.extend(_parse_remaining_args(args.tx_args))
    return cmd


def _rx_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "rx_flowgraph.py"),
    ]
    cmd.extend(_parse_remaining_args(args.rx_args))
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture UHD traffic with dumpcap while running TX/RX flowgraphs."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    tx_parser = subparsers.add_parser("tx", help="Capture packets during transmission.")
    _add_shared_arguments(tx_parser)
    tx_parser.add_argument(
        "--tx-args",
        nargs=argparse.REMAINDER,
        help="Arguments passed verbatim to tx_flowgraph.py (prefix with --).",
    )

    rx_parser = subparsers.add_parser("rx", help="Capture packets during reception.")
    _add_shared_arguments(rx_parser)
    rx_parser.add_argument(
        "--rx-args",
        nargs=argparse.REMAINDER,
        help="Arguments passed verbatim to rx_flowgraph.py (prefix with --).",
    )

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _ensure_parent(Path(args.pcap))

    dumpcap_cmd = _build_dumpcap_cmd(
        dumpcap_path=args.dumpcap_path,
        interface=args.interface,
        pcap_path=args.pcap,
        udp_port=args.udp_port,
        use_sudo=args.use_sudo,
        ring_buffer_size_mb=args.ring_buffer_mb,
        extra_flags=args.extra_dumpcap_flags,
        timestamp_mode=args.timestamp_mode,
    )

    dumpcap_proc = _start_dumpcap(dumpcap_cmd)
    try:
        if args.mode == "tx":
            flowgraph_cmd = _tx_command(args)
        else:
            flowgraph_cmd = _rx_command(args)

        if not flowgraph_cmd:
            raise ValueError(
                "No flowgraph arguments provided. Supply options after '--tx-args' or '--rx-args'."
            )

        _run_subprocess(flowgraph_cmd)
    finally:
        _stop_dumpcap(dumpcap_proc)
        if dumpcap_proc.stdout:
            dumpcap_proc.stdout.close()
        if dumpcap_proc.stderr:
            stderr = dumpcap_proc.stderr.read().decode("utf-8", errors="replace")
            dumpcap_proc.stderr.close()
            if stderr.strip():
                print(f"[dumpcap stderr]\n{stderr}")


if __name__ == "__main__":
    main()
