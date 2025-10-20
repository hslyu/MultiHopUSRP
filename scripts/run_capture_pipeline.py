#!/usr/bin/env python3
"""
Automate TX/RX flowgraph runs while recording NIC traffic with dumpcap.

This helper starts dumpcap with appropriate filters, runs the chosen flowgraph,
and then gracefully stops dumpcap so that UHD VITA-49 packets (with optional
hardware timestamps) are preserved in a .pcac file for post-processing.

Example (TX capture):
    python scripts/run_capture_pipeline.py tx \
        --interface ens4f0np0 \
        --pcap outputs/tx_capture.pcap \
        --udp-port 49152 \
        --timestamp-mode adapter_unsynced \
        --tx-args --waveform-file outputs/prach_long.npy --gain 20 --freq 2.45e9

Example (RX capture):
    python scripts/run_capture_pipeline.py rx \
        --interface ens4f1np1 \
        --pcap outputs/rx_capture.pcap \
        --udp-port 49153 \
        --timestamp-mode adapter_unsynced \
        --rx-args --duration 10 --output captures/prach_rx.c32 --gain 30

Example (session capture with alignment):
    python scripts/run_capture_pipeline.py session \
        --rx-interface ens4f1np1 \
        --rx-pcap outputs/rx_capture.pcap \
        --rx-udp-port 49153 \
        --rx-args --duration 5 --output captures/prach_rx.c32 --gain 20 \
        --tx-interface ens4f0np0 \
        --tx-pcap outputs/tx_capture.pcap \
        --tx-udp-port 49152 \
        --tx-args --waveform-file outputs/prach_long.npy --gain 10 --freq 2.45e9 \
        --timestamp-mode adapter_unsynced \
        --tx-delay 1.0 \
        --align-output outputs/alignment.json \
        --align-tx-waveform outputs/prach_long.npy \
        --align-rx-capture captures/prach_rx.c32 \
        --align-samp-rate 1e6
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from scripts.align_delay import align_delays  # type: ignore
except Exception:  # noqa: F401
    align_delays = None  # type: ignore


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


def _split_arg_string(arg_string: str) -> list[str]:
    return shlex.split(arg_string) if arg_string else []


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


def _start_flowgraph(cmd: Sequence[str]) -> subprocess.Popen:
    print(f"Starting flowgraph: {' '.join(shlex.quote(part) for part in cmd)}")
    return subprocess.Popen(cmd)


def _build_session_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "session",
        help="Capture RX first, then trigger TX while both dumpcaps run.",
    )
    parser.add_argument("--rx-interface", required=True)
    parser.add_argument("--rx-pcap", type=Path, required=True)
    parser.add_argument("--rx-udp-port", type=int, default=49153)
    parser.add_argument(
        "--rx-args",
        default="",
        help="Quoted string of arguments passed to rx_flowgraph.py.",
    )

    parser.add_argument("--tx-interface", required=True)
    parser.add_argument("--tx-pcap", type=Path, required=True)
    parser.add_argument("--tx-udp-port", type=int, default=49152)
    parser.add_argument(
        "--tx-args",
        default="",
        help="Quoted string of arguments passed to tx_flowgraph.py.",
    )

    parser.add_argument(
        "--timestamp-mode",
        default=None,
        help="dumpcap timestamp type (applies to both interfaces).",
    )
    parser.add_argument(
        "--tx-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after RX starts before launching TX.",
    )
    parser.add_argument(
        "--use-sudo",
        action="store_true",
        help="Prefix dumpcap commands with sudo -E.",
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
        help="Extra arguments appended to both dumpcap commands.",
    )

    parser.add_argument(
        "--align-output",
        type=Path,
        help="Path to store alignment JSON (requires scripts.align_delay).",
    )
    parser.add_argument("--align-tx-waveform", type=Path)
    parser.add_argument("--align-rx-capture", type=Path)
    parser.add_argument("--align-samp-rate", type=float)
    parser.add_argument(
        "--align-sample-bytes",
        type=int,
        default=8,
        help="Bytes per complex sample for alignment (default: 8).",
    )
    parser.add_argument(
        "--align-channels",
        type=int,
        default=1,
        help="Channel count for alignment (default: 1).",
    )


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

    _build_session_parser(subparsers)

    return parser


def _require_alignment_args(args: argparse.Namespace) -> None:
    if not args.align_output:
        return
    if align_delays is None:
        raise RuntimeError(
            "Alignment requested but scripts.align_delay is unavailable or failed to import."
        )
    missing = []
    if args.align_tx_waveform is None:
        missing.append("--align-tx-waveform")
    if args.align_rx_capture is None:
        missing.append("--align-rx-capture")
    if args.align_samp_rate is None:
        missing.append("--align-samp-rate")
    if missing:
        raise ValueError(
            "Alignment requested but missing parameters: " + ", ".join(missing)
        )


def _session_command(args: argparse.Namespace) -> None:
    _require_alignment_args(args)

    rx_cmd = _build_dumpcap_cmd(
        dumpcap_path="dumpcap",
        interface=args.rx_interface,
        pcap_path=args.rx_pcap,
        udp_port=args.rx_udp_port,
        use_sudo=args.use_sudo,
        ring_buffer_size_mb=args.ring_buffer_mb,
        extra_flags=args.extra_dumpcap_flags,
        timestamp_mode=args.timestamp_mode,
    )
    tx_cmd = _build_dumpcap_cmd(
        dumpcap_path="dumpcap",
        interface=args.tx_interface,
        pcap_path=args.tx_pcap,
        udp_port=args.tx_udp_port,
        use_sudo=args.use_sudo,
        ring_buffer_size_mb=args.ring_buffer_mb,
        extra_flags=args.extra_dumpcap_flags,
        timestamp_mode=args.timestamp_mode,
    )

    rx_flowgraph_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "rx_flowgraph.py"),
    ]
    rx_flowgraph_cmd.extend(_split_arg_string(args.rx_args))

    tx_flowgraph_cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "tx_flowgraph.py"),
    ]
    tx_flowgraph_cmd.extend(_split_arg_string(args.tx_args))

    if len(rx_flowgraph_cmd) == 2:
        raise ValueError("Provide rx_flowgraph arguments after --rx-args.")
    if len(tx_flowgraph_cmd) == 2:
        raise ValueError("Provide tx_flowgraph arguments after --tx-args.")

    rx_dumpcap = _start_dumpcap(rx_cmd)
    tx_dumpcap: Optional[subprocess.Popen] = None
    rx_proc: Optional[subprocess.Popen] = None
    tx_proc: Optional[subprocess.Popen] = None

    try:
        rx_proc = _start_flowgraph(rx_flowgraph_cmd)
        time.sleep(max(0.0, args.tx_delay))

        tx_dumpcap = _start_dumpcap(tx_cmd)
        tx_proc = _start_flowgraph(tx_flowgraph_cmd)

        tx_proc.wait()
        _stop_dumpcap(tx_dumpcap)
        tx_dumpcap = None

        rx_proc.wait()
    finally:
        if tx_proc and tx_proc.poll() is None:
            tx_proc.terminate()
        if rx_proc and rx_proc.poll() is None:
            rx_proc.terminate()
        if tx_dumpcap:
            _stop_dumpcap(tx_dumpcap)
        if rx_dumpcap:
            _stop_dumpcap(rx_dumpcap)

    if args.align_output:
        summary = align_delays(  # type: ignore[arg-type]
            tx_pcap=args.tx_pcap,
            rx_pcap=args.rx_pcap,
            tx_waveform=args.align_tx_waveform,
            rx_capture=args.align_rx_capture,
            samp_rate=args.align_samp_rate,
            sample_bytes=args.align_sample_bytes,
            channel_count=args.align_channels,
        )
        args.align_output.parent.mkdir(parents=True, exist_ok=True)
        args.align_output.write_text(json.dumps(summary, indent=2))
        print(f"Alignment written to {args.align_output}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "session":
        _ensure_parent(args.rx_pcap)
        _ensure_parent(args.tx_pcap)
        if args.align_output:
            _ensure_parent(args.align_output)
        _session_command(args)
        return

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
        flowgraph_cmd = _tx_command(args) if args.mode == "tx" else _rx_command(args)

        if len(flowgraph_cmd) <= 2:
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
