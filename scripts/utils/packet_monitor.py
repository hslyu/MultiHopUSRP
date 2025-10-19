#!/usr/bin/env python3
"""
Lightweight observer for USRP Ethernet data traffic.

This script opens a raw socket on the requested interface and prints a concise
summary of packets that match the configured filters. By default it watches
for traffic exchanged with the N320 radios configured in this repository.
"""
import argparse
import datetime as dt
import select
import socket
import struct
import sys
import time
from typing import Iterable, Optional, Tuple


ETH_P_ALL = 0x0003
ETH_HEADER_LEN = 14
IPV4_ETHERTYPE = 0x0800
UDP_PROTO = 17


def mac_to_str(raw: bytes) -> str:
    """Render a MAC address in the usual colon-delimited format."""
    return ":".join(f"{byte:02x}" for byte in raw)


def parse_ipv4_header(packet: bytes) -> Optional[Tuple[int, int, str, str, int]]:
    """Extract IPv4 header fields needed for filtering and reporting."""
    if len(packet) < ETH_HEADER_LEN + 20:
        return None
    version_ihl = packet[ETH_HEADER_LEN]
    version = version_ihl >> 4
    if version != 4:
        return None
    ihl = (version_ihl & 0x0F) * 4
    if len(packet) < ETH_HEADER_LEN + ihl:
        return None
    total_length = struct.unpack_from("!H", packet, ETH_HEADER_LEN + 2)[0]
    protocol = packet[ETH_HEADER_LEN + 9]
    src_ip = socket.inet_ntoa(packet[ETH_HEADER_LEN + 12 : ETH_HEADER_LEN + 16])
    dst_ip = socket.inet_ntoa(packet[ETH_HEADER_LEN + 16 : ETH_HEADER_LEN + 20])
    return ihl, total_length, src_ip, dst_ip, protocol


def parse_udp_header(packet: bytes, ip_header_len: int) -> Optional[Tuple[int, int]]:
    """Extract UDP source and destination ports."""
    udp_offset = ETH_HEADER_LEN + ip_header_len
    if len(packet) < udp_offset + 8:
        return None
    src_port, dst_port = struct.unpack_from("!HH", packet, udp_offset)
    return src_port, dst_port


def parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor USRP data port traffic via a raw Ethernet socket.",
    )
    parser.add_argument(
        "--interface",
        "-i",
        default="ens4f0np0",
        help=(
            "Network interface that carries the USRP data stream "
            "(default: ens4f0np0)."
        ),
    )
    parser.add_argument(
        "--src-ip",
        action="append",
        default=["192.168.10.1", "192.168.10.2"],
        dest="src_ips",
        help=(
            "Filter by IPv4 source address (repeatable, use `any` to disable; "
            "default monitors 192.168.10.1 and 192.168.10.2)."
        ),
    )
    parser.add_argument(
        "--dst-ip",
        action="append",
        default=["192.168.10.1", "192.168.10.2"],
        dest="dst_ips",
        help=(
            "Filter by IPv4 destination address (repeatable, use `any` to disable; "
            "default monitors 192.168.10.1 and 192.168.10.2)."
        ),
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=0,
        help="Filter by UDP port seen on either side (0 disables the filter).",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="Stop after observing this many matching packets (0 for continuous).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Maximum capture duration in seconds (0 for unlimited).",
    )
    parser.add_argument(
        "--hexdump-bytes",
        type=int,
        default=0,
        help="Print a hex preview of the first N payload bytes (0 disables).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-packet output; useful when only the summary matters.",
    )
    return parser.parse_args(argv)


def should_allow(value: str, allowed: Iterable[str]) -> bool:
    """Return True if filters permit this value."""
    normalized = [item.lower() for item in allowed]
    return "any" in normalized or value in normalized


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    src_filter = args.src_ips or ["any"]
    dst_filter = args.dst_ips or ["any"]

    try:
        raw_sock = socket.socket(
            socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(ETH_P_ALL)
        )
    except PermissionError:
        print(
            "Permission denied: raw sockets require elevated privileges.\n"
            "Re-run with sudo or grant CAP_NET_RAW to this interpreter.",
            file=sys.stderr,
        )
        return 1

    with raw_sock:
        try:
            raw_sock.bind((args.interface, 0))
        except OSError as exc:
            print(f"Failed to bind to interface {args.interface}: {exc}", file=sys.stderr)
            return 1

        print(
            f"Listening on {args.interface} for IPv4 packets "
            f"from {', '.join(src_filter)} to {', '.join(dst_filter)} "
            + (f"with UDP port {args.udp_port}" if args.udp_port else "on any UDP port")
            + ". Press Ctrl-C to stop.",
        )

        observed = 0
        start_time = time.monotonic()
        while True:
            if args.duration > 0.0 and (time.monotonic() - start_time) >= args.duration:
                break
            ready, _, _ = select.select([raw_sock], [], [], 0.5)
            if not ready:
                continue
            packet, _ = raw_sock.recvfrom(65535)
            if len(packet) < ETH_HEADER_LEN:
                continue

            dst_mac, src_mac, eth_type = struct.unpack_from("!6s6sH", packet)
            if eth_type != IPV4_ETHERTYPE:
                continue

            ipv4_info = parse_ipv4_header(packet)
            if ipv4_info is None:
                continue
            ip_header_len, total_length, src_ip, dst_ip, protocol = ipv4_info

            if not should_allow(src_ip, src_filter):
                continue
            if not should_allow(dst_ip, dst_filter):
                continue
            if protocol != UDP_PROTO:
                continue

            udp_info = parse_udp_header(packet, ip_header_len)
            if udp_info is None:
                continue
            src_port, dst_port = udp_info

            if args.udp_port and args.udp_port not in (src_port, dst_port):
                continue

            observed += 1
            if not args.quiet:
                timestamp = dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                payload_start = ETH_HEADER_LEN + ip_header_len + 8
                payload = packet[payload_start:]
                summary = (
                    f"{timestamp} len={len(packet)} "
                    f"{mac_to_str(src_mac)} -> {mac_to_str(dst_mac)} "
                    f"{src_ip}:{src_port} -> {dst_ip}:{dst_port} "
                    f"total_len={total_length}"
                )
                print(summary)
                if args.hexdump_bytes > 0 and payload:
                    preview = payload[: args.hexdump_bytes]
                    hex_bytes = " ".join(f"{byte:02x}" for byte in preview)
                    print(f"  payload[{len(preview)}]: {hex_bytes}")

            if args.max_packets > 0 and observed >= args.max_packets:
                break

    print(f"Captured {observed} matching packet(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
