#!/usr/bin/env python3
"""
Transmit a baseband waveform through a USRP using pre-generated samples.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from gnuradio import blocks
from gnuradio import gr
from gnuradio import uhd


@dataclass
class TxSettings:
    """Runtime configuration for the transmit flowgraph."""

    addr: str = "addr=192.168.10.1"
    mgmt_addr: str = "mgmt_addr=192.168.100.8"
    freq: float = 2.45e9
    samp_rate: float = 1e6
    gain: float = 10.0
    tx_antenna: str = "TX/RX"
    clock_source: str = ""
    time_source: str = ""
    lo_offset: Optional[float] = None
    device_args: str = "serial=34596FE"
    tx_iface: str = "ens4f0np0"
    waveform_path: Optional[Path] = None
    repeat_waveform: bool = False
    _cached_waveform: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def device_address(self) -> str:
        """Compose a UHD device string from the configured parts."""
        return ",".join(filter(None, [self.addr, self.mgmt_addr, self.device_args]))

    def load_waveform(self) -> np.ndarray:
        """Load complex samples from the configured waveform file."""
        if self.waveform_path is None:
            raise ValueError("waveform_path must be provided.")
        path = self.waveform_path
        if path.suffix != ".npy":
            raise ValueError(
                f"Unsupported waveform file '{path}'. Provide a .npy file generated "
                "by scripts/waveform_generator.py."
            )
        if not path.exists():
            raise FileNotFoundError(f"Waveform file '{path}' does not exist.")

        if self._cached_waveform is None:
            samples = np.load(path)
            if samples.ndim != 1:
                raise ValueError("Waveform array must be 1-D.")
            if not np.iscomplexobj(samples):
                raise ValueError("Waveform array must contain complex samples.")
            self._cached_waveform = samples.astype(np.complex64)

        if self._cached_waveform.size == 0:
            raise ValueError("Waveform is empty; nothing to transmit.")

        return self._cached_waveform


class TxFlowgraph(gr.top_block):
    """Transmit a finite set of complex samples."""

    def __init__(self, config: TxSettings):
        super().__init__()
        self.config = config

        device_addr = self.config.device_address()
        self.usrp = uhd.usrp_sink(
            device_addr,
            uhd.stream_args(cpu_format="fc32", channels=[0]),
        )

        self.usrp.set_samp_rate(self.config.samp_rate)
        self.usrp.set_center_freq(self.config.freq, 0)
        self.usrp.set_gain(self.config.gain, 0)
        self.usrp.set_antenna(self.config.tx_antenna, 0)

        if self.config.clock_source:
            self.usrp.set_clock_source(self.config.clock_source, 0)
        if self.config.time_source:
            self.usrp.set_time_source(self.config.time_source, 0)
        if self.config.lo_offset is not None:
            self.usrp.set_lo_offset(self.config.lo_offset, 0)

        samples = self.config.load_waveform().tolist()
        self.packet_source = blocks.vector_source_c(
            samples, self.config.repeat_waveform, 1, []
        )

        self.connect(self.packet_source, self.usrp)


class TxApplication:
    """High-level wrapper that builds and runs the transmit flowgraph."""

    def __init__(self, config: Optional[TxSettings] = None):
        self.config = config or TxSettings()
        self.flowgraph: Optional[TxFlowgraph] = None

    def build_flowgraph(self) -> TxFlowgraph:
        """Instantiate the GNU Radio flowgraph with the configured settings."""
        return TxFlowgraph(self.config)

    def _read_tx_packets(self) -> Optional[int]:
        """Read the NIC's transmitted packet counter from sysfs."""
        stats_path = (
            Path("/sys/class/net") / self.config.tx_iface / "statistics" / "tx_packets"
        )
        try:
            return int(stats_path.read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            return None

    def run(self) -> None:
        """Start the flowgraph and block until interrupted."""
        start_packets = self._read_tx_packets()
        if start_packets is None:
            print(
                f"Warning: Unable to read TX packets from interface {self.config.tx_iface}."
            )

        self.flowgraph = self.build_flowgraph()
        tb = self.flowgraph
        try:
            tb.start()
            source_desc = self.config.waveform_path or "<waveform>"
            print(f"Transmitting waveform from {source_desc}. Press Ctrl-C to stop.")
            tb.wait()
        except KeyboardInterrupt:
            print("\nStopping transmission.")
        finally:
            tb.stop()
            tb.wait()

        end_packets = self._read_tx_packets()
        if start_packets is not None and end_packets is not None:
            delta = end_packets - start_packets
            print(
                f"{delta} packets transmitted via {self.config.tx_iface} "
                f"(start={start_packets}, end={end_packets})."
            )


def transmit_waveform(config: TxSettings) -> None:
    """
    Convenience helper to run the transmit application with the provided settings.
    """
    config.load_waveform()
    TxApplication(config).run()


def send_waveform(
    waveform_path: Path,
    *,
    repeat: bool = False,
    addr: str = TxSettings.addr,
    mgmt_addr: str = TxSettings.mgmt_addr,
    freq: float = TxSettings.freq,
    samp_rate: float = TxSettings.samp_rate,
    gain: float = TxSettings.gain,
    tx_antenna: str = TxSettings.tx_antenna,
    clock_source: str = TxSettings.clock_source,
    time_source: str = TxSettings.time_source,
    lo_offset: Optional[float] = TxSettings.lo_offset,
    device_args: str = TxSettings.device_args,
    tx_iface: str = TxSettings.tx_iface,
) -> None:
    """
    Convenience wrapper to transmit a waveform from disk with minimal setup.

    Example:
        send_waveform(Path(\"outputs/tone.npy\"), repeat=True, gain=5.0)
    """
    settings = TxSettings(
        addr=addr,
        mgmt_addr=mgmt_addr,
        freq=freq,
        samp_rate=samp_rate,
        gain=gain,
        tx_antenna=tx_antenna,
        clock_source=clock_source,
        time_source=time_source,
        lo_offset=lo_offset,
        device_args=device_args,
        tx_iface=tx_iface,
        waveform_path=waveform_path,
        repeat_waveform=repeat,
    )
    transmit_waveform(settings)


__all__ = [
    "TxSettings",
    "TxFlowgraph",
    "TxApplication",
    "transmit_waveform",
    "send_waveform",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transmit a waveform stored as complex64 .npy samples."
    )
    parser.add_argument(
        "--waveform-file",
        type=Path,
        required=True,
        help="Path to a complex64 .npy file generated by waveform_generator.py.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Continuously repeat the waveform until interrupted.",
    )
    parser.add_argument(
        "--addr",
        default=TxSettings.addr,
        help=f"Primary device address string (default: {TxSettings.addr}).",
    )
    parser.add_argument(
        "--mgmt-addr",
        default=TxSettings.mgmt_addr,
        help=f"Management address string (default: {TxSettings.mgmt_addr}).",
    )
    parser.add_argument(
        "--device-args",
        default=TxSettings.device_args,
        help=f"Additional UHD device args (default: {TxSettings.device_args}).",
    )
    parser.add_argument(
        "--tx-iface",
        default=TxSettings.tx_iface,
        help=f"Host NIC used for stats (default: {TxSettings.tx_iface}).",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=TxSettings.freq,
        help=f"RF center frequency in Hz (default: {TxSettings.freq}).",
    )
    parser.add_argument(
        "--samp-rate",
        type=float,
        default=TxSettings.samp_rate,
        help=f"Sample rate in Sa/s (default: {TxSettings.samp_rate}).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=TxSettings.gain,
        help=f"Transmit gain in dB (default: {TxSettings.gain}).",
    )
    parser.add_argument(
        "--tx-antenna",
        default=TxSettings.tx_antenna,
        help=f"Transmit antenna port (default: {TxSettings.tx_antenna}).",
    )
    parser.add_argument(
        "--clock-source",
        default=TxSettings.clock_source,
        help=(
            "External clock source identifier (e.g. 'gpsdo'). "
            "Defaults to internal clock."
        ),
    )
    parser.add_argument(
        "--time-source",
        default=TxSettings.time_source,
        help=(
            "External time source identifier (e.g. 'gpsdo'). "
            "Defaults to internal time."
        ),
    )
    parser.add_argument(
        "--lo-offset",
        type=float,
        default=None,
        help="Optional LO offset in Hz.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    settings = TxSettings(
        addr=args.addr,
        mgmt_addr=args.mgmt_addr,
        freq=args.freq,
        samp_rate=args.samp_rate,
        gain=args.gain,
        tx_antenna=args.tx_antenna,
        clock_source=args.clock_source,
        time_source=args.time_source,
        lo_offset=args.lo_offset,
        device_args=args.device_args,
        tx_iface=args.tx_iface,
        waveform_path=args.waveform_file,
        repeat_waveform=args.repeat,
    )
    transmit_waveform(settings)


if __name__ == "__main__":
    main()
