#!/usr/bin/env python3
"""
Capture complex samples from a USRP using a class-based flowgraph wrapper.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gnuradio import blocks, gr, uhd


@dataclass
class RxSettings:
    """Configuration for the receive flowgraph."""

    addr: str = "addr=192.168.10.2"
    mgmt_addr: str = "mgmt_addr=192.168.100.7"
    freq: float = 2.45e9
    samp_rate: float = 1e6
    gain: float = 20.0
    rx_antenna: str = "RX2"
    clock_source: str = ""
    time_source: str = ""
    lo_offset: Optional[float] = None
    duration: float = 10.0
    output: Path = Path("capture.c32")
    sync_start: bool = False
    device_args: str = "serial=345F127"

    def device_address(self) -> str:
        """Compose a UHD device string from the configured parts."""
        return ",".join(filter(None, [self.addr, self.mgmt_addr, self.device_args]))


class RxFlowgraph(gr.top_block):
    """Receive complex samples from a USRP and store them on disk."""

    def __init__(self, config: RxSettings):
        super().__init__()
        self.config = config

        device_addr = self.config.device_address()
        self.usrp = uhd.usrp_source(
            device_addr,
            uhd.stream_args(cpu_format="fc32", channels=[0]),
        )

        self.usrp.set_samp_rate(self.config.samp_rate)
        self.usrp.set_center_freq(self.config.freq, 0)
        self.usrp.set_gain(self.config.gain, 0)
        self.usrp.set_antenna(self.config.rx_antenna, 0)

        if self.config.clock_source:
            self.usrp.set_clock_source(self.config.clock_source, 0)
        if self.config.time_source:
            self.usrp.set_time_source(self.config.time_source, 0)
        if self.config.lo_offset is not None:
            self.usrp.set_lo_offset(self.config.lo_offset, 0)

        if self.config.sync_start:
            self.usrp.set_time_now(uhd.time_spec(0.0))

        output_path = self.config.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.sink = blocks.file_sink(
            gr.sizeof_gr_complex,
            str(output_path),
            False,
        )
        self.sink.set_unbuffered(False)

        if self.config.duration > 0.0:
            num_samples = int(self.config.duration * self.config.samp_rate)
            self.head = blocks.head(gr.sizeof_gr_complex, num_samples)
            self.connect(self.usrp, self.head, self.sink)
        else:
            self.connect(self.usrp, self.sink)


class RxApplication:
    """High-level wrapper that builds and runs the receive flowgraph."""

    def __init__(self, config: Optional[RxSettings] = None):
        self.config = config or RxSettings()
        self.flowgraph: Optional[RxFlowgraph] = None

    def build_flowgraph(self) -> RxFlowgraph:
        """Instantiate the GNU Radio flowgraph with the configured settings."""
        return RxFlowgraph(self.config)

    def run(self) -> None:
        """Start the flowgraph and block until completion or interruption."""
        self.flowgraph = self.build_flowgraph()
        tb = self.flowgraph
        try:
            tb.start()
            msg = (
                f"Capturing {self.config.duration} s of data to {self.config.output}"
                if self.config.duration > 0.0
                else f"Capturing continuously to {self.config.output}"
            )
            print(msg + ". Press Ctrl-C to stop.")
            start_time = time.time()
            tb.wait()
            if self.config.duration > 0.0:
                elapsed = time.time() - start_time
                print(f"Capture complete in {elapsed:.2f} s.")
        except KeyboardInterrupt:
            print("\nStopping capture.")
        finally:
            tb.stop()
            tb.wait()


def capture_samples(config: RxSettings) -> None:
    """
    Convenience helper to run the receive application with the provided settings.
    """
    RxApplication(config).run()


def record_samples(
    output: Path,
    *,
    duration: float = RxSettings.duration,
    addr: str = RxSettings.addr,
    mgmt_addr: str = RxSettings.mgmt_addr,
    device_args: str = RxSettings.device_args,
    freq: float = RxSettings.freq,
    samp_rate: float = RxSettings.samp_rate,
    gain: float = RxSettings.gain,
    rx_antenna: str = RxSettings.rx_antenna,
    clock_source: str = RxSettings.clock_source,
    time_source: str = RxSettings.time_source,
    lo_offset: Optional[float] = RxSettings.lo_offset,
    sync_start: bool = RxSettings.sync_start,
) -> None:
    """
    Convenience wrapper to record samples with minimal setup.

    Example:
        record_samples(Path("captures/zc.c32"), duration=0.1)
    """
    settings = RxSettings(
        addr=addr,
        mgmt_addr=mgmt_addr,
        device_args=device_args,
        freq=freq,
        samp_rate=samp_rate,
        gain=gain,
        rx_antenna=rx_antenna,
        clock_source=clock_source,
        time_source=time_source,
        lo_offset=lo_offset,
        duration=duration,
        output=output,
        sync_start=sync_start,
    )
    capture_samples(settings)


__all__ = [
    "RxSettings",
    "RxFlowgraph",
    "RxApplication",
    "capture_samples",
    "record_samples",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture samples from a USRP and save them to disk."
    )
    parser.add_argument(
        "--addr",
        default=RxSettings.addr,
        help=f"Primary device address string (default: {RxSettings.addr}).",
    )
    parser.add_argument(
        "--mgmt-addr",
        default=RxSettings.mgmt_addr,
        help=f"Management address string (default: {RxSettings.mgmt_addr}).",
    )
    parser.add_argument(
        "--device-args",
        default=RxSettings.device_args,
        help=f"Additional UHD device args (default: {RxSettings.device_args}).",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=RxSettings.freq,
        help=f"RF center frequency in Hz (default: {RxSettings.freq}).",
    )
    parser.add_argument(
        "--samp-rate",
        type=float,
        default=RxSettings.samp_rate,
        help=f"Sample rate in Sa/s (default: {RxSettings.samp_rate}).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=RxSettings.gain,
        help=f"Receive gain in dB (default: {RxSettings.gain}).",
    )
    parser.add_argument(
        "--rx-antenna",
        default=RxSettings.rx_antenna,
        help=f"Receive antenna port (default: {RxSettings.rx_antenna}).",
    )
    parser.add_argument(
        "--clock-source",
        default=RxSettings.clock_source,
        help=(
            "External clock source identifier (e.g. 'gpsdo'). "
            "Defaults to internal clock."
        ),
    )
    parser.add_argument(
        "--time-source",
        default=RxSettings.time_source,
        help=(
            "External time source identifier (e.g. 'gpsdo'). Defaults to internal time."
        ),
    )
    parser.add_argument(
        "--lo-offset",
        type=float,
        default=None,
        help="Optional LO offset in Hz.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=RxSettings.duration,
        help=(
            "Capture duration in seconds. Use 0 for continuous capture until Ctrl-C. "
            f"(default: {RxSettings.duration})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RxSettings.output,
        help=(
            "Output filename for raw complex samples. Directories are created "
            "if needed."
        ),
    )
    parser.add_argument(
        "--sync-start",
        action="store_true",
        help="Reset device time to 0 before capture for timed operation.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    settings = RxSettings(
        addr=args.addr,
        mgmt_addr=args.mgmt_addr,
        device_args=args.device_args,
        freq=args.freq,
        samp_rate=args.samp_rate,
        gain=args.gain,
        rx_antenna=args.rx_antenna,
        clock_source=args.clock_source,
        time_source=args.time_source,
        lo_offset=args.lo_offset,
        duration=args.duration,
        output=args.output,
        sync_start=args.sync_start,
    )

    capture_samples(settings)


if __name__ == "__main__":
    main()
