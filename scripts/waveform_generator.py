#!/usr/bin/env python3
"""
Generate baseband waveforms (monotone or Zadoff-Chu) for MultiHopUSRP tests.

This tool can export complex samples to disk and create amplitude/phase plots
for quick visual verification before transmission.
"""

import argparse
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class WaveformSettings:
    """Parameters that define a packetized baseband waveform."""

    waveform: str = "nr_prach_long"  # options: "nr_prach_long", "zadoff_chu", "monotone"
    amplitude: float = 0.5
    tone_freq: float = 100e3
    samp_rate: float = 1e6
    packet_len: int = 839
    num_packets: int = 1
    guard_len: int = 0  # optional zero padding between packets
    zc_root: int = 1
    nr_prach_m_count: int = 64
    nr_prach_m_index: int = 0
    nr_prach_cp_len: int = 3168
    nr_prach_ncs: Optional[int] = None

    def build(self) -> np.ndarray:
        """Generate the complex waveform according to the configuration."""
        if self.packet_len <= 0:
            raise ValueError("packet_len must be positive.")
        if self.num_packets <= 0:
            raise ValueError("num_packets must be positive.")

        if self.waveform == "zadoff_chu":
            packet = self._build_zadoff_chu_packet()
        elif self.waveform == "monotone":
            packet = self._build_monotone_packet()
        elif self.waveform == "nr_prach_long":
            packet = self._build_nr_prach_long_packet()
        else:
            raise ValueError(f"Unsupported waveform '{self.waveform}'.")

        if self.guard_len < 0:
            raise ValueError("guard_len must be non-negative.")
        if self.guard_len:
            guard = np.zeros(self.guard_len, dtype=np.complex64)
            packet = np.concatenate([packet, guard])

        waveform = np.tile(packet, self.num_packets)
        return waveform.astype(np.complex64)

    def _build_monotone_packet(self) -> np.ndarray:
        """Create a single complex exponential packet for the monotone case."""
        phase_step = 2.0 * np.pi * self.tone_freq / self.samp_rate
        indices = np.arange(self.packet_len, dtype=np.float32)
        packet = self.amplitude * np.exp(1j * phase_step * indices)
        return packet.astype(np.complex64)

    def _build_zadoff_chu_packet(self) -> np.ndarray:
        """Create a single Zadoff-Chu packet."""
        root = int(self.zc_root) % self.packet_len
        if root == 0 or gcd(root, self.packet_len) != 1:
            raise ValueError(
                f"zc_root ({self.zc_root}) must be coprime with packet_len ({self.packet_len})."
            )
        n = np.arange(self.packet_len, dtype=np.float32)
        phase = -np.pi * root * n * (n + 1) / self.packet_len
        packet = self.amplitude * np.exp(1j * phase)
        return packet.astype(np.complex64)

    def _build_nr_prach_long_packet(self) -> np.ndarray:
        """Create a 5G NR long PRACH preamble with CP and cyclic shift."""
        n_zc = self.packet_len
        if n_zc != 839:
            raise ValueError(
                f"NR long PRACH requires packet_len=839, received {self.packet_len}."
            )

        root = int(self.zc_root) % n_zc
        if root == 0 or gcd(root, n_zc) != 1:
            raise ValueError(f"zc_root ({self.zc_root}) must be coprime with 839.")

        n = np.arange(n_zc, dtype=np.float32)
        phase = -np.pi * root * n * (n + 1) / n_zc
        base = np.exp(1j * phase)

        m_count = max(1, int(self.nr_prach_m_count))
        m_index = int(self.nr_prach_m_index) % m_count
        if self.nr_prach_ncs is not None:
            n_cs = int(self.nr_prach_ncs)
            if n_cs <= 0:
                raise ValueError("nr_prach_ncs must be positive when provided.")
        else:
            n_cs = max(1, n_zc // m_count)

        cyclic_shift = (m_index * n_cs) % n_zc
        preamble = np.roll(base, -cyclic_shift)
        preamble = (self.amplitude * preamble).astype(np.complex64)

        cp_len = max(0, int(self.nr_prach_cp_len))
        if cp_len:
            if cp_len <= n_zc:
                cp = preamble[-cp_len:]
            else:
                repeats = (cp_len + n_zc - 1) // n_zc
                extended = np.tile(preamble, repeats + 1)
                cp = extended[-cp_len:]
            packet = np.concatenate([cp, preamble])
        else:
            packet = preamble

        return packet.astype(np.complex64)


def _format_complex(value: complex) -> str:
    return f"{value.real:+.6f} {value.imag:+.6f}j"


def print_samples(samples: np.ndarray, limit: Optional[int] = None) -> None:
    """Print complex samples up to 'limit' entries."""
    total = samples.size
    count = total if limit is None else min(limit, total)
    for idx in range(count):
        print(f"n={idx:4d}: {_format_complex(samples[idx])}")
    if count < total:
        print(f"... truncated at {count} / {total} samples ...")


def compute_autocorrelation(
    samples: np.ndarray, max_lag: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return autocorrelation values and corresponding lags."""
    if samples.size == 0:
        raise ValueError("Sequence is empty; autocorrelation undefined.")

    autocorr = np.correlate(samples, samples, mode="full")
    lags = np.arange(-samples.size + 1, samples.size, dtype=np.int32)
    if max_lag is not None:
        mask = np.abs(lags) <= max_lag
        autocorr = autocorr[mask]
        lags = lags[mask]
    return lags, autocorr


def print_autocorrelation(samples: np.ndarray, max_lag: Optional[int] = None) -> None:
    """Print the autocorrelation sequence, optionally limited to ±max_lag."""
    try:
        lags, autocorr = compute_autocorrelation(samples, max_lag)
    except ValueError:
        print("Sequence is empty; autocorrelation skipped.")
        return

    peak = np.max(np.abs(autocorr))
    for lag, value in zip(lags, autocorr):
        magnitude = np.abs(value)
        norm_mag = magnitude / peak if peak > 0 else 0.0
        print(
            f"lag={lag:+5d}: {_format_complex(value)} |mag|={magnitude:.6f} "
            f"norm={norm_mag:.6f}"
        )


def export_waveform_plot(
    samples: np.ndarray,
    output_path: Path,
    sample_count: Optional[int] = None,
) -> Path:
    """
    Export an IQ constellation view and wrapped phase plot for the samples.
    """
    if samples.size == 0:
        raise ValueError("Waveform is empty; nothing to plot.")

    plot_samples = samples
    if sample_count is not None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive when provided.")
        plot_samples = plot_samples[:sample_count]

    max_plot = min(plot_samples.size, 1000)
    plot_samples = plot_samples[:max_plot]
    indices = np.arange(plot_samples.size, dtype=np.int32)
    iq_real = plot_samples.real
    iq_imag = plot_samples.imag
    amplitudes = np.abs(plot_samples)
    phases = np.unwrap(np.angle(plot_samples))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )

    axes[0].plot(indices, iq_real)
    axes[0].set_ylabel("Real")
    axes[0].grid(True, which="both", linestyle=":")

    axes[1].plot(indices, iq_imag)
    axes[1].set_ylabel("Imag")
    axes[1].grid(True, which="both", linestyle=":")

    axes[2].plot(indices, amplitudes)
    axes[2].set_ylabel("|x[n]|")
    axes[2].grid(True, which="both", linestyle=":")

    axes[3].plot(indices, phases)
    axes[3].set_ylabel("Phase [rad]")
    axes[3].set_xlabel("Sample Index")
    axes[3].grid(True, which="both", linestyle=":")

    fig.suptitle("Waveform Real/Imag/Amplitude/Phase")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_waveform(samples: np.ndarray, path: Path) -> Path:
    """Persist complex samples (complex64) to a .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, samples.astype(np.complex64))
    return path


def export_autocorrelation_plot(
    samples: np.ndarray,
    output_path: Path,
    max_lag: Optional[int] = None,
) -> Path:
    """Save an autocorrelation magnitude plot for the provided samples."""
    lags, autocorr = compute_autocorrelation(samples, max_lag)
    peak = np.max(np.abs(autocorr))
    norm_mag = np.abs(autocorr) / peak if peak > 0 else np.zeros_like(autocorr)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    markerline, stemlines, baseline = ax.stem(lags, norm_mag)
    markerline.set_markerfacecolor("tab:blue")
    markerline.set_markeredgecolor("tab:blue")
    plt.setp(stemlines, "color", "tab:blue")
    if baseline is not None:
        baseline.set_color("black")
        baseline.set_linewidth(0.5)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Normalized |Rxx|")
    ax.set_title("Autocorrelation Magnitude")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_ylim(0, 1.05)

    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate monotone or Zadoff-Chu waveforms for USRP transmission."
    )
    parser.add_argument(
        "--waveform",
        choices=("nr_prach_long", "zadoff_chu", "monotone"),
        default=WaveformSettings.waveform,
        help="Waveform type to generate.",
    )
    parser.add_argument(
        "--packet-len",
        type=int,
        default=WaveformSettings.packet_len,
        help="Samples per packet before optional guard padding.",
    )
    parser.add_argument(
        "--num-packets",
        type=int,
        default=WaveformSettings.num_packets,
        help="Number of packets to tile in the output waveform.",
    )
    parser.add_argument(
        "--guard-len",
        type=int,
        default=WaveformSettings.guard_len,
        help="Zero-padding samples after each packet.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=WaveformSettings.amplitude,
        help="Amplitude applied to the generated samples.",
    )
    parser.add_argument(
        "--tone-freq",
        type=float,
        default=WaveformSettings.tone_freq,
        help="Tone frequency in Hz (monotone waveform).",
    )
    parser.add_argument(
        "--samp-rate",
        type=float,
        default=WaveformSettings.samp_rate,
        help="Sample rate in Hz used for the monotone waveform.",
    )
    parser.add_argument(
        "--zc-root",
        type=int,
        default=WaveformSettings.zc_root,
        help="Zadoff-Chu root index (coprime with packet length).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the complex waveform as .npy.",
    )
    parser.add_argument(
        "--nr-prach-m-count",
        type=int,
        default=WaveformSettings.nr_prach_m_count,
        help="Number of cyclic shifts (M) for NR long PRACH (default: 64).",
    )
    parser.add_argument(
        "--nr-prach-m-index",
        type=int,
        default=WaveformSettings.nr_prach_m_index,
        help="Selected PRACH preamble index (m) within the M cyclic shifts.",
    )
    parser.add_argument(
        "--nr-prach-cp-len",
        type=int,
        default=WaveformSettings.nr_prach_cp_len,
        help="Cyclic prefix length for NR long PRACH (samples).",
    )
    parser.add_argument(
        "--nr-prach-ncs",
        type=int,
        default=None,
        help="Override cyclic shift parameter N_CS (samples).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Path to save an amplitude/phase plot for verification.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=None,
        help="Limit plot to the first N samples.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=None,
        help="Print the first N samples (default: disable).",
    )
    parser.add_argument(
        "--autocorr",
        action="store_true",
        help="Print autocorrelation of the generated waveform.",
    )
    parser.add_argument(
        "--autocorr-max-lag",
        type=int,
        default=None,
        help="Limit autocorrelation output to +/- N lags.",
    )
    parser.add_argument(
        "--autocorr-plot",
        type=Path,
        help="Path to save an autocorrelation magnitude plot.",
    )
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> WaveformSettings:
    return WaveformSettings(
        waveform=args.waveform,
        amplitude=args.amplitude,
        tone_freq=args.tone_freq,
        samp_rate=args.samp_rate,
        packet_len=args.packet_len,
        num_packets=args.num_packets,
        guard_len=args.guard_len,
        zc_root=args.zc_root,
        nr_prach_m_count=args.nr_prach_m_count,
        nr_prach_m_index=args.nr_prach_m_index,
        nr_prach_cp_len=args.nr_prach_cp_len,
        nr_prach_ncs=args.nr_prach_ncs,
    )


def generate_waveform_and_plot(
    settings: WaveformSettings,
    output_waveform: Path,
    plot_path: Path,
    plot_sample_count: Optional[int] = None,
) -> tuple[Path, Path]:
    """
    Build a waveform based on settings, save the samples, and export a
    verification plot in one call.
    """
    waveform = settings.build()
    save_waveform(waveform, output_waveform)
    export_waveform_plot(waveform, plot_path, plot_sample_count)
    return output_waveform, plot_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.plot_samples is not None and args.plot_samples <= 0:
        raise ValueError("plot-samples must be positive when provided.")
    if args.print_samples is not None and args.print_samples <= 0:
        raise ValueError("print-samples must be positive when provided.")
    if args.autocorr_max_lag is not None and args.autocorr_max_lag < 0:
        raise ValueError("autocorr-max-lag must be non-negative when provided.")

    settings = settings_from_args(args)
    waveform = settings.build()

    waveform_path = args.output or Path("outputs/waveform_preview.npy")
    plot_path = args.plot or waveform_path.with_suffix(".png")

    print(f"Generated waveform '{settings.waveform}' with {waveform.size} samples.")

    if args.print_samples is not None:
        print_samples(waveform, args.print_samples)

    if args.autocorr:
        print("Autocorrelation (time-shifted lags):")
        print_autocorrelation(waveform, args.autocorr_max_lag)

    save_waveform(waveform, waveform_path)
    location_note = "" if args.output else " (default path)"
    print(f"Waveform saved to {waveform_path}{location_note}.")

    export_waveform_plot(waveform, plot_path, args.plot_samples)
    plot_note = "" if args.plot else " (default path)"
    print(f"Verification plot saved to {plot_path}{plot_note}.")

    autocorr_plot_path: Optional[Path]
    if args.autocorr_plot is not None:
        autocorr_plot_path = args.autocorr_plot
    elif args.autocorr or args.autocorr_max_lag is not None:
        autocorr_plot_path = plot_path.with_name(plot_path.stem + "_autocorr.png")
    else:
        autocorr_plot_path = None

    if autocorr_plot_path is not None:
        export_autocorrelation_plot(waveform, autocorr_plot_path, args.autocorr_max_lag)
        extra_note = "" if args.autocorr_plot else " (default path)"
        print(f"Autocorrelation plot saved to {autocorr_plot_path}{extra_note}.")


__all__ = [
    "WaveformSettings",
    "print_samples",
    "print_autocorrelation",
    "compute_autocorrelation",
    "export_waveform_plot",
    "save_waveform",
    "export_autocorrelation_plot",
    "generate_waveform_and_plot",
    "parse_args",
    "settings_from_args",
]


if __name__ == "__main__":
    main()
