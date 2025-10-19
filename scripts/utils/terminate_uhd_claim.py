#!/usr/bin/env python3
"""
Utility for forcefully terminating local processes that hold UHD device claims.
"""
from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PROC_ROOT = Path("/proc")


@dataclass
class TerminatorConfig:
    """
    Configuration parameters for the UHD claim terminator.

    keyword_matches:
        Additional substrings that will trigger a match when they are present
        in the process command line. These complement the libuhd detection.
    grace_period:
        Time in seconds to wait between SIGTERM and SIGKILL to allow graceful shutdown.
    verbose:
        Emit status messages describing which processes matched and how termination went.
    """

    keyword_matches: tuple[str, ...] = ("uhd", "usrp", "tx_flowgraph", "rx_flowgraph")
    grace_period: float = 2.0
    verbose: bool = True


@dataclass
class ProcessInfo:
    """Lightweight snapshot of a process we may need to terminate."""

    pid: int
    cmdline: List[str]

    def display_name(self) -> str:
        if not self.cmdline:
            return f"PID {self.pid}"
        return f"PID {self.pid} ({' '.join(self.cmdline)})"


class UhdClaimTerminator:
    """Detect processes that load libuhd or match configured keywords and stop them."""

    def __init__(self, config: TerminatorConfig | None = None):
        self.config = config or TerminatorConfig()
        self._self_pid = os.getpid()

    def run(self) -> None:
        """Locate offending processes and terminate them."""
        matches = list(self._find_candidates())
        if not matches and self.config.verbose:
            print("No processes using UHD detected.")
            return

        for proc in matches:
            self._terminate_process(proc)

    def _find_candidates(self) -> Iterable[ProcessInfo]:
        """Scan /proc for processes loading libuhd or matching keywords."""
        for entry in PROC_ROOT.iterdir():
            if not entry.name.isdigit():
                continue

            pid = int(entry.name)
            if pid == self._self_pid:
                continue

            cmdline = self._read_cmdline(pid)
            if not cmdline and not self._maps_contains_libuhd(pid):
                continue

            if self._maps_contains_libuhd(pid):
                yield ProcessInfo(pid, cmdline)
                continue

            if self._cmdline_matches(cmdline):
                yield ProcessInfo(pid, cmdline)

    def _read_cmdline(self, pid: int) -> List[str]:
        cmd_file = PROC_ROOT / str(pid) / "cmdline"
        try:
            data = cmd_file.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return []

        if not data:
            return []

        entries = data.split(b"\0")
        return [part.decode(errors="ignore") for part in entries if part]

    def _maps_contains_libuhd(self, pid: int) -> bool:
        maps_file = PROC_ROOT / str(pid) / "maps"
        try:
            for line in maps_file.read_text(errors="ignore").splitlines():
                if "libuhd" in line:
                    return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return False
        return False

    def _cmdline_matches(self, cmdline: List[str]) -> bool:
        if not cmdline:
            return False
        joined = " ".join(cmdline).lower()
        return any(term.lower() in joined for term in self.config.keyword_matches)

    def _terminate_process(self, proc: ProcessInfo) -> None:
        if self.config.verbose:
            print(f"Attempting to stop {proc.display_name()}")

        if not self._send_signal(proc.pid, signal.SIGTERM):
            return

        time.sleep(self.config.grace_period)

        if self._is_running(proc.pid):
            if self.config.verbose:
                print(f"{proc.display_name()} ignored SIGTERM; sending SIGKILL.")
            self._send_signal(proc.pid, signal.SIGKILL)
        elif self.config.verbose:
            print(f"{proc.display_name()} stopped cleanly.")

    def _send_signal(self, pid: int, sig: signal.Signals) -> bool:
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            if self.config.verbose:
                print(f"Process {pid} is no longer running.")
            return False
        except PermissionError:
            if self.config.verbose:
                print(f"Permission denied trying to signal PID {pid}.")
            return False

    def _is_running(self, pid: int) -> bool:
        return (PROC_ROOT / str(pid)).exists()


def main() -> None:
    terminator = UhdClaimTerminator()
    terminator.run()


if __name__ == "__main__":
    main()
