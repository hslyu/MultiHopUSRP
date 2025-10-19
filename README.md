# MultiHopUSRP Reference Guide

This document walks through the core scripts that ship with the repository so you can generate PRACH waveforms, transmit and receive them with USRPs, capture the associated VITA‑49 traffic, and analyse over‑the‑air delay end to end.

---

## 1. Environment

```bash
conda env create -f env/environment.yml
conda activate usrp
```

All USRP addresses, serial numbers, and clock settings are supplied at runtime through the CLI flags shown below. Adjust them to match your hardware.

---

## 2. Waveform Generator (`scripts/waveform_generator.py`)

The defaults produce a 5G NR long PRACH preamble (N<sub>ZC</sub>=839, M=64, CP=3168). Example:

```bash
python scripts/waveform_generator.py \
  --waveform nr_prach_long \
  --packet-len 839 \
  --zc-root 1 \
  --output outputs/prach_long.npy \
  --plot outputs/prach_long.png
```

- Override PRACH parameters with `--nr-prach-m-count`, `--nr-prach-m-index`, `--nr-prach-cp-len`, `--nr-prach-ncs`.
- The preview plot shows the first 1000 samples across four panels: real, imaginary, magnitude, and unwrapped phase.
- You can switch to other modes (`zadoff_chu`, `monotone`) via `--waveform`.

---

## 3. Transmit Flowgraph (`scripts/tx_flowgraph.py`)

```bash
python scripts/tx_flowgraph.py \
  --waveform-file outputs/prach_long.npy \
  --addr addr=192.168.10.1 \
  --mgmt-addr mgmt_addr=192.168.100.8 \
  --freq 2.45e9 \
  --samp-rate 1e6 \
  --gain 10
```

- `--waveform-file` must point to a complex64 `.npy` file.
- Add `--repeat` to loop indefinitely.
- Additional UHD controls (`--tx-antenna`, `--clock-source`, `--time-source`, `--lo-offset`, …) are available through `--help`.
- On shutdown the script prints NIC TX statistics so you can confirm packets were emitted.

---

## 4. Receive Flowgraph (`scripts/rx_flowgraph.py`)

```bash
python scripts/rx_flowgraph.py \
  --output captures/prach_rx.c32 \
  --addr addr=192.168.10.2 \
  --mgmt-addr mgmt_addr=192.168.100.7 \
  --freq 2.45e9 \
  --samp-rate 1e6 \
  --gain 20 \
  --duration 10
```

- Output is written as complex32 raw samples (`.c32`).
- Use `--duration 0` for continuous capture (Ctrl‑C to stop) and `--sync-start` to reset device time to zero before recording.
- Directories are created automatically if they do not exist.

---

## 5. Delay Analysis (`scripts/analyze_delay.py`)

```bash
python scripts/analyze_delay.py \
  --tx-waveform outputs/prach_long.npy \
  --rx-capture captures/prach_rx.c32 \
  --samp-rate 1e6 \
  --plot outputs/corr.png \
  --waveform-plot outputs/rx_wave.png \
  --print-rx-samples 8
```

- Uses FFT‑based cross‑correlation to compute `sample_delay`, `time_delay`, and a normalised peak metric.
- `--plot` stores the correlation magnitude, while `--waveform-plot` visualises the received samples (first 1000 points).
- Optional switches (`--allow-negative`, `--max-lag`) confine the search window.

---

## 6. Packet Capture Pipeline (`scripts/run_capture_pipeline.py`)

Wraps `dumpcap` around the flowgraphs so that VITA‑49 UDP packets are captured while IQ samples stream. Grant `dumpcap` the required capability beforehand (e.g. `sudo setcap cap_net_raw,cap_net_admin=eip $(which dumpcap)`), or run with `--use-sudo` if your policy permits.

### TX capture

```bash
python scripts/run_capture_pipeline.py tx \
  --interface ens4f0np0 \
  --pcap outputs/tx_capture.pcap \
  --udp-port 49152 \
  --timestamp-mode adapter_unsynced \
  --tx-args --waveform-file outputs/prach_long.npy --gain 10 --freq 2.45e9
```

### RX capture

```bash
python scripts/run_capture_pipeline.py rx \
  --interface ens4f1np1 \
  --pcap outputs/rx_capture.pcap \
  --udp-port 49153 \
  --timestamp-mode adapter_unsynced \
  --rx-args --duration 5 --output captures/prach_rx.c32
```

- `--timestamp-mode` should match one of the values listed by `dumpcap --list-time-stamp-types` (for ConnectX adapters, `adapter_unsynced` reports PHC timestamps).
- Additional knobs: `--extra-dumpcap-flags`, `--ring-buffer-mb`, `--use-sudo`.
- Flowgraphs are spawned as subprocesses; once they exit, `dumpcap` is interrupted and the PCAP is closed cleanly.

---

## 7. Post-processing VITA-49 traffic

Captured PCAP files contain UHD VITA-49 headers. Parse them with Scapy, Pyshark, or a custom Python script to extract sequence numbers and timestamps. Combined with the `sample_delay` reported by `analyze_delay.py`, you can map a specific sample index to a concrete packet and hardware timestamp.

For convenience, `scripts/align_delay.py` performs this alignment end-to-end:

```bash
python scripts/align_delay.py \
  --tx-pcap outputs/tx_capture.pcap \
  --rx-pcap outputs/rx_capture.pcap \
  --tx-waveform outputs/prach_long.npy \
  --rx-capture captures/prach_rx.c32 \
  --samp-rate 1e6 \
  --sample-bytes 8 \
  --channels 1 \
  --json
```

The script expects `scapy` to be available (`pip install scapy`). Output includes sample delay, hardware timestamps, and the inferred over-the-air latency.

---

## 8. Utilities (`scripts/utils`)

Key helpers include:

- `terminate_uhd_claim.py` — releases N3xx/N2xx devices that were left in a `claimed` state.
- `packet_monitor.py` — quick NIC telemetry watcher (packet counters, drops, etc.).

See `scripts/utils/README.md` for details and extend the directory with additional tooling as needed.

---

## 9. Recommended workflow

1. Generate the desired waveform with `waveform_generator.py`.
2. Run TX and RX flowgraphs to verify RF connectivity.
3. Use `run_capture_pipeline.py` to capture VITA‑49 packets while streaming IQ.
4. Post‑process with `analyze_delay.py` and, if required, parse PCAP timestamps.
5. Ensure no other process retains the USRPs (`terminate_uhd_claim.py`) and keep PHC/PTP synchronisation (`ptp4l`, `phc2sys`) active for reliable timestamping.

Customize IP addresses, ports, gains, and clock sources to reflect your deployment. Each script supports `--help` for the complete argument list.
