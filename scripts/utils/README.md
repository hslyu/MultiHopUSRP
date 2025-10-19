# Utility Scripts

This directory collects small helpers that support the main flowgraphs.

- **terminate_uhd_claim.py**
  - Sends an RPC request to N3xx/N2xx devices to release a stale `claimed` handle.
  - Useful when UHD prints `Someone tried to claim this device again`.
  - Example: `python scripts/utils/terminate_uhd_claim.py --addr 192.168.10.1`.

- **packet_monitor.py**
  - Periodically prints NIC statistics (packets, drops, errors) for sanity checks during experiments.
  - Run `python scripts/utils/packet_monitor.py --interface ens4f0np0 --interval 1.0` to watch a port once per second.

Add new utilities here and update this README with a short description so others can discover them quickly.
