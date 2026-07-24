#!/usr/bin/env bash
# Fix NVIDIA "Driver/library version mismatch" on this box.
# Run with: sudo bash fix-nvidia.sh
set -euo pipefail

echo "== Before =="
echo -n "Loaded kernel module: "; cat /proc/driver/nvidia/version 2>/dev/null | head -1 || echo "(none)"
echo -n "Userspace NVML:       "; (nvidia-smi 2>&1 | head -1) || true
echo

echo "== Aligning kernel module to userspace (535.309.01) =="
apt-get update
# Reinstall the driver + the HWE kernel-module meta so the matching module
# gets (re)built/installed for the latest supported kernel.
apt-get install --reinstall -y \
  nvidia-driver-535 \
  linux-modules-nvidia-535-generic-hwe-24.04

echo
echo "== After (on-disk module versions) =="
for k in /lib/modules/*/kernel/nvidia-535/nvidia.ko; do
  printf '%s => %s\n' "$k" "$(modinfo -F version "$k" 2>/dev/null || echo '?')"
done

echo
echo "Done installing. A REBOOT is required to load the matching module:"
echo "    sudo reboot"
echo
echo "After reboot, verify:"
echo "    nvidia-smi"
echo "    cd ~/repo/z-image && .venv/bin/python -c 'import torch; print(torch.cuda.is_available())'"
