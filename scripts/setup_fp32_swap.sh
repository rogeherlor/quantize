#!/usr/bin/env bash
# setup_fp32_swap.sh — Create an NVMe-backed swapfile to support FP32 VGGT inference.
#
# Why NVMe swap instead of zram:
#   The 6x zram devices (/dev/zram0..5) are RAM-backed compressed swap — they
#   compress pages but still consume the same 7.8 GB physical pool. Only disk-
#   backed swap actually extends addressable memory beyond physical RAM.
#
# CUDA allocations on Tegra are pinned and cannot be paged out. Swap helps by
# evicting CPU-side pages (Python heap, page cache) to make room for the CUDA
# allocator to complete the FP32 weight burst (~4.8 GB).
#
# REQUIRES: sudo (or run as root inside --privileged Docker container)
# TEARDOWN: sudo swapoff /ssd/swapfile_fp32 && sudo rm /ssd/swapfile_fp32

set -euo pipefail

SWAPFILE="/ssd/swapfile_fp32"
SIZE_GB=8
SWAPPINESS=100   # max — must be aggressive for FP32; default 60 is too conservative

echo "=== FP32 Swap Setup ==="
echo "Target: ${SWAPFILE} (${SIZE_GB} GB on NVMe)"

if swapon --show | grep -q "${SWAPFILE}"; then
    echo "Swapfile ${SWAPFILE} is already active. Nothing to do."
    swapon --show
    exit 0
fi

if [ ! -f "${SWAPFILE}" ]; then
    echo "Allocating ${SIZE_GB}G swapfile at ${SWAPFILE} ..."
    fallocate -l "${SIZE_GB}G" "${SWAPFILE}"
else
    echo "File ${SWAPFILE} already exists ($(du -sh "${SWAPFILE}" | cut -f1)). Skipping allocation."
fi

chmod 600 "${SWAPFILE}"
mkswap "${SWAPFILE}"
# Priority -10: lower than zram (priority 5) so zram fills first for normal CPU pressure.
# NVMe swap only activates when zram is exhausted — which is what we need for the
# FP32 CUDA allocation burst.
swapon --priority -10 "${SWAPFILE}"
sysctl -w vm.swappiness=${SWAPPINESS}

# Flush page cache so CUDA gets the ~420 MB gap it needs for FP32 (4.8 GB weights,
# ~4.4 GB typically free at idle — page cache fills the difference).
sync
echo 3 > /proc/sys/vm/drop_caches
echo "Page cache dropped."

echo ""
echo "=== Active swap devices ==="
swapon --show
echo ""
echo "=== Memory after setup ==="
free -h
echo ""
echo "Setup complete."
echo "NOTE: not added to /etc/fstab — re-run after each reboot."
echo "Teardown: sudo swapoff ${SWAPFILE} && sudo rm ${SWAPFILE}"
