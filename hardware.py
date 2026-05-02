"""Hardware detection: CPU, RAM, and GPU(s) across vendors.

Supports NVIDIA (nvidia-smi), AMD (rocm-smi), Intel (lspci/WMI),
and Apple Silicon (sysctl). Multi-GPU aware. No vendor-specific
Python libs required - everything goes through subprocess.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import psutil


@dataclass
class GPUInfo:
    index: int
    name: str
    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "unknown"
    total_vram_mb: int
    free_vram_mb: int

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_mb / 1024

    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_mb / 1024


@dataclass
class SystemInfo:
    os_name: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float
    free_ram_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    # GPUs that were detected but considered too small/auxiliary to use for
    # inference (typically integrated GPUs alongside a discrete card). Kept
    # for transparency in the menu header.
    ignored_gpus: List[GPUInfo] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.total_vram_gb for g in self.gpus)

    @property
    def free_vram_gb(self) -> float:
        return sum(g.free_vram_gb for g in self.gpus)

    @property
    def primary_vendor(self) -> str:
        if not self.gpus:
            return "cpu"
        return max(self.gpus, key=lambda g: g.total_vram_mb).vendor

    @property
    def is_multi_gpu(self) -> bool:
        return len(self.gpus) > 1


# ---------------------------------------------------------------------------
# Helpers

def _run(cmd: List[str], timeout: float = 5) -> Optional[str]:
    """Run a command and return stdout, or None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="ignore",
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


# ---------------------------------------------------------------------------
# GPU detection per vendor

def _detect_nvidia() -> List[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus: List[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vendor="nvidia",
                    total_vram_mb=int(parts[2]),
                    free_vram_mb=int(parts[3]),
                ))
            except ValueError:
                continue
    return gpus


def _detect_amd_rocm() -> List[GPUInfo]:
    if not shutil.which("rocm-smi"):
        return []

    # Try JSON first - more reliable across rocm-smi versions
    out = _run(["rocm-smi", "--showmeminfo", "vram",
                "--showproductname", "--json"])
    if out:
        try:
            data = json.loads(out)
            gpus: List[GPUInfo] = []
            for key, info in data.items():
                m = re.match(r"card(\d+)", key, re.IGNORECASE)
                if not m:
                    continue
                idx = int(m.group(1))
                total_b = 0
                used_b = 0
                name = (info.get("Card Series")
                        or info.get("Card model")
                        or info.get("Card SKU")
                        or f"AMD GPU {idx}")
                for k, v in info.items():
                    if "VRAM Total Memory" in k or k == "Total Memory (B)":
                        try:
                            total_b = int(str(v).strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif "VRAM Total Used" in k or k == "Used Memory (B)":
                        try:
                            used_b = int(str(v).strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                total_mb = total_b // (1024 * 1024)
                used_mb = used_b // (1024 * 1024)
                gpus.append(GPUInfo(
                    index=idx,
                    name=name,
                    vendor="amd",
                    total_vram_mb=total_mb,
                    free_vram_mb=max(0, total_mb - used_mb),
                ))
            if gpus:
                return gpus
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Text fallback
    out = _run(["rocm-smi", "--showmeminfo", "vram"])
    if not out:
        return []
    by_idx: dict = {}
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d+)\].*?Total\s+Memory.*?(\d+)\s*$", line,
                     re.IGNORECASE)
        if m:
            by_idx.setdefault(int(m.group(1)), {})["total"] = int(m.group(2))
    return [
        GPUInfo(
            index=i,
            name=f"AMD GPU {i}",
            vendor="amd",
            total_vram_mb=info.get("total", 0) // (1024 * 1024),
            free_vram_mb=info.get("total", 0) // (1024 * 1024),
        )
        for i, info in by_idx.items()
    ]


def _vendor_from_name(name: str) -> str:
    """Best-effort vendor inference from a GPU's display name."""
    n = name.lower()
    if any(s in n for s in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
        return "nvidia"
    if any(s in n for s in ("amd", "radeon", "rx ", "rdna")):
        return "amd"
    if "intel" in n or "arc" in n:
        return "intel"
    return "unknown"


# PowerShell snippet: enumerates every PCI video adapter, reads VRAM from the
# 64-bit registry value (HardwareInformation.qwMemorySize) so 16 GB+ cards
# are reported correctly. Win32_VideoController.AdapterRAM is signed 32-bit
# and overflows at 4 GB, so it's only used as a last-resort fallback.
_WIN_GPU_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$adapters = Get-CimInstance Win32_VideoController |
    Where-Object {
        $_.PNPDeviceID -like 'PCI*' -and
        $_.Name -notmatch 'Basic Render|Remote Display|Hyper-V|RDP|Mirror'
    }
$regBase = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$regKeys = Get-ChildItem $regBase -ErrorAction SilentlyContinue
$results = @()
foreach ($a in $adapters) {
    $vram = [int64]0
    foreach ($k in $regKeys) {
        $p = Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue
        if ($null -ne $p -and $p.DriverDesc -eq $a.Name) {
            $qw = $p.'HardwareInformation.qwMemorySize'
            if ($null -ne $qw) { $vram = [int64]$qw }
            break
        }
    }
    if ($vram -le 0 -and $a.AdapterRAM -gt 0) {
        $vram = [int64]$a.AdapterRAM
    }
    $results += [PSCustomObject]@{
        Name = $a.Name
        VRAM = $vram
        PNP  = $a.PNPDeviceID
    }
}
$results | ConvertTo-Json -Compress -Depth 3
"""


def _detect_windows_gpus(skip_names: Optional[set] = None) -> List[GPUInfo]:
    """Enumerate every PCI video adapter on Windows via WMI + registry.

    Key fix vs the old code: reads `HardwareInformation.qwMemorySize` from the
    registry, which is 64-bit. WMI's `AdapterRAM` is signed 32-bit and wraps
    on cards with > 4 GB of VRAM, so a 16 GB Radeon shows as 0 or garbage.

    `skip_names` is for de-duplicating against vendor-specific detectors
    (e.g. an RTX card already found via nvidia-smi shouldn't be re-added).
    """
    if platform.system() != "Windows":
        return []
    skip = {n.lower() for n in (skip_names or set())}

    out = _run([
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-Command", _WIN_GPU_PS,
    ], timeout=12)
    if not out:
        return []

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]

    gpus: List[GPUInfo] = []
    for d in data:
        name = (d.get("Name") or "").strip()
        if not name or name.lower() in skip:
            continue
        try:
            vram = int(d.get("VRAM") or 0)
        except (TypeError, ValueError):
            vram = 0
        if vram < 0:  # paranoia: 32-bit overflow
            vram = 0
        total_mb = vram // (1024 * 1024)
        # No reliable cross-vendor "free VRAM" on Windows without driver SDKs.
        # Assume mostly free; user can override via --ctx if needed.
        free_mb = int(total_mb * 0.95)
        gpus.append(GPUInfo(
            index=len(gpus),
            name=name,
            vendor=_vendor_from_name(name),
            total_vram_mb=total_mb,
            free_vram_mb=free_mb,
        ))
    return gpus


def _detect_linux_other_gpus(skip_names: Optional[set] = None) -> List[GPUInfo]:
    """Linux: catch GPUs that nvidia-smi/rocm-smi missed (mainly Intel iGPUs).

    Uses lspci for naming. VRAM is unknown without vendor SDKs, so it stays 0
    and these GPUs end up filtered out when a real dGPU is also present.
    """
    if platform.system() != "Linux":
        return []
    skip = {n.lower() for n in (skip_names or set())}
    out = _run(["lspci", "-mm"])
    if not out:
        return []
    gpus: List[GPUInfo] = []
    for line in out.splitlines():
        if not re.search(r'"(VGA|3D|Display)', line):
            continue
        parts = [p.strip('"') for p in re.findall(r'"[^"]*"', line)]
        if len(parts) < 4:
            continue
        vendor_str = parts[1]
        name = parts[2]
        if name.lower() in skip:
            continue
        gpus.append(GPUInfo(
            index=len(gpus),
            name=f"{vendor_str} {name}".strip(),
            vendor=_vendor_from_name(f"{vendor_str} {name}"),
            total_vram_mb=0,
            free_vram_mb=0,
        ))
    return gpus


def _detect_apple() -> List[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    # Apple Silicon = unified memory; treat the whole RAM as addressable VRAM
    out = _run(["sysctl", "-n", "hw.memsize"])
    if not out:
        return []
    try:
        mem_b = int(out.strip())
    except ValueError:
        return []
    mem_mb = mem_b // (1024 * 1024)
    name_out = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
    label = f"Apple Silicon ({name_out.strip()})" if name_out else "Apple Silicon"
    return [GPUInfo(
        index=0,
        name=label,
        vendor="apple",
        total_vram_mb=mem_mb,
        free_vram_mb=mem_mb,
    )]


def _detect_cpu_name() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    elif platform.system() == "Windows":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor).Name",
        ])
        if out:
            return out.strip()
    return platform.processor() or "Unknown CPU"


# ---------------------------------------------------------------------------
# Public API

def _filter_inference_gpus(
    gpus: List[GPUInfo],
) -> Tuple[List[GPUInfo], List[GPUInfo]]:
    """Split detected GPUs into (used for inference, ignored).

    Heuristic: when one GPU is clearly dominant (>= 2x the VRAM of the next),
    keep only it and mark the smaller ones as ignored. This is what catches
    the "iGPU + dGPU" case — without it, llama.cpp's tensor-split would
    bottleneck the dGPU on the iGPU's tiny VRAM and shared bandwidth.

    Also drops GPUs with 0 reported VRAM when at least one GPU has measured
    VRAM (those are usually iGPUs whose memory we couldn't read).
    """
    if len(gpus) < 2:
        return gpus, []

    measured = [g for g in gpus if g.total_vram_mb > 0]
    unmeasured = [g for g in gpus if g.total_vram_mb <= 0]

    # If we have at least one measured GPU, drop the unmeasured ones —
    # almost always iGPUs without registry VRAM info.
    if measured and unmeasured:
        kept_pool = measured
        ignored = list(unmeasured)
    else:
        kept_pool = list(gpus)
        ignored = []

    if len(kept_pool) < 2:
        return kept_pool, ignored

    sorted_g = sorted(kept_pool, key=lambda g: g.total_vram_mb, reverse=True)
    largest = sorted_g[0]
    used: List[GPUInfo] = [largest]
    for g in sorted_g[1:]:
        # Keep as a peer if it's at least half the largest's VRAM
        if g.total_vram_mb * 2 >= largest.total_vram_mb:
            used.append(g)
        else:
            ignored.append(g)
    return used, ignored


def detect_system() -> SystemInfo:
    """Detect everything in one call. Best-effort; never raises."""
    vm = psutil.virtual_memory()

    raw: List[GPUInfo] = []
    raw.extend(_detect_nvidia())
    raw.extend(_detect_amd_rocm())
    raw.extend(_detect_apple())

    # OS-specific catch-all detectors fill in whatever the vendor-specific
    # ones missed (Windows: AMD without ROCm, Intel Arc; Linux: Intel iGPUs).
    found_names = {g.name.lower() for g in raw}
    raw.extend(_detect_windows_gpus(skip_names=found_names))
    found_names = {g.name.lower() for g in raw}
    raw.extend(_detect_linux_other_gpus(skip_names=found_names))

    # Re-index in detection order for stable display
    for i, g in enumerate(raw):
        g.index = i

    used, ignored = _filter_inference_gpus(raw)

    return SystemInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_name=_detect_cpu_name(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        total_ram_gb=vm.total / (1024 ** 3),
        free_ram_gb=vm.available / (1024 ** 3),
        gpus=used,
        ignored_gpus=ignored,
    )


def format_system(info: SystemInfo) -> str:
    """Human-readable summary, used for the menu header."""
    lines = [
        f"OS:   {info.os_name}",
        f"CPU:  {info.cpu_name} "
        f"({info.cpu_cores_physical}C/{info.cpu_cores_logical}T)",
        f"RAM:  {info.total_ram_gb:.1f} GB total, "
        f"{info.free_ram_gb:.1f} GB free",
    ]
    if info.gpus:
        for g in info.gpus:
            tag = f"[{g.vendor}]"
            if g.total_vram_mb > 0:
                lines.append(
                    f"GPU{g.index}: {tag} {g.name} "
                    f"({g.total_vram_gb:.1f} GB total, "
                    f"{g.free_vram_gb:.1f} GB free)"
                )
            else:
                lines.append(f"GPU{g.index}: {tag} {g.name} (VRAM unknown)")
    else:
        lines.append("GPU:  none detected (CPU-only inference)")

    for g in info.ignored_gpus:
        size = (f"{g.total_vram_gb:.1f} GB" if g.total_vram_mb > 0
                else "VRAM unknown")
        lines.append(
            f"      (ignored: [{g.vendor}] {g.name}, {size} — "
            f"too small or auxiliary)"
        )
    return "\n".join(lines)
