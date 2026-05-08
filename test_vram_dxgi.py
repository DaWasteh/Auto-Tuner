"""Test-Skript: VRAM-\u00dcberwachung \u00fcber Win32-Registry und psutil.

Verwendet pywin32 (winreg, win32com), PyQt6 und psutil f\u00fcr GPU-Erkennung
und System-Ressourcen (RAM, CPU). Funktioniert auf Windows 10/11.

Dieser Ansatz vermeidet komplexe ctypes lpVtbl-Manipulationen und nutzt
stattdessen die stabilen Win32-APIs und PyQt6 f\u00fcr eine m\u00f6gliche GUI.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import psutil

from dataclasses import dataclass, field
from typing import List, Optional

# PyQt6-Imports für die Live-Monitor-GUI
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Repr\u00e4sentation einer GPU mit VRAM-Informationen."""
    name: str
    vendor: str
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
    """Gesamte System-Informationen."""
    os_name: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float
    free_ram_gb: float
    ram_percent: float
    cpu_percent: float
    gpus: List[GPUInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU-Erkennung \u00fcber Windows Registry (64-bit VRAM)
# ---------------------------------------------------------------------------

def _get_vram_from_registry() -> dict[str, int]:
    """Ermittle GPU-Infos aus der Windows Registry.
    
    Liest HardwareInformation.qwMemorySize (64-bit) aus dem
    Registry-Key des GPU-Drivers. Vermeidet den 32-bit Overflow
    von Win32_VideoController.AdapterRAM.
    """
    import winreg

    gpus: dict[str, int] = {}
    reg_path_base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"

    try:
        base_key = winreg.HKEY_LOCAL_MACHINE
    except Exception:
        return gpus

    # Durchsuche Registry nach GPU-Keys
    for i in range(100):
        key_name = f"000{i}"
        try:
            key = winreg.OpenKey(base_key, reg_path_base + "\\" + key_name, 0, winreg.KEY_READ)
        except FileNotFoundError:
            break
        except OSError:
            continue

        driver_desc = ""
        vram_qw = 0
        try:
            driver_desc, _ = winreg.QueryValueEx(key, "DriverDesc")
            qw_mem, _ = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")
            vram_qw = int(qw_mem)
        except (FileNotFoundError, OSError, ValueError):
            pass
        finally:
            try:
                winreg.CloseKey(key)
            except Exception:
                pass

        if not driver_desc or vram_qw <= 0:
            continue

        # \u00dcberspringe integrierte/auxiliary GPUs
        desc_lower = driver_desc.lower()
        if any(skip in desc_lower for skip in ("basic render", "remote display", "hyper-v", "rdp", "microsoft")):
            continue

        gpus[driver_desc] = vram_qw

    return gpus


# ---------------------------------------------------------------------------
# GPU-Erkennung \u00fcber PowerShell (JSON-Ausgabe)
# ---------------------------------------------------------------------------

def _get_gpu_info_powershell() -> List[GPUInfo]:
    """Ermittle GPU-Infos via PowerShell (WMI + Registry)."""
    ps_script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$regBase = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$regKeys = Get-ChildItem $regBase -ErrorAction SilentlyContinue
$adapters = Get-CimInstance Win32_VideoController |
    Where-Object {
        $_.PNPDeviceID -like 'PCI*' -and
        $_.Name -notmatch 'Basic Render|Remote Display|Hyper-V|RDP|Mirror'
    }
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
        VendorId = $a.Manufacturer
    }
}
$results | ConvertTo-Json -Compress -Depth 3
"""

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True, text=True, timeout=15, errors="ignore"
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]

        gpus: List[GPUInfo] = []
        for d in data:
            name = (d.get("Name") or "").strip()
            if not name:
                continue
            try:
                vram = int(d.get("VRAM") or 0)
            except (TypeError, ValueError):
                vram = 0
            if vram < 0:
                vram = 0
            total_mb = vram // (1024 * 1024)

            vendor = "unknown"
            desc_lower = name.lower()
            if any(s in desc_lower for s in ("nvidia", "geforce", "rtx", "gtx")):
                vendor = "nvidia"
            elif any(s in desc_lower for s in ("amd", "radeon", "rx ")):
                vendor = "amd"
            elif any(s in desc_lower for s in ("intel", "iris", "hd graphics")):
                vendor = "intel"

            # free_vram_mb wird später durch echte Tools gesetzt (nvidia-smi / rocm-smi)
            # PowerShell kann keinen echten freien VRAM auslesen, daher 0 als Platzhalter
            free_mb = 0

            gpus.append(GPUInfo(
                name=name,
                vendor=vendor,
                total_vram_mb=total_mb,
                free_vram_mb=free_mb,
            ))
        return gpus
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
        return []


# ---------------------------------------------------------------------------
# NVIDIA-spezifisch (nvidia-smi)
# ---------------------------------------------------------------------------

def _get_nvidia_gpu_info() -> List[GPUInfo]:
    """Ermittle GPU-Infos via nvidia-smi (pr\u00e4zise free VRAM)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, errors="ignore"
        )
        if result.returncode != 0:
            return []

        gpus: List[GPUInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpus.append(GPUInfo(
                        name=parts[1],
                        vendor="nvidia",
                        total_vram_mb=int(parts[2]),
                        free_vram_mb=int(parts[3]),
                    ))
                except ValueError:
                    continue
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


# ---------------------------------------------------------------------------
# AMD-spezifisch (rocm-smi + Windows DXGI fallback)
# ---------------------------------------------------------------------------

def _get_amd_vram_dxgi() -> List[dict]:
    """Ermittle AMD GPU VRAM-Infos über DXGI/PowerShell unter Windows.
    
    Dies ist notwendig, da rocm-smi auf Windows oft nicht verfügbar ist
    oder bei neueren GPUs (RX 9000 series) falsche Werte liefert.
    """
    ps_script = r"""
    $ErrorActionPreference = 'SilentlyContinue'
    
    # Methode 1: Overcommit-Status über WMI (Windows 10/11)
    $results = @()
    
    # Versuche AMD-specific Performance Counters
    $counters = Get-Counter -ListSet amd*,*ad* -ErrorAction SilentlyContinue |
        Where-Object { $_.CounterSetName -like "*amd*" -or $_.CounterSetName -like "*gpu*" }
    
    # Methode 2: Direct3D adapter info via WMI CIM
    $adapters = Get-CimInstance Win32_VideoController |
        Where-Object {
            $_.PNPDeviceID -like 'PCI\VEN_100*' -or  # 100h = AMD
            ($_.Name -match 'Radeon|RX |AMD')
        }
    
    foreach ($adapter in $adapters) {
        $totalMb = 0
        $usedMb = 0
        
        # DedicatedVRAM = total dedicated video memory
        $totalMb = [int64]($adapter.DedicatedVideoMemory / 1MB)
        
        # AvailableVRAM = currently available dedicated video memory
        $availableMb = [int64]($adapter.AvailableVideoMemory / 1MB)
        
        # SystemMemory = shared system memory
        $systemTotalMb = [int64]($adapter.AdapterRAM / 1MB) -as [int64]
        if ($systemTotalMb -lt 0) { $systemTotalMb = 0 }
        $systemAvailableMb = [int64]($adapter.AdapterMemoryLoaded -as [int64])
        
        # Calculate: free = availableVRAM + portion of shared that can be used
        # But for accurate GPU-only VRAM, use Dedicated - Used
        if ($totalMb -gt 0) {
            $usedMb = $totalMb - $availableMb
            if ($usedMb -lt 0) { $usedMb = 0 }
        }
        
        $results += [PSCustomObject]@{
            Name = $adapter.Name
            TotalMB = $totalMb
            FreeMB = $availableMb
            UsedMB = $usedMb
            VendorId = ($adapter.PNPDeviceID -split 'VEN_')[1].Substring(0,4)
        }
    }
    
    # Methode 3: Fallback - DXGI via native API (nur wenn keine WMI-Ergebnisse)
    if ($results.Count -eq 0) {
        try {
            Add-Type -AssemblyName PresentationCore
            $dxgi = [Runtime.InteropServices.ComVisibilityAttribute]::new($true)
            
            # Simple PowerShell WMI fallback for AMD VRAM
            $regBase = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
            $regKeys = Get-ChildItem $regBase -ErrorAction SilentlyContinue
            
            $adapters2 = Get-CimInstance Win32_VideoController |
                Where-Object { $_.PNPDeviceID -like 'PCI\VEN_100*' }
            
            foreach ($a in $adapters2) {
                $vramTotal = 0
                foreach ($k in $regKeys) {
                    $p = Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue
                    if ($null -ne $p -and $p.DriverDesc -eq $a.Name) {
                        $qw = $p.'HardwareInformation.qwMemorySize'
                        if ($null -ne $qw) { $vramTotal = [int64]$qw }
                        break
                    }
                }
                
                # Get current usage via Get-CimInstance Win32_PerfFormattedData
                $perf = Get-CimInstance Win32_PerfFormattedData_AMD0C01_GPU -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -eq 'GPU' }
                
                if ($perf) {
                    $usedBytes = [int64]($perf.Memory % 1MB)
                    $totalBytes = [int64]$vramTotal * 1024 * 1024
                    $freeBytes = $totalBytes - $usedBytes
                    if ($freeBytes -lt 0) { $freeBytes = 0 }
                    
                    $results += [PSCustomObject]@{
                        Name = $a.Name
                        TotalMB = [int64]($vramTotal / (1024*1024))
                        FreeMB = [int64]($freeBytes / (1024*1024))
                        UsedMB = [int64]($usedBytes / (1024*1024))
                        VendorId = '100'
                    }
                } else {
                    # No perf counter - use registry total as both total and estimate free
                    $results += [PSCustomObject]@{
                        Name = $a.Name
                        TotalMB = [int64]($vramTotal / (1024*1024))
                        FreeMB = 0
                        UsedMB = 0
                        VendorId = '100'
                    }
                }
            }
        } catch {
            # Ignore errors
        }
    }
    
    $results | ConvertTo-Json -Compress -Depth 3
    """

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True, text=True, timeout=15, errors="ignore"
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]

        return [
            {
                "name": d.get("Name", "AMD GPU"),
                "total_mb": int(d.get("TotalMB") or 0),
                "free_mb": int(d.get("FreeMB") or 0),
                "used_mb": int(d.get("UsedMB") or 0),
            }
            for d in data if d.get("Name")
        ]
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
        return []


def _get_amd_gpu_info() -> List[GPUInfo]:
    """Ermittle GPU-Infos via rocm-smi oder Windows DXGI/WMI.
    
    Versucht zuerst rocm-smi, dann Windows-spezifische Methoden
    (WMI CIM, DXGI) für aktuelle AMD GPUs wie RX 9000 series.
    """
    # Versuch 1: rocm-smi (wenn installiert)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=10, errors="ignore"
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            gpus: List[GPUInfo] = []
            for key, info in data.items():
                total_b = 0
                used_b = 0
                name = info.get("Card Series") or info.get("Card model") or f"AMD GPU {key}"
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
                    name=name,
                    vendor="amd",
                    total_vram_mb=total_mb,
                    free_vram_mb=max(0, total_mb - used_mb),
                ))
            if gpus:
                return gpus
    except (json.JSONDecodeError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Versuch 2: Windows DXGI/WMI fallback
    dxgi_results = _get_amd_vram_dxgi()
    if dxgi_results:
        gpus = []
        for r in dxgi_results:
            total_mb = r["total_mb"]
            free_mb = r["free_mb"]
            # Wenn DXGI keinen freien VRAM liefern konnte, versuche es mit einer
            # besseren Schätzung basierend auf System-VRAM-Verfügbarkeit
            if free_mb == 0 and total_mb > 0:
                # Fallback: Registry-Wert als Total, aber 0 als Free (signalisiert "unbekannt")
                free_mb = 0
            gpus.append(GPUInfo(
                name=r["name"],
                vendor="amd",
                total_vram_mb=total_mb,
                free_vram_mb=free_mb,
            ))
        return gpus

    return []


# ---------------------------------------------------------------------------
# PyQt6-basierte Live-\u00dcberwachung (optional)
# ---------------------------------------------------------------------------

def _create_live_monitor() -> Optional[type]:
    """Erstelle PyQt6-Monitor-Klasse f\u00fcr Live-\u00dcberwachung.
    
    Gibt None zur\u00fcck, wenn PyQt6 nicht verf\u00fcgbar ist.
    """
    try:
        from PyQt6.QtWidgets import (
            QMainWindow, QWidget, QVBoxLayout,
            QHBoxLayout, QLabel, QProgressBar, QGroupBox,
        )
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QFont

        class SystemMonitorWindow(QMainWindow):
            """Hauptfenster f\u00fcr die Live-\u00dcberwachung."""

            def __init__(self) -> None:
                super().__init__()
                self.setWindowTitle("System Monitor (psutil + pywin32)")
                self.setMinimumSize(500, 400)

                self.cpu_label = QLabel()
                self.ram_label = QLabel()
                self.gpu_labels: list[QLabel] = []
                self.ram_progress = QProgressBar()
                self.gpu_progresses: list[QProgressBar] = []

                self._setup_ui()
                self.timer = QTimer(self)
                self.timer.timeout.connect(self._update)
                self.timer.start(1000)  # Update jede Sekunde

            def _setup_ui(self) -> None:
                central = QWidget()
                self.setCentralWidget(central)
                layout = QVBoxLayout(central)

                # CPU Group
                cpu_group = QGroupBox("CPU")
                cpu_layout = QVBoxLayout()
                self.cpu_label.setFont(QFont("Consolas", 10))
                cpu_layout.addWidget(self.cpu_label)
                cpu_group.setLayout(cpu_layout)
                layout.addWidget(cpu_group)

                # RAM Group
                ram_group = QGroupBox("RAM")
                ram_layout = QVBoxLayout()
                self.ram_label.setFont(QFont("Consolas", 10))
                ram_layout.addWidget(self.ram_label)
                self.ram_progress.setMaximumHeight(20)
                ram_layout.addWidget(self.ram_progress)
                ram_group.setLayout(ram_layout)
                layout.addWidget(ram_group)

                # GPU Group
                gpu_group = QGroupBox("GPU(s)")
                layout.addWidget(gpu_group)
                self.gpu_layout_inner = QVBoxLayout(gpu_group)
                gpu_group.setLayout(self.gpu_layout_inner)

            def _update(self) -> None:
                vm = psutil.virtual_memory()
                cpu_pct = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count(logical=False) or 1
                cpu_count_logical = psutil.cpu_count(logical=True) or 1

                self.cpu_label.setText(
                    f"Auslastung: {cpu_pct:.1f}%  |  "
                    f"{cpu_count}C/{cpu_count_logical}T"
                )

                self.ram_label.setText(
                    f"Gesamt: {vm.total / (1024**3):.1f} GB  |  "
                    f"Frei: {vm.available / (1024**3):.1f} GB  |  "
                    f"Nutzung: {vm.percent:.1f}%"
                )
                self.ram_progress.setValue(int(vm.percent))

                # GPU-Updates
                gpus = self._get_current_gpus()
                while len(self.gpu_labels) < len(gpus):
                    label = QLabel()
                    label.setFont(QFont("Consolas", 9))
                    progress = QProgressBar()
                    progress.setMaximumHeight(15)
                    self.gpu_layout_inner.addWidget(label)
                    self.gpu_layout_inner.addWidget(progress)
                    self.gpu_labels.append(label)
                    self.gpu_progresses.append(progress)

                for i, gpu in enumerate(gpus):
                    free_pct = (gpu.free_vram_mb / gpu.total_vram_mb * 100) if gpu.total_vram_mb > 0 else 0
                    self.gpu_labels[i].setText(
                        f"[{gpu.vendor.upper()}] {gpu.name}\n"
                        f"VRAM: {gpu.free_vram_gb:.2f}/{gpu.total_vram_gb:.2f} GB ({free_pct:.1f}% frei)"
                    )
                    self.gpu_progresses[i].setValue(int(free_pct))

                # \u00dcbersch\u00fcssige GPUs ausblenden
                for i in range(len(gpus), len(self.gpu_labels)):
                    self.gpu_labels[i].setVisible(False)
                    self.gpu_progresses[i].setVisible(False)

            def _get_current_gpus(self) -> List[GPUInfo]:
                gpus: List[GPUInfo] = []
                gpus.extend(_get_nvidia_gpu_info())
                gpus.extend(_get_amd_gpu_info())
                if not gpus:
                    gpus.extend(_get_gpu_info_powershell())
                return gpus

        return SystemMonitorWindow

    except ImportError:
        return None


def start_gui_monitor() -> None:
    """Starte die PyQt6-GUI-\u00dcberwachung."""
    window_class = _create_live_monitor()
    if window_class is None:
        print("PyQt6 ist nicht installiert. Bitte ausf\u00fchren:")
        print("  pip install PyQt6")
        return

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = window_class()
    window.show()
    sys.exit(app.exec())


# ---------------------------------------------------------------------------
# psutil-Integration f\u00fcr System-Ressourcen
# ---------------------------------------------------------------------------

def detect_system() -> SystemInfo:
    """Ermittle vollst\u00e4ndige System-Infos (GPU + RAM + CPU)."""
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_cores_physical = psutil.cpu_count(logical=False) or 1
    cpu_cores_logical = psutil.cpu_count(logical=True) or 1

    # CPU-Name ermitteln
    cpu_name = platform.processor() or "Unknown"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cpu_name = result.stdout.strip()
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                cpu_name = result.stdout.strip()
        except Exception:
            pass

    # GPU-Infos sammeln (priorisiere vendor-spezifische Tools)
    gpus: List[GPUInfo] = []
    gpus.extend(_get_nvidia_gpu_info())
    gpus.extend(_get_amd_gpu_info())

    if not gpus:
        # Fallback: Registry/PowerShell
        gpus.extend(_get_gpu_info_powershell())
    if not gpus:
        registry_vram = _get_vram_from_registry()
        for name, vram_bytes in registry_vram.items():
            total_mb = vram_bytes // (1024 * 1024)
            free_mb = int(total_mb * 0.95)
            vendor = "unknown"
            desc_lower = name.lower()
            if any(s in desc_lower for s in ("nvidia", "geforce", "rtx", "gtx")):
                vendor = "nvidia"
            elif any(s in desc_lower for s in ("amd", "radeon", "rx ")):
                vendor = "amd"
            elif any(s in desc_lower for s in ("intel", "iris", "hd graphics")):
                vendor = "intel"
            gpus.append(GPUInfo(
                name=name,
                vendor=vendor,
                total_vram_mb=total_mb,
                free_vram_mb=free_mb,
            ))

    return SystemInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_cores_physical,
        cpu_cores_logical=cpu_cores_logical,
        total_ram_gb=vm.total / (1024 ** 3),
        free_ram_gb=vm.available / (1024 ** 3),
        ram_percent=vm.percent,
        cpu_percent=cpu_percent,
        gpus=gpus,
    )


def print_system_info(info: Optional[SystemInfo] = None) -> None:
    """Drucke System-Infos formatiert aus."""
    if info is None:
        info = detect_system()

    print("=" * 60)
    print("System-Informationen (psutil/pywin32 Test)")
    print("=" * 60)
    print(f"OS:            {info.os_name}")
    print(f"CPU:           {info.cpu_name}")
    print(f"CPU-Kerne:     {info.cpu_cores_physical} physisch / {info.cpu_cores_logical} logisch")
    print(f"CPU-Auslastung: {info.cpu_percent:.1f}%")
    print(f"RAM gesamt:    {info.total_ram_gb:.2f} GB")
    print(f"RAM verf\u00fcgbar: {info.free_ram_gb:.2f} GB")
    print(f"RAM genutzt:   {info.ram_percent:.1f}%")
    print("-" * 60)
    print("GPUs:")
    if info.gpus:
        for i, gpu in enumerate(info.gpus):
            print(f"  GPU{i}: [{gpu.vendor.upper()}] {gpu.name}")
            print(f"    VRAM gesamt:   {gpu.total_vram_gb:.2f} GB")
            print(f"    VRAM frei:     {gpu.free_vram_gb:.2f} GB ({gpu.free_vram_mb * 100 // gpu.total_vram_mb if gpu.total_vram_mb > 0 else 0}%)")
    else:
        print("  Keine GPUs erkannt.")
    print("=" * 60)


def live_monitor_text() -> None:
    """Text-basierter Live-Monitor f\u00fcr die Konsole."""
    print("\n" + "=" * 60)
    print("Live-Monitor (STRG+C zum Beenden)")
    print("=" * 60)
    try:
        while True:
            vm = psutil.virtual_memory()
            cpu_pct = psutil.cpu_percent(interval=0.5)
            
            gpus: List[GPUInfo] = []
            gpus.extend(_get_nvidia_gpu_info())
            gpus.extend(_get_amd_gpu_info())
            if not gpus:
                gpus.extend(_get_gpu_info_powershell())

            print(f"\rCPU: {cpu_pct:.1f}%  |  "
                  f"RAM: {vm.available / (1024**3):.1f}/{vm.total / (1024**3):.1f} GB ({vm.percent:.1f}%)  |  ", end="", flush=True)
            
            for i, gpu in enumerate(gpus):
                free_pct = (gpu.free_vram_mb / gpu.total_vram_mb * 100) if gpu.total_vram_mb > 0 else 0
                print(f"GPU{i}: {gpu.free_vram_gb:.1f}/{gpu.total_vram_gb:.1f} GB ({free_pct:.0f}%)  |  ", end="", flush=True)
            
            print()
    except KeyboardInterrupt:
        print("\nMonitor beendet.")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System- und VRAM-\u00dcberwachung")
    parser.add_argument("--gui", action="store_true", help="PyQt6-GUI starten")
    parser.add_argument("--live", action="store_true", help="Text-basierter Live-Monitor")
    parser.add_argument("--json", action="store_true", help="Als JSON ausgeben")
    args = parser.parse_args()

    if args.gui:
        start_gui_monitor()
    elif args.live:
        live_monitor_text()
    elif args.json:
        info = detect_system()
        data = {
            "os": info.os_name,
            "cpu": info.cpu_name,
            "cpu_cores": {"physical": info.cpu_cores_physical, "logical": info.cpu_cores_logical},
            "cpu_percent": info.cpu_percent,
            "ram": {
                "total_gb": round(info.total_ram_gb, 2),
                "free_gb": round(info.free_ram_gb, 2),
                "percent": info.ram_percent,
            },
            "gpus": [
                {
                    "name": gpu.name,
                    "vendor": gpu.vendor,
                    "total_vram_gb": round(gpu.total_vram_gb, 2),
                    "free_vram_gb": round(gpu.free_vram_gb, 2),
                    "free_percent": round(gpu.free_vram_mb * 100 / gpu.total_vram_mb, 1) if gpu.total_vram_mb > 0 else 0,
                }
                for gpu in info.gpus
            ],
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("Teste GPU-Erkennung...")
        print()

        # Teste vendor-spezifische Tools
        print("1. NVIDIA (nvidia-smi):")
        nvidia_gpus = _get_nvidia_gpu_info()
        for gpu in nvidia_gpus:
            print(f"   {gpu.name}: {gpu.total_vram_gb:.2f} GB gesamt, {gpu.free_vram_gb:.2f} GB frei")
        if not nvidia_gpus:
            print("   Nicht gefunden oder nicht installiert.")

        print()
        print("2. AMD (rocm-smi):")
        amd_gpus = _get_amd_gpu_info()
        for gpu in amd_gpus:
            print(f"   {gpu.name}: {gpu.total_vram_gb:.2f} GB gesamt, {gpu.free_vram_gb:.2f} GB frei")
        if not amd_gpus:
            print("   Nicht gefunden oder nicht installiert.")

        print()
        print("3. PowerShell (WMI + Registry):")
        ps_gpus = _get_gpu_info_powershell()
        for gpu in ps_gpus:
            print(f"   {gpu.name}: {gpu.total_vram_gb:.2f} GB gesamt, {gpu.free_vram_gb:.2f} GB frei")
        if not ps_gpus:
            print("   Nicht gefunden.")

        print()
        print("4. Registry (winreg):")
        reg_gpus = _get_vram_from_registry()
        for name, vram in reg_gpus.items():
            print(f"   {name}: {vram / (1024**3):.2f} GB")
        if not reg_gpus:
            print("   Nicht gefunden.")

        print()
        print("Vollst\u00e4ndige System-Info:")
        print_system_info()
