#!/usr/bin/env bash
set -euo pipefail

# Run AutoTuner's Linux checks inside a clean distribution container.
# Usage:
#   linux-distro-ci.sh source
#   linux-distro-ci.sh binary /artifacts/AutoTuner-Linux-x64.zip

mode="${1:-source}"
artifact="${2:-}"

retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    if (( attempt < 3 )); then
      echo "Command failed (attempt ${attempt}/3); retrying in $((attempt * 5))s..." >&2
      sleep $((attempt * 5))
    fi
  done
  return 1
}

install_runtime_dependencies() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    retry apt-get update
    retry apt-get install -y --no-install-recommends \
      ca-certificates coreutils util-linux unzip procps pciutils \
      libegl1 libgl1 libglib2.0-0 libfontconfig1 libdbus-1-3 \
      libxkbcommon0 libxkbcommon-x11-0 libxcb1 libxcb-cursor0 \
      libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
      libxcb-render-util0 libxcb-shape0 libxcb-sync1 libxcb-xfixes0 \
      libxcb-xinerama0 libxcb-xkb1
    if [[ "$mode" == "source" ]]; then
      retry apt-get install -y --no-install-recommends python3 python3-pip python3-venv
    fi
  elif command -v dnf >/dev/null 2>&1; then
    retry dnf -y install \
      ca-certificates coreutils util-linux unzip procps-ng pciutils \
      mesa-libEGL mesa-libGL glib2 fontconfig dbus-libs \
      libxkbcommon libxkbcommon-x11 libxcb xcb-util-cursor xcb-util-wm \
      xcb-util-image xcb-util-keysyms xcb-util-renderutil
    if [[ "$mode" == "source" ]]; then
      retry dnf -y install python3 python3-pip
    fi
  elif command -v pacman >/dev/null 2>&1; then
    retry pacman -Syu --noconfirm
    retry pacman -S --needed --noconfirm \
      ca-certificates coreutils util-linux unzip procps-ng pciutils \
      libglvnd glib2 fontconfig dbus libxkbcommon libxkbcommon-x11 \
      libxcb xcb-util-cursor xcb-util-wm xcb-util-image \
      xcb-util-keysyms xcb-util-renderutil
    if [[ "$mode" == "source" ]]; then
      retry pacman -S --needed --noconfirm python python-pip
    fi
  else
    echo "Unsupported container: apt-get, dnf, or pacman was not found." >&2
    exit 1
  fi
}

show_distribution() {
  echo "============================================================"
  if [[ -r /etc/os-release ]]; then
    cat /etc/os-release
  fi
  echo "Kernel: $(uname -a)"
  echo "Architecture: $(uname -m)"
  echo "Mode: $mode"
  echo "============================================================"
}

prepare_runtime_environment() {
  export HOME=/tmp/autotuner-home
  export XDG_CONFIG_HOME="$HOME/.config"
  export XDG_CACHE_HOME="$HOME/.cache"
  export XDG_RUNTIME_DIR=/tmp/autotuner-runtime
  export QT_QPA_PLATFORM=offscreen
  export PYTHONDONTWRITEBYTECODE=1
  mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
}

run_source_checks() {
  local workdir=/tmp/autotuner-source
  rm -rf "$workdir"
  mkdir -p "$workdir"
  cp -R /workspace/. "$workdir/"
  cd "$workdir"

  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt pytest

  python -c "import auto_tuner, hardware, scanner, settings_loader, tuner, launcher, qt_launcher; print('all imports OK')"
  python -c "from hardware import detect_system, format_system; system = detect_system(); print(format_system(system))"
  python -c "from settings_loader import load_profiles; from pathlib import Path; profiles = load_profiles(Path('settings')); assert profiles, 'no profiles loaded'; [print(f'  OK  {profile.source_file}') for profile in profiles]"
  python -m pytest -v -p no:cacheprovider
}

run_binary_smoke_test() {
  if [[ -z "$artifact" || ! -f "$artifact" ]]; then
    echo "Linux release artifact not found: ${artifact:-<missing path>}" >&2
    exit 1
  fi

  local workdir=/tmp/autotuner-binary
  local log_file=/tmp/autotuner-binary.log
  local binary
  rm -rf "$workdir" "$log_file"
  mkdir -p "$workdir"
  unzip -q "$artifact" -d "$workdir"
  binary="$(find "$workdir" -type f -name 'AutoTuner-Linux' -print -quit)"
  if [[ -z "$binary" ]]; then
    echo "AutoTuner-Linux was not found inside $artifact" >&2
    find "$workdir" -maxdepth 3 -type f -print >&2
    exit 1
  fi
  chmod +x "$binary"

  echo "Checking shared-library resolution for $binary"
  if ldd "$binary" | tee /tmp/autotuner-ldd.log | grep -q 'not found'; then
    echo "The release binary has unresolved shared libraries." >&2
    exit 1
  fi

  echo "Launching the frozen GUI with QT_QPA_PLATFORM=offscreen..."
  "$binary" >"$log_file" 2>&1 &
  local pid=$!
  sleep 10
  if ! kill -0 "$pid" 2>/dev/null; then
    local status=0
    wait "$pid" || status=$?
    echo "AutoTuner exited during the smoke-test window (status $status)." >&2
    cat "$log_file" >&2 || true
    exit 1
  fi

  kill -TERM "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  echo "Frozen Linux GUI stayed alive for 10 seconds: OK"
  cat "$log_file" || true
}

case "$mode" in
  source|binary) ;;
  *)
    echo "Unknown mode '$mode' (expected 'source' or 'binary')." >&2
    exit 2
    ;;
esac

show_distribution

if [[ "${AUTOTUNER_CI_DEPENDENCIES_READY:-0}" != "1" ]]; then
  install_runtime_dependencies

  # Docker starts as root so dependencies can be installed. AutoTuner is a
  # desktop application, and its mlock safety behavior intentionally differs
  # for root. Run the actual tests/binary as an ordinary user (UID 65534) to
  # match how viewers launch it on their desktop distributions.
  if [[ "$(id -u)" == "0" ]]; then
    exec setpriv --reuid=65534 --regid=65534 --clear-groups \
      env AUTOTUNER_CI_DEPENDENCIES_READY=1 \
      bash "$0" "$@"
  fi
fi

prepare_runtime_environment

if [[ "$mode" == "source" ]]; then
  run_source_checks
else
  run_binary_smoke_test
fi
