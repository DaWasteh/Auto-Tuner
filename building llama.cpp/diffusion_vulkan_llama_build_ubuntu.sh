#!/usr/bin/env bash
set -Eeuo pipefail

# Ubuntu / Vulkan recipe for the reviewed Diffusion-Gemma PR #24427.
# The source commit matches the Windows Vulkan/HIP siblings exactly.
# Usage: ./diffusion_vulkan_llama_build_ubuntu.sh [workspace]

BASE_DIR="${1:-${LLAMA_BUILD_WORKSPACE:-/home/dawasteh/local_ai}}"
REMOTE_URL="https://github.com/ggml-org/llama.cpp.git"
FETCH_REF="pull/24427/head"
EXPECTED_COMMIT="dd0cf04459b0c4f43aa6667dbc0879ac0cd50323"
PARALLEL="${LLAMA_BUILD_PARALLEL:-$(nproc)}"

if [[ ! "$PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid LLAMA_BUILD_PARALLEL value: %s\n' "$PARALLEL" >&2
    exit 2
fi
for tool in git cmake; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf "Required command '%s' was not found in PATH\n" "$tool" >&2
        exit 1
    fi
done

mkdir -p "$BASE_DIR"
BASE_DIR="$(cd "$BASE_DIR" && pwd)"
staging="$BASE_DIR/_tmp_d_ubuntu_vulkan_llama_$$"
if [[ -e "$staging" ]]; then
    printf 'Staging directory already exists: %s\n' "$staging" >&2
    exit 1
fi
cleanup() {
    if [[ -n "${staging:-}" && -e "$staging" ]]; then
        rm -rf -- "$staging"
        printf '==> Removed failed Diffusion-Gemma staging tree: %s\n' "$staging" >&2
    fi
}
trap cleanup EXIT

# Build SPIRV-Headers only when the shared installation is absent. Never replace
# an existing checkout or installation implicitly.
spirv="$BASE_DIR/SPIRV-Headers"
if [[ ! -d "$spirv/install" ]]; then
    if [[ ! -e "$spirv" ]]; then
        git clone https://github.com/KhronosGroup/SPIRV-Headers.git "$spirv"
    elif [[ ! -d "$spirv/.git" ]]; then
        printf 'SPIRV-Headers exists but is not a Git checkout: %s\n' "$spirv" >&2
        exit 1
    fi
    cmake -S "$spirv" -B "$spirv/build" \
        -DCMAKE_INSTALL_PREFIX="$spirv/install"
    cmake --build "$spirv/build" --config Release --parallel "$PARALLEL"
    cmake --install "$spirv/build" --config Release
fi

# Fetch and attest the exact reviewed PR head. A moved PR must be reviewed and
# repinned rather than silently changing the produced binary.
git clone "$REMOTE_URL" "$staging"
git -C "$staging" fetch origin "$FETCH_REF"
fetched_commit="$(git -C "$staging" rev-parse FETCH_HEAD)"
if [[ "$fetched_commit" != "$EXPECTED_COMMIT" ]]; then
    printf 'Diffusion-Gemma PR changed: expected %s, fetched %s\n' \
        "$EXPECTED_COMMIT" "$fetched_commit" >&2
    exit 1
fi
git -C "$staging" checkout --detach "$EXPECTED_COMMIT"
actual_commit="$(git -C "$staging" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    printf 'Checkout mismatch: expected %s, found %s\n' \
        "$EXPECTED_COMMIT" "$actual_commit" >&2
    exit 1
fi

base="$(git -C "$staging" merge-base HEAD origin/master)"
build="$(git -C "$staging" rev-list --count "$base")"
if [[ ! "$build" =~ ^[0-9]+$ || "$build" -lt 1000 ]]; then
    printf 'Incomplete history or invalid llama.cpp build number: %s\n' "$build" >&2
    exit 1
fi

dir="d_b${build}_ubuntu_vulkan_llama.cpp"
repo="$BASE_DIR/$dir"
if [[ -e "$repo" ]]; then
    if [[ ! -d "$repo/.git" ]]; then
        printf 'Output exists but is not a Git checkout; nothing replaced: %s\n' "$repo" >&2
        exit 1
    fi
    existing_commit="$(git -C "$repo" rev-parse HEAD)"
    if [[ "$existing_commit" != "$EXPECTED_COMMIT" ]]; then
        printf 'Output exists at a different commit; nothing replaced: %s\n' "$repo" >&2
        exit 1
    fi
    rm -rf -- "$staging"
    staging=""
    printf '==> Reusing exact source tree: %s\n' "$repo"
else
    mv -- "$staging" "$repo"
    staging=""
fi

printf '==> Build directory: %s (b%s, commit %s)\n' \
    "$repo" "$build" "$EXPECTED_COMMIT"

# UI source moved across llama.cpp generations. Build whichever supported layout
# this pinned fork contains, otherwise let CMake use the prebuilt UI.
ui_source=""
for candidate in tools/ui tools/server/webui; do
    if [[ -f "$repo/$candidate/package.json" ]]; then
        ui_source="$repo/$candidate"
        break
    fi
done
ui_prebuilt="ON"
if [[ -n "$ui_source" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
        printf "Required command 'npm' was not found in PATH\n" >&2
        exit 1
    fi
    if [[ ! -d "$ui_source/dist" ]]; then
        printf '==> Building Web UI from %s\n' "$ui_source"
        pushd "$ui_source" >/dev/null
        if [[ -f package-lock.json ]]; then
            npm ci
        else
            npm install
        fi
        npm run build
        popd >/dev/null
    else
        printf '==> Reusing existing Web UI dist: %s\n' "$ui_source/dist"
    fi
    ui_prebuilt="OFF"
else
    printf '==> No UI source found; using the prebuilt UI\n'
fi

cmake -S "$repo" -B "$repo/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=ON \
    -DLLAMA_USE_PREBUILT_UI="$ui_prebuilt" \
    -DLLAMA_CURL=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_VULKAN_CHECK_RESULTS=OFF \
    -DCMAKE_PREFIX_PATH="$spirv/install"

cmake --build "$repo/build" --config Release --parallel "$PARALLEL"
printf 'Success: %s (Diffusion-Gemma PR #24427, Vulkan, %s)\n' \
    "$repo" "$EXPECTED_COMMIT"
