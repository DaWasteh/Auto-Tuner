# Ubuntu / Vulkan recipe for Diffusion-Gemma PR #24427.
# Output is backend-qualified so it cannot be confused with HIP or CUDA.
BASE_DIR="/home/dawasteh/local_ai"

mkdir -p "$BASE_DIR"
cd "$BASE_DIR" || exit 1


get_llama_build_tag() {
    commit="${1:-HEAD}"
    remote="${2:-origin}"

    sha=$(git rev-parse "$commit" 2>/dev/null || true)
    if [ -n "$sha" ]; then
        # llama.cpp b10470+ pushes lightweight b-tags explicitly from CI. Ask
        # the remote tag namespace directly first so just-created tags are seen.
        remote_tag=$(git ls-remote --tags "$remote" 'refs/tags/b*' 2>/dev/null |
            awk -v sha="$sha" '$1 == sha { sub("refs/tags/", "", $2); print $2 }' |
            grep -E '^b[0-9]+$' |
            sort -t b -k 2,2nr |
            head -n 1)
        if [ -n "$remote_tag" ]; then
            printf '%s\n' "$remote_tag"
            return 0
        fi
    fi

    git fetch "$remote" --tags --force 2>/dev/null || true

    tag=$(git tag --points-at "$commit" --list 'b[0-9]*' 2>/dev/null |
        grep -E '^b[0-9]+$' |
        sort -t b -k 2,2nr |
        head -n 1)
    if [ -n "$tag" ]; then
        printf '%s\n' "$tag"
        return 0
    fi

    tag=$(git describe --tags --abbrev=0 --match 'b[0-9]*' "$commit" 2>/dev/null || true)
    if printf '%s' "$tag" | grep -Eq '^b[0-9]+$'; then
        printf '%s\n' "$tag"
        return 0
    fi

    printf '%s\n' 'bUNKNOWN'
}


# SPIRV-Headers nur bauen, falls noch nicht vorhanden (liegt dort schon
# vom Mainline-Build)
if [ ! -d "SPIRV-Headers/install" ]; then
    if [ ! -d "SPIRV-Headers" ]; then
        git clone https://github.com/KhronosGroup/SPIRV-Headers.git
    fi
    cmake -S ./SPIRV-Headers -B ./SPIRV-Headers/build \
      -DCMAKE_INSTALL_PREFIX="$BASE_DIR/SPIRV-Headers/install"
    cmake --build ./SPIRV-Headers/build --config Release
    cmake --install ./SPIRV-Headers/build --config Release
fi

# --- Diffusion-Gemma: PR #24427 (noch nicht in mainline gemerged) ---
git clone https://github.com/ggml-org/llama.cpp.git _tmp_d_llama

pushd _tmp_d_llama > /dev/null
git fetch origin pull/24427/head:pr-diffusiongemma
git checkout pr-diffusiongemma
# b-Nummer der mainline-Basis ermitteln, auf der der PR aufsetzt
git fetch origin --tags
base=$(git merge-base HEAD origin/master 2>/dev/null)
[ -z "$base" ] && base="HEAD"
ver=$(get_llama_build_tag "$base" origin)
popd > /dev/null

dir="d_${ver}_ubuntu_vulkan_llama.cpp"
if [ -d "$dir" ]; then
    rm -rf "$dir"
fi
mv _tmp_d_llama "$dir"

echo "==> Build-Verzeichnis: $BASE_DIR/$dir"

# --- UI: layout-tolerant (tools/ui ab b9174, sonst tools/server/webui) ---
ui_src=""
for cand in "tools/ui" "tools/server/webui"; do
    if [ -f "$dir/$cand/package.json" ]; then ui_src="$cand"; break; fi
done
if [ -n "$ui_src" ]; then
    echo "==> UI aus Source bauen ($dir/$ui_src)"
    pushd "$dir/$ui_src" > /dev/null
    if [ -f package-lock.json ]; then npm ci || npm install; else npm install; fi
    npm run build
    popd > /dev/null
    ui_prebuilt="OFF"
else
    echo "==> keine UI-Quellen in $dir — verwende Prebuilt-UI von HF"
    ui_prebuilt="ON"
fi

cmake -S "./$dir" -B "./$dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON \
  -DGGML_FMA=ON \
  -DGGML_F16C=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=ON \
  -DLLAMA_USE_PREBUILT_UI=$ui_prebuilt \
  -DLLAMA_CURL=OFF \
  -DGGML_CCACHE=OFF \
  -DGGML_VULKAN_CHECK_RESULTS=OFF \
  -DCMAKE_PREFIX_PATH="$BASE_DIR/SPIRV-Headers/install"

cmake --build "./$dir/build" --config Release --parallel $(nproc)

# Ergebnis: $dir/build/bin/llama-diffusion-gemma-server (+ -gemma-cli).
# Danach im AutoTuner-Fork-Dropdown den d_*-Build auswählen — der
# Resolver findet llama-diffusion-gemma-server auch als Sibling von
# LLAMA_CPP_DIR automatisch.
