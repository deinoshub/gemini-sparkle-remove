#!/usr/bin/env bash
# Optional: fetch official Tencent/ncnn zip, or cmake-build from source.
# cargo build --features video-fdncnn downloads the zip by default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV="${NCNN_REV:-20260526}"
TARGET="${TARGET:-$(rustc --print host-tuple 2>/dev/null || uname -m)}"
DEST="$ROOT/third_party/ncnn-prebuilt/$REV/$TARGET"
JOBS="${JOBS:-2}"

asset=""
case "$TARGET" in
  x86_64-unknown-linux-gnu) asset="ncnn-${REV}-ubuntu-2404.zip" ;;
  *-apple-darwin) asset="ncnn-${REV}-macos.zip" ;;
  x86_64-pc-windows-msvc|aarch64-pc-windows-msvc) asset="ncnn-${REV}-windows-vs2022.zip" ;;
esac

if [[ -n "$asset" && "${GUM_NCNN_FROM_SOURCE:-}" != "1" ]]; then
  mkdir -p "$DEST"
  if [[ ! -f "$DEST/_ok" ]]; then
    url="https://github.com/Tencent/ncnn/releases/download/${REV}/${asset}"
    echo "Downloading $url"
    curl -fL --retry 3 -o "$DEST/_dl.zip" "$url"
    unzip -o -q "$DEST/_dl.zip" -d "$DEST"
    rm -f "$DEST/_dl.zip"
    touch "$DEST/_ok"
  fi
  echo "ncnn prebuilt in $DEST"
  exit 0
fi

SRC="${NCNN_SRC:-$ROOT/third_party/ncnn-src}"
BUILD="${SRC}/build-${TARGET}"
if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
  git clone --depth 1 --branch "$REV" https://github.com/Tencent/ncnn.git "$SRC"
fi
cmake -S "$SRC" -B "$BUILD" \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF -DNCNN_OPENMP=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --config Release --parallel "$JOBS"
echo "Built ncnn in $BUILD"
