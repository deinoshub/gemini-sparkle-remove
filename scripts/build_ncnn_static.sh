#!/usr/bin/env bash
# Build static CPU libncnn for the host (or $TARGET) via CMake.
# cargo build --features video-fdncnn also does this from build.rs when missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${NCNN_SRC:-$ROOT/third_party/ncnn-src}"
REV="${NCNN_REV:-20260526}"
TARGET="${TARGET:-$(rustc --print host-tuple 2>/dev/null || uname -m)}"
BUILD="${SRC}/build-${TARGET}"

if [[ -n "${JOBS:-}" ]]; then
  :
else
  # Cap at 2 by default — unlimited -j OOMs 7GB GitHub Ubuntu runners
  # when cargo is also compiling rav1e in the same job.
  JOBS=2
fi

if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
  git clone --depth 1 --branch "$REV" https://github.com/Tencent/ncnn.git "$SRC"
fi

cmake -S "$SRC" -B "$BUILD" \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF -DNCNN_OPENMP=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --config Release --parallel "$JOBS"
echo "Built ncnn in $BUILD"
