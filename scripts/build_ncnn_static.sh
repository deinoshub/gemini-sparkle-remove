#!/usr/bin/env bash
# Build static CPU libncnn into third_party/ncnn (persistent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${NCNN_SRC:-$ROOT/third_party/ncnn-src}"
OUT="$ROOT/third_party/ncnn"
JOBS="$(nproc)"
if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 https://github.com/Tencent/ncnn.git "$SRC"
fi
cmake -S "$SRC" -B "$SRC/build" \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build" -j"$JOBS"
mkdir -p "$OUT/include" "$OUT/lib" "$OUT/shim"
cp -a "$SRC"/src/*.h "$SRC"/src/*.hpp "$OUT/include/" 2>/dev/null || true
cp -a "$SRC"/build/src/*.h "$OUT/include/" 2>/dev/null || true
cp "$SRC/build/src/libncnn.a" "$OUT/lib/"
echo "Installed libncnn.a → $OUT/lib"
