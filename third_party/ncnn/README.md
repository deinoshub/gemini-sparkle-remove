# Bundled static CPU ncnn

`video-fdncnn` cmake-builds Tencent/ncnn (Vulkan off, OpenMP off, shared off)
for the Cargo **TARGET** into `third_party/ncnn-src/build-<triple>/` (gitignored).

Pinned tag: **`20260526`**. Override with `NCNN_SRC` / `NCNN_REV`.

```bash
./scripts/build_ncnn_static.sh
# or just:
cargo build --features video-fdncnn,system-ffmpeg
```

Requires CMake, a C++ compiler, and git (first clone).

`shim/gwr_ncnn_opt.cpp` sets FDnCNN Option flags (including `use_fp16_arithmetic=false`)
not exposed by `c_api.h`.
