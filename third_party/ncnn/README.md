# ncnn for `video-fdncnn`

By default `build.rs` **downloads** the official Tencent/ncnn static zip for the
Cargo TARGET (tag **`20260526`**) into `third_party/ncnn-prebuilt/` (gitignored).

| TARGET | Zip |
|--------|-----|
| `x86_64-unknown-linux-gnu` | `ncnn-*-ubuntu-2404.zip` |
| `*-apple-darwin` | `ncnn-*-macos.zip` (universal) |
| `x86_64-pc-windows-msvc` / `aarch64-pc-windows-msvc` | `ncnn-*-windows-vs2022.zip` |

Other targets, or `GUM_NCNN_FROM_SOURCE=1`, cmake-build from `third_party/ncnn-src`.

```bash
cargo build --features video-fdncnn,system-ffmpeg
# force compile from source:
GUM_NCNN_FROM_SOURCE=1 cargo build --features video-fdncnn
```

`shim/gwr_ncnn_opt.cpp` sets FDnCNN Option flags (including `use_fp16_arithmetic=false`)
not exposed by `c_api.h`.
