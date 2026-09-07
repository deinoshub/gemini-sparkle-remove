# Bundled static CPU ncnn

Prebuilt `lib/libncnn.a` (Tencent/ncnn, Vulkan off, shared off) for the
`video-fdncnn` in-process FDnCNN path.

Rebuild with:

```bash
./scripts/build_ncnn_static.sh
```

Headers under `include/` come from the same build (`platform.h`, `ncnn_export.h`, …).
`shim/gwr_ncnn_opt.cpp` sets FDnCNN Option flags (including `use_fp16_arithmetic=false`)
not exposed by `c_api.h`.
