# Video watermark maps

Embedded alpha (+ optional RGB) overlays used by the optional `video` feature
for Gemini diamond reverse-blend on video frames.

Maps and the FDnCNN weights were derived from GeminiWatermarkTool-Video /
NcnnDenoiser; they are approximate, not a bit-exact dump of that tool.

## `diamond_alpha_720p_48x48.f32`

| | |
|---|---|
| Size | 48×48 |
| Format | little-endian `f32`, row-major, values in \[0, 1\] |
| Bytes | 9216 (`48 * 48 * 4`) |
| Peak α | ≈ 0.345 |

LANCZOS-downscaled from a 96×96 grayscale diamond. This is **not** the
still-image `assets/sparkle_rgba.bin` template (different capture / peak α).

## `diamond_rgb_720p_48x48.bin`

| | |
|---|---|
| Size | 48×48×3 |
| Format | raw RGB8, row-major |

**Not used as blend logo RGB.** This file is an alpha-shaped grayscale
preview (high-α pixels ≈ mid-gray 127). Runtime maps set `rgb: None` so the
template uses **pure white** chrome; calibrated scale ≈ 0.97–1.0.

## `diamond_alpha_720p_48x48.png`

Human-readable preview of the alpha map (8-bit L). Not loaded at runtime.

## Compact 44×44

`diamond_map_720p_compact()` returns `None`. No authentic 44×44 map is
shipped; do not invent one by rescaling.

## Geometry prior (720p standard)

On 1280×720 the 48×48 diamond sits with bottom-right margin ≈ 72px →
top-left `(1280 − 72 − 48, 720 − 72 − 48) = (1136, 576)`.

## `diamond_alpha_1080p_72x72.f32`

| | |
|---|---|
| Size | 72×72 |
| Format | little-endian `f32`, row-major, values in \[0, 1\] |
| Bytes | 20736 (`72 * 72 * 4`) |
| Peak α | ≈ 0.365 |

Same 96×96 source as the 720p map, LANCZOS-downscaled to 72×72
(= 1.5× the 48×48 720p operating size).

## Geometry prior (1080p standard)

On 1920×1080 the 72×72 diamond sits with bottom-right margin ≈ 144px →
top-left `(1920 − 144 − 72, 1080 − 144 − 72) = (1704, 864)`.

## Operating scale + residual cleanup

1. Adaptive alpha nominal **0.96** (embedded map peak ≈ 0.345).
2. Classical path: opaque-unrecoverable fill, then an optional thin `|∇α|`
   ring. No full-footprint TELEA.
3. With feature `video-fdncnn`, FDnCNN runs in-process via bundled libncnn
   (no Python / OpenCV).

## `fdncnn/` (feature = `"video-fdncnn"`)

| File | Role |
|------|------|
| `fdncnn.param.bin` | NCNN binary param (magic 7767517, 21 layers; blobs 0→20) |
| `fdncnn.bin` | FDnCNN weights (~1.3 MB) |

Same model family as froggeric/allenk `NcnnDenoiser`.

Runtime call: `sigma=75`, `strength=1.8` (180%), `padding=64` → ROI
176×176 (720p 48 map) / 200×200 (1080p 72 map); gradient-masked footprint.
Input is 4-channel `[R,G,B,σ/255]` float; output is denoised RGB (not a
residual). Blend only inside the footprint weight mask.

Runtime: in-process Rust (`src/video/fdncnn.rs`) + `third_party/ncnn` — no Python.
