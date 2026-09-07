# Video watermark maps

Embedded alpha (+ optional RGB) overlays used by the optional `video` feature
for Gemini diamond / Veo reverse-blend on video frames.

## `diamond_alpha_720p_48x48.f32`

| | |
|---|---|
| Size | 48×48 |
| Format | little-endian `f32`, row-major, values in \[0, 1\] |
| Bytes | 9216 (`48 * 48 * 4`) |
| Peak α | ≈ 0.345 |

**Provenance**

1. Extracted grayscale diamond alpha PNG (96×96, mode L, peak byte 94) from
   embedded resources inside `/workspace/tools/GeminiWatermarkTool-Video`
   (PNG at file offset ≈ `40458592`).
2. LANCZOS-downscaled to 48×48 and stored as float opacity (`byte / 255`).
3. **Validated** against `/workspace/video-in/input.mp4` vs GWT output
   `/workspace/video-out/cleaned.mp4` at the known diamond ROI
   `(x, y, w, h) = (1136, 576, 48, 48)` on 1280×720:
   - Per-frame estimate assuming near-white overlay:
     `α ≈ (watermarked − cleaned) / (255 − cleaned)` on luma, median across
     start/mid/end frames.
   - Pearson correlation vs this embedded map: **≈ 0.99**; RMSE ≈ 0.02.
   - Best scale `calib ≈ s · map` with `s ≈ 0.97` (near unity — map already at
     operating opacity; per-frame adaptive scale still applies later).

This is **not** the still-image `assets/sparkle_rgba.bin` template (different
capture / peak α). Treat it as GWT-derived and approximate vs any future GWT
golden rebuilds.

## `diamond_rgb_720p_48x48.bin`

| | |
|---|---|
| Size | 48×48×3 |
| Format | raw RGB8, row-major |
| Source | Companion RGB diamond PNG (48×48) from the same GWT binary
  (offset ≈ `40470848`) |

**Not used as blend logo RGB.** This file is an alpha-shaped grayscale
preview (Pearson corr with the α map ≈ 0.997; high-α pixels ≈ mid-gray 127).
Feeding it to `reverse_alpha_blend` under-removes (~50% of GWT BR mean-diff
on the sample clip). Runtime maps set `rgb: None` so the template uses
**pure white** chrome; calibrated scale ≈ 0.97–1.0 matches GWT.

## `diamond_alpha_720p_48x48.png`

Human-readable preview of the alpha map (8-bit L). Not loaded at runtime.

## Compact 44×44

`diamond_map_720p_compact()` returns `None` in phase 1. GWT advertises
`720p-2 compact` profiles, and a 36×36 grayscale diamond exists in the binary,
but no authentic **44×44** resource was recovered. Do not invent one by
rescaling without a documented golden.

## Geometry prior (720p standard)

On 1280×720, GWT places the 48×48 diamond with bottom-right margin ≈ 72px →
top-left `(1280 − 72 − 48, 720 − 72 − 48) = (1136, 576)`, matching the sample
hit used for calibration.

## `diamond_alpha_1080p_72x72.f32`

| | |
|---|---|
| Size | 72×72 |
| Format | little-endian `f32`, row-major, values in \[0, 1\] |
| Bytes | 20736 (`72 * 72 * 4`) |
| Peak α | ≈ 0.365 (byte 93 / 255) |

**Provenance**

1. Same GWT 96×96 grayscale diamond PNG as the 720p map (offset ≈ `40458592`).
2. LANCZOS-downscaled to 72×72 (= 1.5× the 48×48 720p operating size).
3. **Validated** vs GWT on 1920×1080 user clip: locked region `(1704, 864, 72, 72)`
   (BR margin 144), mid-frame NCC ≈ 0.96 vs this map.

## Geometry prior (1080p standard)

On 1920×1080, GWT places the 72×72 diamond with bottom-right margin ≈ 144px →
top-left `(1920 − 144 − 72, 1080 − 144 − 72) = (1704, 864)`.

## Operating scale + residual cleanup (2026-09-06)

GWT video mode always runs FDnCNN edge denoise (`sigma=75`, `strength=180%`)
even with `--denoise off`, over the full map footprint (48×48 → 2304 edge
pixels). Our classical path:

1. Adaptive alpha nominal **0.92** (GWT seed ≈ x0.58 on large map peak ≈ 0.51
   → effective peak ≈ 0.30; our map peak ≈ 0.345 → 0.92 ≈ match).
2. Full-footprint onion-peel fill + exterior HF grain in Rust.
3. In-process onion-peel TELEA-style fill in Rust (`src/video/telea.rs`).
   With feature `video-fdncnn`, GWT-matching FDnCNN runs in-process via bundled libncnn
   (no Python / OpenCV).

## `fdncnn/` (feature = `"video-fdncnn"`)

| File | Role |
|------|------|
| `fdncnn.param.bin` | NCNN binary param (magic 7767517, 21 layers; blobs 0→20) |
| `fdncnn.bin` | FDnCNN weights (~1.3 MB) |

**Provenance:** extracted from `/workspace/tools/GeminiWatermarkTool-Video` (param @ ≈39107116, bin @ ≈39108580). Same model as froggeric/allenk `NcnnDenoiser`.

**GWT video call:** `sigma=75`, `strength=1.8` (180%), `padding=64` → ROI 176×176 (720p 48 map) / 200×200 (1080p 72 map); gradient-masked footprint (Sobel|∇α|; at strength=180% → 2304 / 5184 edge px). Input is 4-channel `[R,G,B,σ/255]` float; output is denoised RGB (not a residual). Blend only inside the footprint weight mask.

Runtime: in-process Rust (`src/video/fdncnn.rs`) + `third_party/ncnn` — no Python.
