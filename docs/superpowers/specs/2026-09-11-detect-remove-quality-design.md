# Cross-pipeline quality: image detect recall + video (non-FDnCNN) remove

Date: 2026-09-11
Status: approved in conversation; awaiting spec review
Approach: A — cross-pollinate techniques, keep two pipelines

## Summary

Still-image removal is already good; misses are almost all detector failures. Video (non-FDnCNN) detection is already good; leftovers are both a ghost diamond outline (under-subtract) and a plastic smear (full-footprint TELEA). This change borrows the other pipeline’s strength for each failure, without unifying public APIs or touching FDnCNN.

## Problem

| Path | Detect | Remove |
|------|--------|--------|
| Image | Silhouette search is diagonal-only (`inset_x == inset_y`). Best-survival candidate is gated once; a failing impostor yields `NotFound` even if a real mark sits nearby. Wide Gate 1 stays at 0.80. NCC fallback uses `MIN_NCC = 0.70` then Gate 2 ≤ 3.0. | `reverse_alpha_blend` on the sparkle RGBA template. Keep as-is. |
| Video (no FDnCNN) | NCC on diamond maps, `MIN_NCC = 0.55`, multi-frame probe, independent X/Y. Keep as-is. | Reverse-blend then TELEA over the **full α footprint** (`FOOT_HARD_MIX = 0.90`, `RESID_STRENGTH = 0.92`). Smears recovered texture. Remaining outline is the blend/alpha miss that fill was papering over. |

Observed miss mix on stills: most common is a clear bottom-right mark that still returns `NotFound`; faint/JPEG/busy-texture and off-diagonal / non-48px also occur.

Video leftovers are type C: outline **and** smear.

## Goals

1. Raise still-image detect recall on clear BR marks, off-diagonal-within-canonical, and busy-texture marks that already invert cleanly.
2. Keep still-image precision: gravel-like impostors (~NCC 0.61) must not become proposals; documented rock impostor (~Gate 2 3.9) must still fail; wood (~3.46) and faint asphalt/gravel fixtures must still pass.
3. On video without FDnCNN: reverse-blend is the recovery step; residual fill is optional and small. Ghost outline drops via better per-shot alpha. Plastic smear drops by deleting full-footprint TELEA.
4. FDnCNN path, video detect, CLI, and public API stay behavior-compatible.

## Non-goals

- Unifying image/video into one detect/remove API.
- Extracting shared NCC helpers from `src/detect.rs` and `src/video/detect.rs`.
- Changing video detection, Veo marks, or the FDnCNN postpass (`blend → NcnnDenoiser`, no TELEA).
- Feeding the still sparkle RGBA template into video (different mark family).
- Using the on-disk grayscale diamond preview as blend RGB.
- Shipping large video fixtures or gating CI on `examples/quality_*`.
- New CLI flags.

## Constraints (from review)

- False-positive policy: **B or C** — balance / precision-first, not recall-at-all-costs. Do not lower `MIN_NCC` below 0.70.
- Image remove stays reverse-blend only (no TELEA on stills).
- Video alpha stays one scale per shot (no per-frame flicker).
- `force_alpha` still bypasses adaptive lock and the new silhouette pick.

## Architecture

Two entry points unchanged:

```
image:  match_watermark → reverse_alpha_blend
video:  detect_from_probe_frames → lock alpha → reverse_alpha_blend → residual policy
```

Residual policy after reverse-blend:

| Path | Residual |
|------|----------|
| Image | None (unchanged). |
| Video, no `video-fdncnn` | Opaque-unrecoverable pixels only; optional thin `\|∇α\|` ring if silhouette survival stays high. No full-footprint TELEA, no `FOOT_HARD_MIX`. |
| Video + `video-fdncnn` | Unchanged: `remove_on_frame_blend_only` then FDnCNN. |

`remove_gemini_sparkle` / `remove_at` / `remove_video` / `VideoRemoveOptions` signatures do not change.

## Image detect (`src/detect.rs`)

`match_watermark` keeps the three-tier order: canonical silhouette → wide silhouette → NCC.

### Canonical window (88–104)

Search **independent** `inset_x` and `inset_y` in `[88, 104]`. Gate 1 stays `MAX_SILHOUETTE_SURVIVAL_CANON = 0.98`. Do not widen the window; wide + NCC still cover placements outside it.

Size `42..=56` and opacity scales `[1.0, 1.25, 1.55, 1.9]` stay.

### Wide window (40–116)

Keep **diagonal** insets only (cost: a 77×77 independent scan is out of scope). Gate 1 stays `MAX_SILHOUETTE_SURVIVAL = 0.80` so off-mark rock fits still fail.

### Candidate selection (canonical and wide)

Today: keep the single lowest-survival placement, then run gates; failure → `None`.

Change: keep the **top-K = 5** placements by ascending survival. Return the first that passes `gates_ok` for that tier’s `max_survival`. If none pass, that tier returns `None` and the next tier runs.

A candidate is a distinct `(x, y, size, template)` (opacity is baked into `template`). Ties: lower survival first; if equal, smaller `|inset_x - inset_y|` then smaller size.

### NCC fallback

Unchanged:

- Independent margins in `[MIN_INSET, MAX_INSET]`.
- `MIN_NCC = 0.70` (gravel false peaks ~0.61 never propose).
- Size step 2.
- Reject if removal **adds** silhouette edges (`survival >= NCC_SURVIVAL` with `NCC_SURVIVAL = 1.0`).

Gate 2 change, NCC tier only:

- If `survival <= 0.85`, use `STRONG_GATE2 = 3.75` (wood ~3.46 passes, rock ~3.9 fails).
- Else keep `MAX_SILHOUETTE_VS_CONTROL = 3.0`.

Do not lower `MIN_NCC`. Do not drop Gate 2. Canonical/wide `gates_ok` logic is otherwise unchanged (`STRONG_SURVIVAL = 0.70` path stays).

### Image remove

No change. `remove_gemini_sparkle` still calls `reverse_alpha_blend` with `OPAQUE_CUTOFF = 0.95` on the winning template.

## Video remove, non-FDnCNN

Video detect (`src/video/detect.rs`) is unchanged. Overlay RGB stays white (`VideoMap.rgb = None`).

### Per-shot alpha (`src/video/alpha.rs` + `pipeline.rs`)

1. Existing `seed_alpha_locked`: median of `refine_alpha_bisection` on five evenly spaced frames; if median `< 0.95`, lift to `1.0`.
2. **New silhouette pick** on the same probe frames. Trial scales: `{seed, 1.0, 1.05, 1.12}` (unique, each clamped to `[0.78, 1.12]`). For each scale, reverse-blend only on each probe ROI and score mean silhouette survival (same idea as still `silhouette_edges`: edge energy after/before, weighted by `|∇α|`).
3. Discard a scale that **digs a dark hole**: after blend, mean luma of pixels with map `α > 0.05` is more than **6.0** below the mean luma of the exterior ring (map `α < 0.02`, inside the ROI). If every scale is discarded, keep `seed`.
4. Among remaining scales, pick the **lowest mean survival**. Ties: scale closer to `1.0`.
5. `VideoRemoveOptions.force_alpha = Some(_)` skips steps 1–4 and uses the forced value clamped to `[0.05, 2.0]` as today.
6. Apply one scale to every frame.

Log one line: `alpha_scale={picked:.4} seed={seed:.4} region=(x,y,w,h)` (replaces the current `seed_alpha_locked scale=…` line).

`SCALE_MAX` used inside `estimate_alpha` / `refine_alpha_bisection` stays `1.05`. The new pick’s ceiling `1.12` applies only to this discrete trial set, not to the LS estimator.

### Residual (`src/video/frame.rs`)

`remove_on_frame` (non-FDnCNN) after `reverse_alpha_blend`:

1. **Opaque fill.** Inpaint only pixels whose reverse-blend mask is set (`α_template >= 0.95`). Diamond peak ~0.34, so this is usually empty. Use existing `inpaint_telea` on that mask only (radius 5 at 48×48, 7 at 72×72).
2. **Thin ring, conditional.** If ROI silhouette survival after step 1 is **> 0.40**, build a ring: `α >= 0.008` and `|∇α| > 0.022`, **dilate 1** (not 3). Do **not** union the dilated full footprint (`FOOT_ALPHA_THR` / `FOOT_DILATE` / `FOOT_HARD_MIX` go away). Mix filled vs blended with strength **0.40** (not 0.92), no extra hard mix. Optional grain reinjection **only** on this ring, at current `GRAIN_STRENGTH` or lower; do not grain the whole footprint.
3. If survival ≤ 0.40, skip step 2 entirely — leave reverse-blend pixels.

`remove_on_frame_blend_only` is unchanged (alpha trials and FDnCNN).

Do not run full-footprint TELEA on any path. Do not put TELEA before FDnCNN.

`inpaint_telea` stays in the crate (opaque mask, optional ring).

## Error handling

- Image: no candidate passing any tier → `RemoveResult::NotFound`, buffer unchanged (existing contract).
- Video: probe detect miss → existing `VideoError::Detect`. This spec does not loosen video detect.
- Invalid geometry / empty frames: existing `VideoError::InvalidInput`.
- FDnCNN postpass failure: existing stderr warning; frames still encoded (unchanged).
- `force_alpha` out of range: existing clamp `[0.05, 2.0]`.

No new error variants.

## Testing

### Image — must keep passing

- `detects_sparkle_on_fixture` (~1255, 647, residual ≤ 0.80)
- `detects_sparkle_on_2k_gravel_crop` / `_wood_crop` / `_asphalt_crop`
- `unmarked_busy_noise_is_not_a_match`
- `detects_off_diagonal_sparkle_via_ncc` (insets 96×72, outside canonical — still NCC)
- `tests/api_remove.rs`: fixture `Removed`, solid `NotFound` unchanged, `remove_at`

### Image — add

- Off-diagonal **inside** canonical: synthetic sparkle at insets 96×92 on a mid-grey canvas; `match_watermark` hits within 4 px; this must succeed even if the NCC tier is skipped (canonical independent search).
- Top-K: two composited candidates; the lowest-survival one fails gates (e.g. a bright square that does not match the sparkle silhouette); the real sparkle nearby must still be returned.
- Precision: high-frequency noise without an overlay stays `NotFound`. Overlay the sparkle on that noise → `Removed` near the planted origin.

### Video non-FDnCNN — must keep / add

- Keep `synthetic_diamond_patch_recovers_background` (high-α pixels within ±3 of the solid bg).
- Anti-smear: unit test in `src/video/frame.rs` on a synthetic checker/noise background. After `remove_on_frame`, per-pixel RGB stddev inside the high-α footprint must be at least 50% of the stddev of the exterior ring (map `α < 0.02`). A full-footprint TELEA mix at the old 0.90/0.92 strengths on the same buffer must fall below that 50% bar, so the new path is stricter than the old fill.
- Dark-hole: a probe whose seed is under-locked must not pick a scale that drops high-α luma more than 6.0 below the exterior ring.
- `force_alpha: Some(s)` is the scale used; silhouette pick does not run (assert via a hook, a `pub(crate)` pick function that pipeline skips, or by checking that an adversarial seed would have picked a different scale).
- `remove_on_frame_blend_only` tests and map-to-template tests stay.

### Must not regress (run, no logic change)

- `cargo test`
- `cargo test --features video`
- `cargo test --features video-fdncnn` (or `video-fdncnn,system-ffmpeg` as CI does)
- CLI classify tests

`examples/quality_*` remain manual.

## Implementation order

Stop after each step if tests for that step fail.

1. **`src/detect.rs`**: independent canonical insets; top-K=5; NCC Gate 2 uses 3.75 when `survival <= 0.85`. Add still-image tests. `cargo test` green.
2. **`src/video/alpha.rs`**: silhouette pick helper; hole reject; ceiling 1.12 on the trial set only. Unit tests for pick/hole/`force_alpha` bypass.
3. **`src/video/frame.rs`**: delete full-footprint TELEA / `FOOT_HARD_MIX`; opaque-only then conditional thin ring at mix 0.40. Update synthetic tests; add anti-smear.
4. **`src/video/pipeline.rs`**: call silhouette pick after `seed_alpha_locked` unless `force_alpha` is set; log chosen scale. `cargo test --features video` and `--features video-fdncnn`.

Do not extract a shared NCC module in this work.

## Files touched (expected)

| File | Change |
|------|--------|
| `src/detect.rs` | Canonical independent search, top-K, NCC Gate 2, tests |
| `src/video/alpha.rs` | Silhouette pick + hole reject + tests |
| `src/video/frame.rs` | Residual policy + tests |
| `src/video/pipeline.rs` | Wire pick, logging |
| `src/lib.rs` | No change |
| `README.md` | No change |

## Success criteria

- Existing still fixtures still detect at the documented coordinates.
- New canonical off-diagonal synthetic detects without relying on NCC.
- Noise-only stills stay `NotFound`.
- Non-FDnCNN synthetic diamond still recovers the background.
- Non-FDnCNN textured synthetic does not collapse to a plastic patch.
- FDnCNN unit tests pass unchanged.
- No public API or CLI flag changes.
