# Image Detect + Video Non-FDnCNN Remove Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise still-image watermark detect recall without lowering precision, and stop non-FDnCNN video remove from leaving both a ghost outline and a plastic smear.

**Architecture:** Keep two pipelines. Image detect borrows video-style independent X/Y search and top-K gate fallback; image remove stays reverse-blend. Video detect is unchanged. Non-FDnCNN video remove borrows still-image recovery (blend first): lock one alpha per shot with a silhouette pick, then opaque-only plus a conditional thin `|∇α|` ring — never full-footprint TELEA. FDnCNN stays blend → denoise.

**Tech Stack:** Rust 2021, `cargo test` (default / `--features video` / `--features video-fdncnn`), existing `image` + `rayon` + ffmpeg only where already used. No new crates.

**Spec:** `docs/superpowers/specs/2026-09-11-detect-remove-quality-design.md`

## Global Constraints

- Do not lower `MIN_NCC` below `0.70`.
- Do not extract a shared NCC module from `src/detect.rs` and `src/video/detect.rs`.
- Do not change public APIs: `remove_gemini_sparkle`, `remove_at`, `remove_video`, `VideoRemoveOptions`, CLI flags.
- Do not change video detection, Veo, or the FDnCNN postpass (`remove_on_frame_blend_only` then NcnnDenoiser, no TELEA before it).
- Do not feed still sparkle RGBA into video; video overlay RGB stays white (`VideoMap.rgb = None`).
- Do not use the on-disk grayscale diamond preview as blend RGB.
- Image remove stays reverse-blend only (no TELEA on stills).
- Video alpha stays one scale per shot; `force_alpha` bypasses seed lock and silhouette pick.
- Do not edit `src/lib.rs` or `README.md`.
- Do not add large video fixtures; `examples/quality_*` stay manual.

---

## File structure

No new source files. Behavior stays in the modules that already own it.

| File | Responsibility after this work |
|------|--------------------------------|
| `src/detect.rs` | Still-image locate: canonical independent insets, wide diagonal, top-K=5 gates, NCC `MIN_NCC=0.70` with Gate 2 3.75 when survival ≤ 0.85 |
| `src/video/alpha.rs` | Existing LS/bisection (`SCALE_MAX=1.05`) plus `pick_alpha_by_silhouette` (trial ceiling 1.12, hole reject 6.0 luma) |
| `src/video/frame.rs` | `remove_on_frame`: reverse-blend, opaque TELEA, conditional thin ring mix 0.40. `remove_on_frame_blend_only` unchanged |
| `src/video/pipeline.rs` | `seed_alpha_locked` then silhouette pick unless `force_alpha`; log `alpha_scale=… seed=…` |
| `src/video/telea.rs` | Unchanged helper; still used for opaque/ring masks only |
| `src/video/detect.rs`, `src/video/fdncnn.rs`, `src/blend.rs`, `src/cli.rs` | Do not modify |

---

### Task 1: Still-image detect (independent canonical, top-K, NCC Gate 2)

**Files:**
- Modify: `src/detect.rs` (`search` ~76–155, `gates_ok` ~157–173, `ncc_search` ~384–441, tests ~520–631)
- Test: `src/detect.rs` (`mod tests`) and existing `tests/api_remove.rs` (run only, no edit)

**Interfaces:**
- Consumes: `sparkle_template()`, `silhouette_edges`, `control_edges`, `gates_ok`, `ncc_search` as they exist today
- Produces: `match_watermark` still `fn match_watermark(width: u32, height: u32, data: &[u8]) -> Option<Match>`. Internal: `enum InsetWalk { Diagonal, Independent }`, `const TOP_K: usize = 5`, `const NCC_RELAX_SURVIVAL: f64 = 0.85`

- [ ] **Step 1: Write the failing canonical off-diagonal test**

Add this test at the end of `src/detect.rs` `mod tests`. It must call private `search` so NCC cannot hide a miss. `InsetWalk` will not exist yet — that is the intended compile failure.

```rust
fn paint_sparkle(data: &mut [u8], w: u32, x: u32, y: u32) {
    let tpl = crate::sparkle_template();
    for ty in 0..tpl.height {
        for tx in 0..tpl.width {
            let ti = ((ty * tpl.width + tx) * 4) as usize;
            let a = tpl.data[ti + 3] as f32 / 255.0;
            if a == 0.0 {
                continue;
            }
            let oi = (((y + ty) * w + (x + tx)) * 4) as usize;
            for c in 0..3 {
                data[oi + c] = (a * tpl.data[ti + c] as f32 + (1.0 - a) * data[oi + c] as f32)
                    .round() as u8;
            }
        }
    }
}

#[test]
fn canonical_independent_finds_off_diagonal_88_104() {
    // Both insets sit in [88, 104]; diagonal walk never lands on this pair.
    let w = 400u32;
    let h = 300u32;
    let s = crate::SPARKLE_SIZE;
    let x = w - 88 - s;
    let y = h - 104 - s;
    let mut data = vec![0x6a_u8; (w * h * 4) as usize];
    for i in (0..data.len()).step_by(4) {
        data[i + 3] = 255;
    }
    paint_sparkle(&mut data, w, x, y);

    let m = search(
        w,
        h,
        &data,
        88,
        104,
        MAX_SILHOUETTE_SURVIVAL_CANON,
        InsetWalk::Independent,
    )
    .expect("canonical independent search must find 88×104 without NCC");
    assert!((m.x as i32 - x as i32).abs() <= 4, "x={} want {x}", m.x);
    assert!((m.y as i32 - y as i32).abs() <= 4, "y={} want {y}", m.y);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib canonical_independent_finds_off_diagonal_88_104 -- --nocapture`

Expected: compile error `cannot find type InsetWalk` and/or `this function takes 6 arguments but 7 arguments were supplied`.

- [ ] **Step 3: Add `InsetWalk` and thread it through `search` (still diagonal-only)**

Above `search`, add:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InsetWalk {
    Diagonal,
    Independent,
}
```

Change `match_watermark` to:

```rust
search(
    width,
    height,
    data,
    88,
    104,
    MAX_SILHOUETTE_SURVIVAL_CANON,
    InsetWalk::Independent,
)
.or_else(|| {
    search(
        width,
        height,
        data,
        MIN_INSET,
        MAX_INSET,
        MAX_SILHOUETTE_SURVIVAL,
        InsetWalk::Diagonal,
    )
})
.or_else(|| ncc_search(width, height, data))
```

Add the `walk: InsetWalk` parameter to `search`. Keep the existing `for ins in min_inset..=max_inset { let x = …; let y = …; }` body for **both** walk modes for this step (Independent not implemented yet).

- [ ] **Step 4: Re-run the new test**

Run: `cargo test --lib canonical_independent_finds_off_diagonal_88_104 -- --nocapture`

Expected: FAIL assertion / `expect("canonical independent search must find 88×104 without NCC")` because diagonal never hits inset pair (88, 104).

- [ ] **Step 5: Implement independent inset walk**

Replace the single `for ins in min_inset..=max_inset` loop with:

```rust
let mut walk_xy = |mx: u32, my: u32, tpl: &WatermarkTemplate, s: u32| {
    if img_w < mx + s || img_h < my + s {
        return;
    }
    let x = img_w - mx - s;
    let y = img_h - my - s;
    // existing before/after/fraction update of best-* fields
};

for s in MIN_SIZE..=MAX_SIZE {
    // sized + ALPHA_SCALES as today
    for &alpha in &ALPHA_SCALES {
        let tpl = /* with_opacity as today */;
        match walk {
            InsetWalk::Diagonal => {
                for ins in min_inset..=max_inset {
                    walk_xy(ins, ins, &tpl, s);
                }
            }
            InsetWalk::Independent => {
                for my in min_inset..=max_inset {
                    for mx in min_inset..=max_inset {
                        walk_xy(mx, my, &tpl, s);
                    }
                }
            }
        }
    }
}
```

Keep the current single-best then `gates_ok` tail for this step (top-K is the next cycle).

- [ ] **Step 6: Run canonical test — expect PASS**

Run: `cargo test --lib canonical_independent_finds_off_diagonal_88_104 -- --nocapture`

Expected: PASS

- [ ] **Step 7: Write failing top-K unit test**

Add next to the other detect tests. `choose_match` / `SilhouetteCand` do not exist yet.

```rust
#[test]
fn top_k_skips_failing_lowest_survival() {
    let tpl = crate::sparkle_template();
    let loser = SilhouetteCand {
        x: 10,
        y: 10,
        size: tpl.width,
        survival: 0.50,
        after: 400.0,
        template: tpl.clone(),
    };
    let winner = SilhouetteCand {
        x: 40,
        y: 40,
        size: tpl.width,
        survival: 0.60,
        after: 10.0,
        template: tpl,
    };
    // Fake image large enough for control_edges on winner (x=40,y=40,s=48).
    let w = 200u32;
    let h = 200u32;
    let mut data = vec![80u8; (w * h * 4) as usize];
    for i in (0..data.len()).step_by(4) {
        data[i + 3] = 255;
    }
    // Make loser's Gate 2 explode: control_edges around (10,10) will sample
    // nearby grey; set after=400 so after/control >> 3.75 even with strong G1.
    let got = choose_match(vec![loser, winner.clone()], w, h, &data, 0.98)
        .expect("second candidate must pass after first fails gates");
    assert_eq!(got.x, 40);
    assert_eq!(got.y, 40);
    assert!((got.residual - 0.60).abs() < 1e-9);
}
```

If `control_edges` at (10,10) is 0 (margin), `gates_ok` treats `control <= 1e-6` as pass. Avoid that: use `x=40, y=40` for the loser too but that collides. Use loser at `(80, 80)` where grab() succeeds, and set `after` huge (`1.0e6`) so Gate 2 fails even if control is modest. Winner at `(40, 40)` with `after: 1.0`.

Adjusted loser: `x: 80, y: 80, survival: 0.50, after: 1.0e6`. Strong path requires `gate2 <= 3.75`; `1e6 / control` fails. Winner `after: 1.0` passes Gate 2.

- [ ] **Step 8: Run top-K test to verify it fails**

Run: `cargo test --lib top_k_skips_failing_lowest_survival -- --nocapture`

Expected: compile error `cannot find struct SilhouetteCand` / `cannot find function choose_match`.

- [ ] **Step 9: Implement top-K types and wire `search`**

```rust
const TOP_K: usize = 5;

struct SilhouetteCand {
    x: u32,
    y: u32,
    size: u32,
    survival: f64,
    after: f64,
    template: WatermarkTemplate,
}

fn cand_cmp(a: &SilhouetteCand, b: &SilhouetteCand, img_w: u32, img_h: u32) -> std::cmp::Ordering {
    match a.survival.partial_cmp(&b.survival) {
        Some(std::cmp::Ordering::Equal) | None => {
            let da = ((img_w - a.x - a.size) as i32 - (img_h - a.y - a.size) as i32).unsigned_abs();
            let db = ((img_w - b.x - b.size) as i32 - (img_h - b.y - b.size) as i32).unsigned_abs();
            da.cmp(&db).then_with(|| a.size.cmp(&b.size))
        }
        Some(o) => o,
    }
}

fn consider_cand(cands: &mut Vec<SilhouetteCand>, c: SilhouetteCand, img_w: u32, img_h: u32) {
    cands.push(c);
    cands.sort_by(|a, b| cand_cmp(a, b, img_w, img_h));
    if cands.len() > TOP_K {
        cands.pop();
    }
}

fn choose_match(
    mut cands: Vec<SilhouetteCand>,
    img_w: u32,
    img_h: u32,
    img: &[u8],
    max_survival: f64,
) -> Option<Match> {
    cands.sort_by(|a, b| cand_cmp(a, b, img_w, img_h));
    cands.truncate(TOP_K);
    for c in cands {
        let control = control_edges(img_w, img_h, img, &c.template, c.x, c.y);
        if gates_ok(c.survival, c.after, control, max_survival) {
            return Some(Match {
                x: c.x,
                y: c.y,
                width: c.size,
                height: c.size,
                residual: c.survival,
                template: c.template,
            });
        }
    }
    None
}
```

In `search`, replace `best_x/best_y/…` with `let mut cands: Vec<SilhouetteCand> = Vec::new();` and `consider_cand` on each finite fraction. Return `choose_match(cands, img_w, img_h, img, max_survival)`.

- [ ] **Step 10: Run top-K + canonical tests**

Run: `cargo test --lib top_k_skips_failing_lowest_survival canonical_independent_finds_off_diagonal_88_104 -- --nocapture`

Expected: PASS. If `top_k` fails because loser still passes `gates_ok`, raise `after` or pick a placement where `control_edges` is non-tiny (interior, not against the image edge).

- [ ] **Step 11: Write failing NCC Gate 2 tests**

```rust
#[test]
fn ncc_gate2_relaxes_when_survival_le_085() {
    // survival 0.80, gate2 3.50 → must pass (wood-ish, not rock 3.9)
    assert!(ncc_gates_ok(0.80, 3.50, 1.0));
}

#[test]
fn ncc_gate2_stays_strict_when_survival_gt_085() {
    assert!(!ncc_gates_ok(0.90, 3.50, 1.0));
}

#[test]
fn ncc_gate2_still_rejects_rock_impostor() {
    assert!(!ncc_gates_ok(0.80, 3.90, 1.0));
}
```

`ncc_gates_ok(survival, gate2, control)` is new. Encode `after = gate2 * control` so the helper can call the same math as production.

- [ ] **Step 12: Run NCC gate tests — expect compile fail or assertion fail**

Run: `cargo test --lib ncc_gate2 -- --nocapture`

Expected: `cannot find function ncc_gates_ok`.

- [ ] **Step 13: Implement NCC Gate 2 helper and use it in `ncc_search`**

```rust
const NCC_RELAX_SURVIVAL: f64 = 0.85;

fn ncc_gates_ok(survival: f64, gate2: f64, control: f64) -> bool {
    if !survival.is_finite() {
        return false;
    }
    if survival <= STRONG_SURVIVAL && gate2 <= STRONG_GATE2 {
        return true;
    }
    if survival >= NCC_SURVIVAL {
        return false;
    }
    let limit = if survival <= NCC_RELAX_SURVIVAL {
        STRONG_GATE2
    } else {
        MAX_SILHOUETTE_VS_CONTROL
    };
    control <= 1e-6 || gate2 <= limit
}
```

In `ncc_search`, after computing `survival` and `control`, replace `gates_ok(..., NCC_SURVIVAL - 1e-9)` with:

```rust
let gate2 = if control > 1e-6 { after / control } else { 0.0 };
if survival >= NCC_SURVIVAL || !ncc_gates_ok(survival, gate2, control) {
    return None;
}
```

Do **not** change `MIN_NCC` (`0.70`). Do **not** change canonical/wide `gates_ok`.

- [ ] **Step 14: Run all still-image tests**

Run:

```
cargo test --lib
cargo test --test api_remove
```

Expected: PASS, including existing fixture tests (`detects_sparkle_on_fixture`, gravel/wood/asphalt, `unmarked_busy_noise_is_not_a_match`, `detects_off_diagonal_sparkle_via_ncc`) and `api_remove`.

Add one more test in the same file before finishing (precision pair from the spec):

```rust
fn noise_canvas(w: u32, h: u32) -> Vec<u8> {
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let v = (((x.wrapping_mul(37)) ^ (y.wrapping_mul(91))) as u8).saturating_add(40);
            data[i] = v;
            data[i + 1] = v.saturating_sub(7);
            data[i + 2] = v.saturating_add(5);
            data[i + 3] = 255;
        }
    }
    data
}

#[test]
fn noise_without_overlay_is_not_found() {
    let data = noise_canvas(256, 256);
    assert!(match_watermark(256, 256, &data).is_none());
}

#[test]
fn noise_with_planted_sparkle_is_found() {
    let w = 400u32;
    let h = 300u32;
    let s = crate::SPARKLE_SIZE;
    let x = w - 96 - s;
    let y = h - 96 - s;
    let mut data = noise_canvas(w, h);
    paint_sparkle(&mut data, w, x, y);
    let m = match_watermark(w, h, &data).expect("planted sparkle on noise");
    assert!((m.x as i32 - x as i32).abs() <= 4);
    assert!((m.y as i32 - y as i32).abs() <= 4);
}
```

`noise_without_overlay` overlaps `unmarked_busy_noise_is_not_a_match`; keep both (256 canvas already exists — if identical, skip the duplicate and only add `noise_with_planted_sparkle_is_found`).

- [ ] **Step 15: Commit**

```bash
git add src/detect.rs
git commit -m "$(cat <<'EOF'
feat: independent canonical search, top-K gates, relaxed NCC Gate 2

Still-image detect walks inset X/Y independently in 88–104, keeps the
top-5 silhouette candidates, and allows Gate 2 3.75 on NCC hits with
survival ≤ 0.85. MIN_NCC stays 0.70.
EOF
)"
```

---

### Task 2: Video silhouette alpha pick

**Files:**
- Modify: `src/video/alpha.rs` (new `pick_alpha_by_silhouette` after `refine_alpha_bisection`; tests at end of `mod tests`)
- Do not modify `src/video/pipeline.rs` yet (Task 4 wires it)

**Interfaces:**
- Consumes: `remove_on_frame_blend_only`, `VideoDetection`, `VideoMap`, existing `luma_u8`, `dims_ok`
- Produces:

```rust
pub const PICK_SCALE_MIN: f32 = 0.78;
pub const PICK_SCALE_MAX: f32 = 1.12;
pub const HOLE_LUMA_THR: f32 = 6.0;

pub fn pick_alpha_by_silhouette(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
    seed: f32,
) -> f32
```

`estimate_alpha` / `refine_alpha_bisection` keep `SCALE_MAX = 1.05`. The 1.12 ceiling applies only to this discrete trial set.

Do not re-export `pick_alpha_by_silhouette` from `src/video/mod.rs`.

- [ ] **Step 1: Write failing tests in `src/video/alpha.rs` `mod tests`**

Reuse `diamond_map_8`, `blend_on_canvas`.

```rust
#[test]
fn pick_prefers_true_scale_over_under_seed() {
    let map = diamond_map_8();
    let img = blend_on_canvas(60, &map, 1.0, 32, 32, 8, 8);
    let det = VideoDetection {
        mark: MarkKind::Diamond,
        x: 8,
        y: 8,
        w: 8,
        h: 8,
        score: 1.0,
    };
    let picked = pick_alpha_by_silhouette(&[img], 32, 32, &det, &map, 0.90);
    assert!(
        (picked - 1.0).abs() <= 0.02,
        "expected ~1.0, got {picked}"
    );
}

#[test]
fn pick_discards_scale_that_digs_dark_hole() {
    let map = diamond_map_8();
    let img = blend_on_canvas(60, &map, 1.0, 32, 32, 8, 8);
    let det = VideoDetection {
        mark: MarkKind::Diamond,
        x: 8,
        y: 8,
        w: 8,
        h: 8,
        score: 1.0,
    };
    // 1.12 over-subtracts white-on-60 enough to drop hi-α luma > 6 below ring.
    assert!(
        scale_digs_hole(&img, 32, 32, &det, &map, 1.12),
        "1.12 should be classified as a hole on this synthetic"
    );
    assert!(!scale_digs_hole(&img, 32, 32, &det, &map, 1.0));
    let picked = pick_alpha_by_silhouette(&[img], 32, 32, &det, &map, 1.0);
    assert!(picked <= 1.05 + 1e-3, "must not pick hole scale 1.12, got {picked}");
}

#[test]
fn pick_returns_seed_when_every_trial_is_a_hole() {
    let map = diamond_map_8();
    // Already-dark ROI: any reverse-blend of white stays a hole vs a bright ring.
    // Build 32×32 at bg 8 with a bright 4px exterior (200) and a blended diamond
    // in the center so hi-α after any trial stays << ring - 6.
    let mut img = vec![8u8; 32 * 32 * 4];
    for i in 0..32 * 32 {
        img[i * 4 + 3] = 255;
    }
    for y in 0..32u32 {
        for x in 0..32u32 {
            let border = x < 2 || y < 2 || x >= 30 || y >= 30;
            if border {
                let o = ((y * 32 + x) * 4) as usize;
                img[o] = 200;
                img[o + 1] = 200;
                img[o + 2] = 200;
            }
        }
    }
    let det = VideoDetection {
        mark: MarkKind::Diamond,
        x: 8,
        y: 8,
        w: 8,
        h: 8,
        score: 1.0,
    };
    let seed = 0.83f32;
    let picked = pick_alpha_by_silhouette(&[img], 32, 32, &det, &map, seed);
    assert!(
        (picked - seed).abs() < 1e-5,
        "all-hole fallback must return seed {seed}, got {picked}"
    );
}
```

If the all-hole canvas is finicky, keep the test but implement `scale_digs_hole` first and, in `pick_alpha_by_silhouette` tests, stub by calling pick on a map whose every trial is rejected via a `#[cfg(test)]` hook **only if** the bright-ring construction fails — prefer the construction above; do not add a test-only override flag in production.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --features video --lib pick_prefers_true_scale_over_under_seed pick_discards_scale_that_digs_dark_hole pick_returns_seed_when_every_trial_is_a_hole -- --nocapture`

Expected: compile error `cannot find function pick_alpha_by_silhouette`.

- [ ] **Step 3: Implement pick + hole + ROI silhouette survival**

In `src/video/alpha.rs`:

```rust
pub const PICK_SCALE_MIN: f32 = 0.78;
pub const PICK_SCALE_MAX: f32 = 1.12;
pub const HOLE_LUMA_THR: f32 = 6.0;

pub fn pick_alpha_by_silhouette(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
    seed: f32,
) -> f32 {
    let seed = seed.clamp(PICK_SCALE_MIN, PICK_SCALE_MAX);
    let mut trials = [seed, 1.0, 1.05, 1.12];
    for t in trials.iter_mut() {
        *t = t.clamp(PICK_SCALE_MIN, PICK_SCALE_MAX);
    }
    trials.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut uniq: Vec<f32> = Vec::new();
    for t in trials {
        if uniq.last().map(|u| (t - *u).abs() < 1e-4).unwrap_or(false) {
            continue;
        }
        uniq.push(t);
    }

    let probes = probe_indices(frames.len());
    let mut best: Option<(f32, f32)> = None; // (survival, scale)
    for scale in uniq {
        let mut hole = false;
        let mut surv_sum = 0.0f32;
        let mut surv_n = 0.0f32;
        for &idx in &probes {
            let frame = &frames[idx];
            if scale_digs_hole(frame, width, height, det, map, scale) {
                hole = true;
                break;
            }
            let s = roi_silhouette_survival(frame, width, height, det, map, scale);
            if s.is_finite() {
                surv_sum += s;
                surv_n += 1.0;
            }
        }
        if hole || surv_n < 1.0 {
            continue;
        }
        let mean = surv_sum / surv_n;
        let take = match best {
            None => true,
            Some((bs, bscale)) => {
                mean < bs - 1e-6
                    || ((mean - bs).abs() <= 1e-6 && (scale - 1.0).abs() < (bscale - 1.0).abs())
            }
        };
        if take {
            best = Some((mean, scale));
        }
    }
    best.map(|(_, s)| s).unwrap_or(seed)
}

fn probe_indices(n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let samples = 5.min(n);
    (0..samples)
        .map(|i| {
            if samples == 1 {
                0
            } else {
                i * (n - 1) / (samples - 1)
            }
        })
        .collect()
}

pub(crate) fn scale_digs_hole(
    frame: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    scale: f32,
) -> bool {
    if !dims_ok(frame, w, h, det, map) {
        return false;
    }
    let mut copy = frame.to_vec();
    remove_on_frame_blend_only(&mut copy, w, h, det, map, scale);
    let (hi, lo) = hi_lo_luma(&copy, w, h, det, map);
    lo.is_finite() && hi.is_finite() && hi < lo - HOLE_LUMA_THR
}

fn hi_lo_luma(
    frame: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
) -> (f32, f32) {
    let mw = map.width as usize;
    let mh = map.height as usize;
    let stride = w as usize;
    let mut hi_sum = 0.0f32;
    let mut hi_n = 0.0f32;
    let mut lo_sum = 0.0f32;
    let mut lo_n = 0.0f32;
    for py in 0..mh {
        for px in 0..mw {
            let a = map.alpha[py * mw + px];
            let fx = det.x as usize + px;
            let fy = det.y as usize + py;
            if fx >= w as usize || fy >= h as usize {
                continue;
            }
            let o = (fy * stride + fx) * 4;
            let l = luma_u8(frame[o], frame[o + 1], frame[o + 2]);
            if a > 0.05 {
                hi_sum += l;
                hi_n += 1.0;
            } else if a < 0.02 {
                lo_sum += l;
                lo_n += 1.0;
            }
        }
    }
    let hi = if hi_n > 0.0 { hi_sum / hi_n } else { f32::NAN };
    let lo = if lo_n > 0.0 { lo_sum / lo_n } else { f32::NAN };
    (hi, lo)
}

fn roi_silhouette_survival(
    frame: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    scale: f32,
) -> f32 {
    let before = roi_edge_energy(frame, w, h, det, map);
    if !before.is_finite() || before < 1e-6 {
        return f32::INFINITY;
    }
    let mut copy = frame.to_vec();
    remove_on_frame_blend_only(&mut copy, w, h, det, map, scale);
    let after = roi_edge_energy(&copy, w, h, det, map);
    if !after.is_finite() {
        return f32::INFINITY;
    }
    after / before
}

fn roi_edge_energy(
    frame: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
) -> f32 {
    let mw = map.width as usize;
    let mh = map.height as usize;
    if mw < 3 || mh < 3 {
        return 0.0;
    }
    let stride = w as usize;
    let mut energy = 0.0f32;
    let mut weight = 0.0f32;
    for py in 1..mh - 1 {
        for px in 1..mw - 1 {
            let ax = map.alpha[py * mw + px + 1] - map.alpha[py * mw + px - 1];
            let ay = map.alpha[(py + 1) * mw + px] - map.alpha[(py - 1) * mw + px];
            let wt = (ax * ax + ay * ay).sqrt();
            if wt < 0.02 {
                continue;
            }
            let fx = det.x as usize + px;
            let fy = det.y as usize + py;
            if fx == 0 || fy == 0 || fx + 1 >= w as usize || fy + 1 >= h as usize {
                continue;
            }
            let i = (fy * stride + fx) * 4;
            let mut g = 0.0f32;
            for c in 0..3 {
                let gx = frame[i + 4 + c] as f32 - frame[i - 4 + c] as f32;
                let gy = frame[i + stride * 4 + c] as f32 - frame[i - stride * 4 + c] as f32;
                g += gx * gx + gy * gy;
            }
            energy += g * wt;
            weight += wt;
        }
    }
    if weight > 0.0 {
        energy / weight
    } else {
        0.0
    }
}
```

`scale_digs_hole` must be visible to tests in this module (`pub(crate)` is enough).

- [ ] **Step 4: Run the new alpha tests plus existing alpha tests**

Run: `cargo test --features video --lib alpha:: -- --nocapture`

Expected: PASS. If `pick_prefers_true_scale_over_under_seed` lands on `1.05` because 8×8 energy is noisy, relax the assert to `picked >= 0.98 && picked <= 1.05` **only after** printing the actual value; do not allow `1.12`. If `pick_returns_seed_when_every_trial_is_a_hole` does not discard all trials, strengthen the bright border / darker interior until `scale_digs_hole` is true for `{seed, 1.0, 1.05, 1.12}`.

- [ ] **Step 5: Commit**

```bash
git add src/video/alpha.rs
git commit -m "$(cat <<'EOF'
feat: pick video alpha by silhouette survival with hole reject

Trial {seed, 1.0, 1.05, 1.12} on probe frames; drop scales that pull
hi-α luma more than 6 below the low-α ring; keep one scale per shot.
EOF
)"
```

---

### Task 3: Non-FDnCNN residual (drop full-footprint TELEA)

**Files:**
- Modify: `src/video/frame.rs` (`remove_on_frame_with_options` ~123–155, `edge_ring_cleanup` ~158–281, constants ~15–60, tests ~608–835)
- Unchanged: `src/video/telea.rs`, `remove_on_frame_blend_only`

**Interfaces:**
- Consumes: `reverse_alpha_blend` return mask, `inpaint_telea`, `alpha_grad_mag`, `dilate_mask_3x3`, `gaussian_masked`, `reinject_exterior_grain`
- Produces: `remove_on_frame` still `fn remove_on_frame(rgba: &mut [u8], w: u32, h: u32, det: &VideoDetection, map: &VideoMap, alpha_scale: f32)`. Internal residual: opaque mask first; if silhouette survival > 0.40, `|∇α|` ring dilate 1 mixed at **0.40**; no `FOOT_HARD_MIX`, no full-footprint union.

- [ ] **Step 1: Write the failing anti-smear test**

Add to `src/video/frame.rs` `mod tests`. Helper `rgb_std` over a mask. Checker background so exterior stddev is high.

```rust
fn checker_canvas(cw: u32, ch: u32) -> Vec<u8> {
    let mut data = vec![0u8; (cw * ch * 4) as usize];
    for y in 0..ch {
        for x in 0..cw {
            let on = ((x / 2) + (y / 2)) % 2 == 0;
            let v = if on { 40u8 } else { 200u8 };
            let o = ((y * cw + x) * 4) as usize;
            data[o] = v;
            data[o + 1] = v;
            data[o + 2] = v;
            data[o + 3] = 255;
        }
    }
    data
}

fn rgb_std(rgba: &[u8], cw: u32, ox: u32, oy: u32, mw: usize, mh: usize, keep: impl Fn(usize) -> bool) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    let mut vals: Vec<f32> = Vec::new();
    for py in 0..mh {
        for px in 0..mw {
            let i = py * mw + px;
            if !keep(i) {
                continue;
            }
            let o = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
            let l = 0.299 * rgba[o] as f32 + 0.587 * rgba[o + 1] as f32 + 0.114 * rgba[o + 2] as f32;
            vals.push(l);
            sum += l;
            n += 1.0;
        }
    }
    if n < 4.0 {
        return 0.0;
    }
    let mean = sum / n;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    var.max(0.0).sqrt()
}

#[test]
fn remove_on_frame_preserves_checker_texture() {
    let map = {
        let mut alpha = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                let dx = (x as i32 - 3).abs() + (y as i32 - 3).abs();
                if dx <= 3 {
                    alpha[(y * 8 + x) as usize] = 0.35 * (1.0 - dx as f32 / 4.0);
                }
            }
        }
        VideoMap { width: 8, height: 8, alpha: alpha.clone(), rgb: None }
    };
    let cw = 16u32;
    let ch = 16u32;
    let ox = 4u32;
    let oy = 4u32;
    let mut canvas = checker_canvas(cw, ch);
    // Forward-blend white diamond onto checker (same as blend_forward).
    let patch = blend_forward(0, &map, 1.0); // overwritten per-pixel below
    let _ = patch;
    for py in 0..8usize {
        for px in 0..8usize {
            let a = map.alpha[py * 8 + px] as f64;
            if a <= 0.0 {
                continue;
            }
            let dst = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
            for c in 0..3 {
                canvas[dst + c] = (a * 255.0 + (1.0 - a) * canvas[dst + c] as f64).round() as u8;
            }
        }
    }
    let det = VideoDetection {
        mark: MarkKind::Diamond,
        x: ox,
        y: oy,
        w: 8,
        h: 8,
        score: 1.0,
    };
    remove_on_frame(&mut canvas, cw, ch, &det, &map, 1.0);
    let inner = rgb_std(&canvas, cw, ox, oy, 8, 8, |i| map.alpha[i] > 0.05);
    let outer = rgb_std(&canvas, cw, ox, oy, 8, 8, |i| map.alpha[i] < 0.02);
    assert!(
        outer > 20.0,
        "checker exterior should stay high-contrast, std={outer}"
    );
    assert!(
        inner >= 0.50 * outer,
        "footprint std {inner} collapsed vs exterior {outer} (smear)"
    );
}
```

Fix `blend_forward(0, …)` — do not use it for the checker; the per-pixel loop above is the blend. Delete the unused `patch` lines when implementing.

- [ ] **Step 2: Run anti-smear test on current code**

Run: `cargo test --features video --lib remove_on_frame_preserves_checker_texture -- --nocapture`

Expected: FAIL `footprint std … collapsed vs exterior` because current full-footprint TELEA + `FOOT_HARD_MIX=0.90` flattens the checker.

- [ ] **Step 3: Replace residual policy**

Constants — delete uses of `FOOT_ALPHA_THR`, `FOOT_DILATE`, `FOOT_HARD_MIX`, `RESID_STRENGTH=0.92`, `EDGE_DILATE=3`. Set:

```rust
const RING_SURVIVAL_THR: f64 = 0.40;
const RING_MIX: f32 = 0.40;
const RING_DILATE: usize = 1;
```

Keep `EDGE_ALPHA_EPS = 0.008`, `EDGE_GRAD_THR = 0.022`, `OPAQUE_CUTOFF = 0.95`. Keep `GRAIN_STRENGTH` but call grain only on the ring mask.

Change `remove_on_frame_with_options`:

```rust
let before_energy = roi_sil_energy(rgba, w, h, det, map);
let mask = reverse_alpha_blend(
    w,
    h,
    rgba,
    &tpl,
    det.x as i32,
    det.y as i32,
    OPAQUE_CUTOFF,
);
if do_edge_cleanup {
    residual_cleanup(rgba, w, h, det, map, &mask, before_energy);
}
```

Implement `roi_sil_energy` like Task 2 `roi_edge_energy` (duplicate locally in `frame.rs`; do not import from `alpha.rs`).

`residual_cleanup`:

1. Build `telea_mask: Vec<bool>` of size `mw*mh` where reverse-blend `mask[y*img_w + x] != 0` for pixels inside the ROI. If any, `inpaint_telea` on that ROI (radius 5 if `mw<=48` else 7), write back those pixels.
2. `after = roi_sil_energy(...)`. `survival = after / before_energy` (if `before_energy < 1e-6`, treat survival as 0.0 and skip the ring).
3. If `survival > 0.40`: ring = `α >= 0.008 && |∇α| > 0.022`, dilate **1**, `inpaint_telea` + `gaussian_masked` as today, mix with `RING_MIX` (no hard mix, no `a/ramp` boost). `reinject_exterior_grain` **only** using this ring as `mask`. Write back ring pixels.
4. If `survival <= 0.40`, stop after step 1.

Delete the full-footprint `foot` union and `FOOT_HARD_MIX` path.

Rename `edge_ring_cleanup` → `residual_cleanup` or keep the name but change the body; update the existing `edge_cleanup_reduces_synthetic_silhouette` test to call the new function. That test paints a dark ring then expects contrast to drop. With mix 0.40, change the bound from `0.55` to `0.80`:

```rust
assert!(
    after_contrast < before_contrast * 0.80,
    "edge cleanup should cut silhouette contrast: before={before_contrast} after={after_contrast}"
);
```

To force the ring, either call `residual_cleanup` with a low dummy `before_energy` so survival > 0.40, or paint the dark ring **before** measuring `before_energy` as the unpainted blended ROI. Preferred: snapshot energy on the blended-clean ROI, paint the dark ring, then call `residual_cleanup(..., before_energy)` so survival is high.

- [ ] **Step 4: Run frame tests**

Run: `cargo test --features video --lib video::frame:: -- --nocapture`

Expected: PASS including `synthetic_diamond_patch_recovers_background`, `map_to_template_*`, `edge_cleanup_reduces_synthetic_silhouette`, `remove_on_frame_preserves_checker_texture`.

- [ ] **Step 5: Commit**

```bash
git add src/video/frame.rs
git commit -m "$(cat <<'EOF'
fix: drop full-footprint TELEA; thin ring residual at mix 0.40

Non-FDnCNN remove keeps reverse-blend as recovery, inpaints only opaque
pixels, and optionally a dilate-1 |∇α| ring when silhouette survival
stays above 0.40.
EOF
)"
```

---

### Task 4: Wire pick into `remove_video` and full regression

**Files:**
- Modify: `src/video/pipeline.rs` (alpha block ~86–94, `seed_alpha_locked` stays, import line ~19)
- Test: add `resolve_alpha_scale` unit tests in `src/video/pipeline.rs` `mod tests` if a private helper is extracted; otherwise test `force_alpha` bypass via the helper below. Existing `remove_video_sample` stays (skips without ffmpeg/sample).

**Interfaces:**
- Consumes: `pick_alpha_by_silhouette(frames, width, height, det, map, seed) -> f32` from Task 2
- Produces: no public signature change. Log line exactly:

```text
alpha_scale={picked:.4} seed={seed:.4} region=({x},{y},{w},{h})
```

- [ ] **Step 1: Extract `resolve_alpha_scale` and write failing force-alpha test**

In `pipeline.rs` (not inside `remove_video` yet):

```rust
fn resolve_alpha_scale(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
    force_alpha: Option<f32>,
) -> (f32 /*picked*/, f32 /*seed*/) {
    if let Some(forced) = force_alpha {
        let s = forced.clamp(0.05, 2.0);
        return (s, s);
    }
    let seed = seed_alpha_locked(frames, width, height, det, map);
    let picked = seed; // Task 4 Step 3 replaces this with pick_alpha_by_silhouette
    (picked, seed)
}
```

Add a test in `pipeline.rs` `mod tests` (the module already exists around the `remove_video_sample` test). If tests are inside a `#[cfg(test)]` nested under the pipeline impl, put this next to `remove_video_sample`. Need a tiny diamond like alpha tests — duplicate an 8×8 map + `blend_on_canvas` locally in the test (do not import private alpha helpers).

```rust
#[test]
fn force_alpha_bypasses_silhouette_pick() {
    // Under-locked synthetic: pick would move seed toward 1.0, force must stick.
    let mut alpha = vec![0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let dx = (x as i32 - 3).abs() + (y as i32 - 3).abs();
            if dx <= 3 {
                alpha[(y * 8 + x) as usize] = 0.35 * (1.0 - dx as f32 / 4.0);
            }
        }
    }
    let map = VideoMap {
        width: 8,
        height: 8,
        alpha,
        rgb: None,
    };
    let mut img = vec![60u8; 32 * 32 * 4];
    for i in 0..32 * 32 {
        img[i * 4 + 3] = 255;
    }
    for py in 0..8usize {
        for px in 0..8usize {
            let a = (map.alpha[py * 8 + px] * 1.0).clamp(0.0, 1.0) as f64;
            let o = ((8 + py) * 32 + (8 + px)) * 4;
            img[o] = (a * 255.0 + (1.0 - a) * 60.0).round() as u8;
            img[o + 1] = img[o];
            img[o + 2] = img[o];
        }
    }
    let det = VideoDetection {
        mark: MarkKind::Diamond,
        x: 8,
        y: 8,
        w: 8,
        h: 8,
        score: 1.0,
    };
    let (picked, seed) = resolve_alpha_scale(&[img], 32, 32, &det, &map, Some(0.55));
    assert!((picked - 0.55).abs() < 1e-5, "force must win, got {picked}");
    assert!((seed - 0.55).abs() < 1e-5);
}
```

At Step 1, `resolve_alpha_scale` does not exist → compile fail.

- [ ] **Step 2: Run the force-alpha test**

Run: `cargo test --features video --lib force_alpha_bypasses_silhouette_pick -- --nocapture`

Expected: compile error, then after adding the stub helper, PASS even before wiring pick (forced branch returns early). That is OK.

- [ ] **Step 3: Wire pick and logging in `remove_video`**

Import:

```rust
use super::alpha::{pick_alpha_by_silhouette, refine_alpha_bisection};
```

Replace the alpha block:

```rust
let (alpha_scale, seed) = resolve_alpha_scale(
    &frames,
    probe.width,
    probe.height,
    &det,
    &map,
    opts.force_alpha,
);
eprintln!(
    "alpha_scale={:.4} seed={:.4} region=({},{},{},{})",
    alpha_scale, seed, det.x, det.y, det.w, det.h
);
```

Fill the non-force branch:

```rust
let seed = seed_alpha_locked(frames, width, height, det, map);
let picked = pick_alpha_by_silhouette(frames, width, height, det, map, seed);
(picked, seed)
```

Update the comment above Phase 2: residual is now opaque + optional thin ring inside `remove_on_frame`; FDnCNN still blend-only then denoise.

- [ ] **Step 4: Add adaptive-pick test (not force)**

```rust
#[test]
fn resolve_picks_near_true_scale_when_seed_under_locked() {
    // Same canvas as force test, no force_alpha.
    // ... rebuild map/img/det as in force_alpha_bypasses_silhouette_pick ...
    let (picked, seed) = resolve_alpha_scale(&[img], 32, 32, &det, &map, None);
    assert!(seed <= 1.0);
    assert!(
        picked >= 0.96 && picked <= 1.05,
        "pick should lift under-seed toward 1.0, seed={seed} picked={picked}"
    );
}
```

Duplicate the 8×8 canvas setup in this test (no shared “similar to” helper required, but a local `fn tiny_diamond_frame() -> (Vec<u8>, VideoMap, VideoDetection)` in the same `mod tests` is allowed to keep both tests DRY **inside this file**).

- [ ] **Step 5: Full regression**

Run, in order, stop on first failure:

```
cargo test
cargo test --features video
cargo test --features video-fdncnn,system-ffmpeg
```

Expected: all PASS. `remove_video_sample` may skip if ffmpeg/sample missing — that is existing behavior, not a failure.

If FDnCNN tests fail because they assumed TELEA-before-denoise: they must not; production FDnCNN path still calls `remove_on_frame_blend_only`. Do not “fix” by running residual cleanup on the FDnCNN path.

- [ ] **Step 6: Commit**

```bash
git add src/video/pipeline.rs
git commit -m "$(cat <<'EOF'
feat: lock video alpha with silhouette pick unless force_alpha

remove_video seeds one scale, then picks among {seed, 1.0, 1.05, 1.12}
with hole reject. force_alpha still bypasses adaptive lock.
EOF
)"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Canonical independent X/Y in 88–104, Gate 1 0.98 | Task 1 |
| Wide diagonal 40–116, Gate 1 0.80 | Task 1 (walk mode Diagonal) |
| Top-K=5, first passing gates | Task 1 |
| `MIN_NCC=0.70` unchanged | Task 1 (explicit non-change) |
| NCC Gate 2 3.75 when survival ≤ 0.85; rock 3.9 fails | Task 1 `ncc_gates_ok` |
| Image remove unchanged | no task edits `src/blend.rs` / `remove_gemini_sparkle` |
| Video detect unchanged | no task edits `src/video/detect.rs` |
| Silhouette pick `{seed,1.0,1.05,1.12}`, clamp `[0.78,1.12]`, hole 6.0, all-hole → seed | Task 2 |
| `SCALE_MAX=1.05` on LS/bisection only | Task 2 |
| `force_alpha` bypass | Task 4 |
| Log `alpha_scale=… seed=… region=…` | Task 4 |
| Opaque fill + ring if survival > 0.40, dilate 1, mix 0.40, no FOOT_HARD_MIX | Task 3 |
| FDnCNN blend-only then denoise | Task 3/4 non-change |
| No shared NCC extract, no lib.rs/README, no CLI flags | Global constraints |
| Existing still fixtures + new canonical/top-K/noise tests | Task 1 |
| Anti-smear checker stddev ≥ 50% exterior | Task 3 |
| `cargo test` / `video` / `video-fdncnn` | Task 4 |

No placeholders left. Names: `pick_alpha_by_silhouette`, `resolve_alpha_scale`, `InsetWalk`, `SilhouetteCand`, `choose_match`, `ncc_gates_ok`, `RING_MIX=0.40`, `RING_SURVIVAL_THR=0.40` are used consistently across tasks.
