//! Per-frame adaptive alpha intensity (estimate + bisection feedback).
//!
//! The embedded diamond alpha map is already near operating opacity
//! (≈ 0.97). Adaptive scale therefore stays close to 1.0: a raw LS fit
//! against ROI border luma tends to overestimate on textured video, so we
//! damp toward unity and score residue against an exterior ring background.

use super::detect::VideoDetection;
use super::frame::remove_on_frame_blend_only;
use super::maps::VideoMap;

/// Cap absolute change vs the previous frame's applied scale.
pub const FRAME_ALPHA_CAP: f32 = 0.05;

/// Minimum map alpha to include in LS / residual scoring.
const ALPHA_EPS: f32 = 0.02;

/// Soft clamp for intensity scales.
///
/// Embedded maps peak ≈ 0.345. Nominal scale ≈ 0.96 matches operating
/// opacity and avoids the dark diamond ghost from scale≈1.0 over-subtraction.
const SCALE_MIN: f32 = 0.78;
const SCALE_MAX: f32 = 1.05;

/// Nominal operating scale for the embedded map.
const SCALE_NOMINAL: f32 = 0.96;

/// Blend weight for the LS estimate vs [`SCALE_NOMINAL`] (rest → nominal).
const ESTIMATE_WEIGHT: f32 = 0.08;

/// Estimate a global intensity scale for this frame's ROI.
///
/// Fits `observed ≈ s·α·logo + (1 − s·α)·bg` in luma over high-α pixels,
/// using a local background from low-α / border samples, then damps toward
/// [`SCALE_NOMINAL`] so textured BR content cannot inflate scale.
pub fn estimate_alpha(frame: &[u8], w: u32, h: u32, det: &VideoDetection, map: &VideoMap) -> f32 {
    if !dims_ok(frame, w, h, det, map) {
        return SCALE_NOMINAL;
    }
    let bg = local_background_luma(frame, w, h, det, map);
    let mut num = 0.0f64;
    let mut den = 0.0f64;

    let mw = map.width as usize;
    let mh = map.height as usize;
    let stride = w as usize;

    for py in 0..mh {
        for px in 0..mw {
            let ai = py * mw + px;
            let a = map.alpha[ai] as f64;
            if a < ALPHA_EPS as f64 {
                continue;
            }
            let fx = det.x as usize + px;
            let fy = det.y as usize + py;
            if fx >= w as usize || fy >= h as usize {
                continue;
            }
            let oi = (fy * stride + fx) * 4;
            let obs = luma_u8(frame[oi], frame[oi + 1], frame[oi + 2]) as f64;
            let (lr, lg, lb) = logo_rgb(map, ai);
            let logo = luma_u8(lr, lg, lb) as f64;
            let diff = logo - bg;
            if diff.abs() < 1.0 {
                continue;
            }
            let coeff = a * diff;
            num += coeff * (obs - bg);
            den += coeff * coeff;
        }
    }

    if den < 1e-6 {
        return SCALE_NOMINAL;
    }
    let raw = ((num / den) as f32).clamp(SCALE_MIN, SCALE_MAX);
    // Damp toward nominal — map peak α already matches operating opacity.
    (ESTIMATE_WEIGHT * raw + (1.0 - ESTIMATE_WEIGHT) * SCALE_NOMINAL).clamp(SCALE_MIN, SCALE_MAX)
}

/// Refine intensity via residue feedback, then cap vs previous.
///
/// Starts from [`estimate_alpha`]. Each round applies a trial reverse-blend on
/// an ROI scratch buffer and measures alpha-weighted luma bias vs an exterior
/// ring background. Positive bias → increase scale; negative → decrease.
/// When `previous` is `Some`, the result is clamped to
/// `previous ± FRAME_ALPHA_CAP`.
pub fn refine_alpha_bisection(
    frame: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    previous: Option<f32>,
) -> f32 {
    if !dims_ok(frame, w, h, det, map) {
        return previous
            .unwrap_or(SCALE_NOMINAL)
            .clamp(SCALE_MIN, SCALE_MAX);
    }

    let mut s = estimate_alpha(frame, w, h, det, map);
    // Prefer previous as soft seed when available (temporal continuity).
    if let Some(prev) = previous {
        s = 0.5 * s + 0.5 * prev;
    }
    s = s.clamp(SCALE_MIN, SCALE_MAX);

    // Residue feedback is conservative: the exterior ring is often darker than
    // the true under-mark background on textured BR content, so chasing
    // alpha-weighted (luma − ring) → 0 systematically over-scales.
    // Only correct clear over-subtraction; allow a tiny bump for strong residue.
    let ring_bg = exterior_ring_luma(frame, w, h, det, map);
    let bias = residual_bias(frame, w, h, det, map, s, ring_bg);
    if bias < -3.0 {
        s = (s - 0.04).max(SCALE_MIN);
    } else if bias > 6.0 {
        s = (s + 0.03).min(SCALE_MAX);
    }

    if let Some(prev) = previous {
        s = s.clamp(prev - FRAME_ALPHA_CAP, prev + FRAME_ALPHA_CAP);
    }
    s.clamp(SCALE_MIN, SCALE_MAX)
}

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

fn hi_lo_luma(frame: &[u8], w: u32, h: u32, det: &VideoDetection, map: &VideoMap) -> (f32, f32) {
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

fn roi_edge_energy(frame: &[u8], w: u32, h: u32, det: &VideoDetection, map: &VideoMap) -> f32 {
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

fn dims_ok(frame: &[u8], w: u32, h: u32, det: &VideoDetection, map: &VideoMap) -> bool {
    if w == 0 || h == 0 || map.width == 0 || map.height == 0 {
        return false;
    }
    let expected = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if frame.len() != expected {
        return false;
    }
    if map.alpha.len() != (map.width as usize) * (map.height as usize) {
        return false;
    }
    if det.w != map.width || det.h != map.height {
        // Still allow if ROI fits; map size drives the blend.
    }
    det.x.saturating_add(map.width) <= w && det.y.saturating_add(map.height) <= h
}

fn logo_rgb(map: &VideoMap, i: usize) -> (u8, u8, u8) {
    match &map.rgb {
        Some(rgb) if rgb.len() >= (i + 1) * 3 => {
            let o = i * 3;
            (rgb[o], rgb[o + 1], rgb[o + 2])
        }
        _ => (255, 255, 255),
    }
}

fn luma_u8(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Background luma: median of low-α pixels inside the ROI, falling back to a
/// 4px ring outside the box when the interior is fully opaque.
fn local_background_luma(
    frame: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
) -> f64 {
    let mut samples: Vec<f32> = Vec::new();
    let mw = map.width as usize;
    let mh = map.height as usize;
    let stride = w as usize;

    for py in 0..mh {
        for px in 0..mw {
            if map.alpha[py * mw + px] >= ALPHA_EPS {
                continue;
            }
            let fx = det.x as usize + px;
            let fy = det.y as usize + py;
            if fx >= w as usize || fy >= h as usize {
                continue;
            }
            let oi = (fy * stride + fx) * 4;
            samples.push(luma_u8(frame[oi], frame[oi + 1], frame[oi + 2]));
        }
    }

    if samples.len() < 8 {
        samples.extend(exterior_ring_samples(frame, w, h, det, map));
    }

    median_f32(&mut samples).unwrap_or(128.0) as f64
}

fn exterior_ring_luma(frame: &[u8], w: u32, h: u32, det: &VideoDetection, map: &VideoMap) -> f32 {
    let mut samples = exterior_ring_samples(frame, w, h, det, map);
    median_f32(&mut samples).unwrap_or(128.0)
}

fn exterior_ring_samples(
    frame: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
) -> Vec<f32> {
    let mut samples = Vec::new();
    let stride = w as usize;
    let x0 = det.x.saturating_sub(4);
    let y0 = det.y.saturating_sub(4);
    let x1 = (det.x + map.width + 4).min(w);
    let y1 = (det.y + map.height + 4).min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            let inside =
                x >= det.x && y >= det.y && x < det.x + map.width && y < det.y + map.height;
            if inside {
                continue;
            }
            let oi = (y as usize * stride + x as usize) * 4;
            samples.push(luma_u8(frame[oi], frame[oi + 1], frame[oi + 2]));
        }
    }
    samples
}

fn median_f32(samples: &mut [f32]) -> Option<f32> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(samples[samples.len() / 2])
}

/// Alpha-weighted mean of `(luma − ring_bg)` after a trial remove.
/// Positive ⇒ watermark residue (under-removed); negative ⇒ over-subtraction.
///
/// Preferring the exterior ring over hi-α vs lo-α inside the ROI avoids
/// chasing natural texture contrast to zero (which over-scales).
fn residual_bias(
    frame: &[u8],
    w: u32,
    _h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    scale: f32,
    ring_bg: f32,
) -> f32 {
    let mw = map.width as usize;
    let mh = map.height as usize;
    let stride = w as usize;

    let mut mini = vec![0u8; mw * mh * 4];
    for py in 0..mh {
        for px in 0..mw {
            let fx = det.x as usize + px;
            let fy = det.y as usize + py;
            let src = (fy * stride + fx) * 4;
            let dst = (py * mw + px) * 4;
            mini[dst..dst + 4].copy_from_slice(&frame[src..src + 4]);
        }
    }
    let mini_det = VideoDetection {
        mark: det.mark,
        x: 0,
        y: 0,
        w: map.width,
        h: map.height,
        score: det.score,
    };
    remove_on_frame_blend_only(&mut mini, map.width, map.height, &mini_det, map, scale);

    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..mw * mh {
        let a = map.alpha[i];
        if a < ALPHA_EPS {
            continue;
        }
        let o = i * 4;
        let l = luma_u8(mini[o], mini[o + 1], mini[o + 2]);
        num += a * (l - ring_bg);
        den += a;
    }
    if den < 1e-3 {
        return 0.0;
    }
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::{MarkKind, VideoDetection, VideoMap};

    fn diamond_map_8() -> VideoMap {
        let mut alpha = vec![0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                let dx = (x as i32 - 3).abs() + (y as i32 - 3).abs();
                if dx <= 3 {
                    alpha[(y * 8 + x) as usize] = 0.35 * (1.0 - dx as f32 / 4.0);
                }
            }
        }
        VideoMap {
            width: 8,
            height: 8,
            alpha,
            rgb: None,
        }
    }

    fn blend_on_canvas(
        bg: u8,
        map: &VideoMap,
        scale: f32,
        canvas_w: u32,
        canvas_h: u32,
        x: u32,
        y: u32,
    ) -> Vec<u8> {
        let mut img = vec![bg; (canvas_w as usize) * (canvas_h as usize) * 4];
        // set alpha channel
        for i in 0..(canvas_w * canvas_h) as usize {
            img[i * 4 + 3] = 255;
        }
        let mw = map.width as usize;
        let mh = map.height as usize;
        let stride = canvas_w as usize;
        for py in 0..mh {
            for px in 0..mw {
                let a = (map.alpha[py * mw + px] * scale).clamp(0.0, 1.0) as f64;
                let o = ((y as usize + py) * stride + (x as usize + px)) * 4;
                let (lr, lg, lb) = (255.0, 255.0, 255.0);
                img[o] = (a * lr + (1.0 - a) * bg as f64).round() as u8;
                img[o + 1] = (a * lg + (1.0 - a) * bg as f64).round() as u8;
                img[o + 2] = (a * lb + (1.0 - a) * bg as f64).round() as u8;
            }
        }
        img
    }

    #[test]
    fn estimate_alpha_near_true_scale() {
        let map = diamond_map_8();
        let true_s = 1.2f32;
        let img = blend_on_canvas(60, &map, true_s, 32, 32, 8, 8);
        let det = VideoDetection {
            mark: MarkKind::Diamond,
            x: 8,
            y: 8,
            w: 8,
            h: 8,
            score: 1.0,
        };
        let est = estimate_alpha(&img, 32, 32, &det, &map);
        // Damped toward nominal: raw≈1.2 → still above nominal, below true.
        assert!(
            est >= SCALE_NOMINAL - 0.01 && est <= 1.05,
            "estimate {est} should sit between nominal {SCALE_NOMINAL} and true {true_s}"
        );
        assert!(
            (est - SCALE_NOMINAL).abs() < 0.15,
            "estimate {est} should stay near nominal {SCALE_NOMINAL}"
        );
    }

    #[test]
    fn refine_caps_vs_previous() {
        let map = diamond_map_8();
        let img = blend_on_canvas(60, &map, 1.5, 32, 32, 8, 8);
        let det = VideoDetection {
            mark: MarkKind::Diamond,
            x: 8,
            y: 8,
            w: 8,
            h: 8,
            score: 1.0,
        };
        let prev = 1.0f32;
        let refined = refine_alpha_bisection(&img, 32, 32, &det, &map, Some(prev));
        assert!(
            (refined - prev).abs() <= FRAME_ALPHA_CAP + 1e-5,
            "refined {refined} drifted > ±{} from {prev}",
            FRAME_ALPHA_CAP
        );
    }

    #[test]
    fn bisection_monotonic_residue_direction() {
        // Under-scaled trial should show positive residual bias.
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
        let ring = exterior_ring_luma(&img, 32, 32, &det, &map);
        let under = residual_bias(&img, 32, 32, &det, &map, 0.4, ring);
        let over = residual_bias(&img, 32, 32, &det, &map, 1.15, ring);
        assert!(under > 0.0, "under-removal bias should be +, got {under}");
        assert!(over < 0.0, "over-removal bias should be -, got {over}");
    }

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
        assert!((picked - 1.0).abs() <= 0.02, "expected ~1.0, got {picked}");
    }

    #[test]
    fn stock_diamond_map_8_does_not_hole_at_pick_ceiling() {
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
        // Production HOLE_LUMA_THR=6.0 and trial 1.12 stay. Peak α≈0.35 on this
        // 8×8 map only drops mean hi-α luma ~4.56 below the α<0.02 ring at 1.12
        // (white-on-60 reverse-blend), so the ceiling does not trip here.
        let mut after = img.clone();
        remove_on_frame_blend_only(&mut after, 32, 32, &det, &map, 1.12);
        let (hi, lo) = hi_lo_luma(&after, 32, 32, &det, &map);
        let drop = lo - hi;
        assert!(
            drop > 4.0 && drop < HOLE_LUMA_THR,
            "stock peak≈0.35 at 1.12 should drop ~4.56 luma, got hi={hi} lo={lo} drop={drop}"
        );
        assert!(
            !scale_digs_hole(&img, 32, 32, &det, &map, 1.12),
            "1.12 must not hole-reject unscaled diamond_map_8 (drop={drop} < {HOLE_LUMA_THR})"
        );
    }

    #[test]
    fn pick_discards_scale_that_digs_dark_hole() {
        let mut map = diamond_map_8();
        // Peak 0.35 on this 8×8 diamond only drops mean hi-α luma 4.56 at 1.12
        // (measured), under HOLE_LUMA_THR. Lift peak so 1.12 holes and 1.0 does not.
        for a in &mut map.alpha {
            *a = (*a * 1.5).min(1.0);
        }
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
        assert!(
            picked <= 1.05 + 1e-3,
            "must not pick hole scale 1.12, got {picked}"
        );
    }

    #[test]
    fn pick_returns_seed_when_every_trial_is_a_hole() {
        let map = diamond_map_8();
        // Bright low-α ring *inside* the 8×8 map (not the image border outside
        // the ROI). Dark hi-α interior: any reverse-blend of white stays >6
        // luma below that in-map ring, so hole reject — not surv_n<1 — discards
        // every trial.
        let mut img = vec![40u8; 32 * 32 * 4];
        for i in 0..32 * 32 {
            img[i * 4 + 3] = 255;
        }
        for py in 0..8usize {
            for px in 0..8usize {
                let a = map.alpha[py * 8 + px];
                let o = ((8 + py) * 32 + (8 + px)) * 4;
                if a < 0.02 {
                    img[o] = 200;
                    img[o + 1] = 200;
                    img[o + 2] = 200;
                } else {
                    img[o] = 8;
                    img[o + 1] = 8;
                    img[o + 2] = 8;
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
        for scale in [seed, 1.0, 1.05, 1.12] {
            assert!(
                scale_digs_hole(&img, 32, 32, &det, &map, scale),
                "trial {scale} must hole-reject against the in-map low-α ring"
            );
            let surv = roi_silhouette_survival(&img, 32, 32, &det, &map, scale);
            assert!(
                surv.is_finite(),
                "trial {scale} must have silhouette energy (surv={surv}); discard via hole, not surv_n<1"
            );
        }
        let picked = pick_alpha_by_silhouette(&[img], 32, 32, &det, &map, seed);
        assert!(
            (picked - seed).abs() < 1e-5,
            "all-hole fallback must return seed {seed}, got {picked}"
        );
    }
}
