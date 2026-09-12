//! Per-frame reverse-blend using a VideoMap + intensity scale.
//!
//! After reverse-blend, residual fill is opaque-unrecoverable pixels first;
//! a thin high-`|∇α|` ring is mixed in only when silhouette survival stays
//! high. No full-footprint TELEA. Classical only.

use crate::blend::reverse_alpha_blend;
use crate::WatermarkTemplate;

use super::detect::VideoDetection;
use super::maps::VideoMap;
use crate::telea::inpaint_telea;

/// Same opaque cutoff as the still-image remove path.
const OPAQUE_CUTOFF: f64 = 0.95;
/// Minimum map alpha to include in the edge-ring mask.
const EDGE_ALPHA_EPS: f32 = 0.008;

/// `|∇α|` threshold for the silhouette ring (central-difference magnitude).
const EDGE_GRAD_THR: f32 = 0.022;

/// Run the thin ring only when reverse-blend leaves this much silhouette energy.
const RING_SURVIVAL_THR: f64 = 0.40;
/// Mix of TELEA fill vs reverse-blend on the ring (no hard mix / α ramp).
const RING_MIX: f32 = 0.40;
/// Binary dilate iterations on the `|∇α|` ring.
const RING_DILATE: usize = 1;

/// Mild Gaussian on the ring after TELEA (radius / sigma).
const EDGE_RING_GAUSS_RADIUS: i32 = 2;
const EDGE_RING_GAUSS_SIGMA: f32 = 1.4;

/// Exterior HF grain reinjection strength inside the footprint.
const GRAIN_STRENGTH: f32 = 0.85;
const GRAIN_BLUR_SIGMA: f32 = 1.1;
const GRAIN_GUIDE_SIGMA: f32 = 1.8;

/// Convert a [`VideoMap`] into a [`WatermarkTemplate`], scaling alpha by `alpha_scale`.
///
/// RGB comes from the map companion when present; otherwise pure white (255) —
/// required for Gemini diamond chrome (grayscale preview bins must not be used).
/// Alpha bytes are `round(clamp(map.alpha * alpha_scale, 0, 1) * 255)`.
pub fn map_to_template(map: &VideoMap, alpha_scale: f32) -> WatermarkTemplate {
    let n = (map.width as usize).saturating_mul(map.height as usize);
    debug_assert_eq!(map.alpha.len(), n);
    let mut data = vec![0u8; n * 4];
    let scale = alpha_scale.max(0.0);
    for i in 0..n {
        let a = (map.alpha[i] * scale).clamp(0.0, 1.0);
        let (r, g, b) = match &map.rgb {
            Some(rgb) if rgb.len() >= n * 3 => {
                let o = i * 3;
                (rgb[o], rgb[o + 1], rgb[o + 2])
            }
            _ => (255, 255, 255),
        };
        let o = i * 4;
        data[o] = r;
        data[o + 1] = g;
        data[o + 2] = b;
        data[o + 3] = (a * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    WatermarkTemplate {
        width: map.width,
        height: map.height,
        data,
    }
}

/// Reverse-blend the watermark ROI on one RGBA frame, then residual cleanup
/// (opaque TELEA; optional thin `|∇α|` ring).
///
/// Scales `map.alpha` by `alpha_scale`, builds a template (logo RGB or white),
/// and calls [`reverse_alpha_blend`] at `(det.x, det.y)`.
pub fn remove_on_frame(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    alpha_scale: f32,
) {
    remove_on_frame_with_options(rgba, w, h, det, map, alpha_scale, true);
}

/// Reverse-blend only (no edge cleanup). Used by adaptive alpha trials so
/// feedback measures pure blend residue.
pub(crate) fn remove_on_frame_blend_only(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    alpha_scale: f32,
) {
    remove_on_frame_with_options(rgba, w, h, det, map, alpha_scale, false);
}

fn remove_on_frame_with_options(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    alpha_scale: f32,
    do_edge_cleanup: bool,
) {
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4));
    assert_eq!(Some(rgba.len()), expected, "rgba length must be w*h*4");
    if map.width == 0 || map.height == 0 {
        return;
    }
    if map.alpha.len() != (map.width as usize) * (map.height as usize) {
        return;
    }
    let tpl = map_to_template(map, alpha_scale);
    let before_energy = roi_sil_energy(rgba, w, h, det, map);
    let mask = reverse_alpha_blend(w, h, rgba, &tpl, det.x as i32, det.y as i32, OPAQUE_CUTOFF);
    if do_edge_cleanup {
        residual_cleanup(rgba, w, h, det, map, &mask, before_energy);
    }
}

/// `|∇α|`-weighted RGB edge energy inside the detection ROI.
/// Duplicated from `alpha::roi_edge_energy`; not shared across modules.
fn roi_sil_energy(frame: &[u8], w: u32, h: u32, det: &VideoDetection, map: &VideoMap) -> f32 {
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

/// Soften leftover mark after reverse-blend: opaque fill, then optional ring.
///
/// 1. TELEA on reverse-blend opaque mask pixels (`α_template >= 0.95`).
/// 2. If silhouette survival > 0.40, TELEA a dilate-1 `|∇α|` ring at mix 0.40.
/// 3. Grain only on that ring. No full-footprint union.
fn residual_cleanup(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
    mask: &[u8],
    before_energy: f32,
) {
    let mw = map.width as usize;
    let mh = map.height as usize;
    if mw == 0 || mh == 0 {
        return;
    }
    let img_w = w as usize;
    let mut roi = vec![0f32; mw * mh * 3];
    copy_rgba_to_roi(rgba, w, h, det, mw, mh, &mut roi);

    let mut telea_mask = vec![false; mw * mh];
    if mask.len() == img_w.saturating_mul(h as usize) {
        for py in 0..mh {
            for px in 0..mw {
                let fx = det.x as usize + px;
                let fy = det.y as usize + py;
                if fx >= w as usize || fy >= h as usize {
                    continue;
                }
                if mask[fy * img_w + fx] != 0 {
                    telea_mask[py * mw + px] = true;
                }
            }
        }
    }
    if telea_mask.iter().any(|&m| m) {
        let radius = if mw <= 48 { 5 } else { 7 };
        inpaint_telea(&mut roi, &telea_mask, mw, mh, radius);
        write_roi_mask_to_rgba(rgba, w, h, det, mw, mh, &roi, &telea_mask);
    }

    let survival = if before_energy < 1e-6 {
        0.0
    } else {
        let after = roi_sil_energy(rgba, w, h, det, map);
        after / before_energy
    };
    if (survival as f64) <= RING_SURVIVAL_THR {
        return;
    }

    let grad = alpha_grad_mag(&map.alpha, mw, mh);
    let mut ring = vec![false; mw * mh];
    for i in 0..mw * mh {
        if map.alpha[i] >= EDGE_ALPHA_EPS && grad[i] > EDGE_GRAD_THR {
            ring[i] = true;
        }
    }
    for _ in 0..RING_DILATE {
        dilate_mask_3x3(&mut ring, mw, mh);
    }
    if !ring.iter().any(|&m| m) {
        return;
    }

    copy_rgba_to_roi(rgba, w, h, det, mw, mh, &mut roi);
    let blended = roi.clone();
    let radius = if mw <= 48 { 5 } else { 7 };
    inpaint_telea(&mut roi, &ring, mw, mh, radius);
    gaussian_masked(
        &mut roi,
        &ring,
        mw,
        mh,
        EDGE_RING_GAUSS_RADIUS,
        EDGE_RING_GAUSS_SIGMA,
    );

    let mix = RING_MIX.clamp(0.0, 1.0);
    for i in 0..mw * mh {
        if !ring[i] {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            roi[o + c] = roi[o + c] * mix + blended[o + c] * (1.0 - mix);
        }
    }

    reinject_exterior_grain(&mut roi, &blended, &ring, &map.alpha, mw, mh);
    write_roi_mask_to_rgba(rgba, w, h, det, mw, mh, &roi, &ring);
}

fn copy_rgba_to_roi(
    rgba: &[u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    mw: usize,
    mh: usize,
    roi: &mut [f32],
) {
    let stride = w as usize;
    for py in 0..mh {
        for px in 0..mw {
            let fx = det.x as usize + px;
            let fy = det.y as usize + py;
            if fx >= w as usize || fy >= h as usize {
                continue;
            }
            let oi = (fy * stride + fx) * 4;
            let ro = (py * mw + px) * 3;
            roi[ro] = rgba[oi] as f32;
            roi[ro + 1] = rgba[oi + 1] as f32;
            roi[ro + 2] = rgba[oi + 2] as f32;
        }
    }
}

fn write_roi_mask_to_rgba(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    mw: usize,
    mh: usize,
    roi: &[f32],
    keep: &[bool],
) {
    let stride = w as usize;
    for py in 0..mh {
        for px in 0..mw {
            let i = py * mw + px;
            if !keep[i] {
                continue;
            }
            let fx = det.x as usize + px;
            let fy = det.y as usize + py;
            if fx >= w as usize || fy >= h as usize {
                continue;
            }
            let oi = (fy * stride + fx) * 4;
            let ro = i * 3;
            rgba[oi] = roi[ro].round().clamp(0.0, 255.0) as u8;
            rgba[oi + 1] = roi[ro + 1].round().clamp(0.0, 255.0) as u8;
            rgba[oi + 2] = roi[ro + 2].round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn alpha_grad_mag(alpha: &[f32], mw: usize, mh: usize) -> Vec<f32> {
    let mut g = vec![0f32; mw * mh];
    if mw < 2 || mh < 2 {
        return g;
    }
    for y in 0..mh {
        for x in 0..mw {
            let xm = if x == 0 { 0 } else { x - 1 };
            let xp = if x + 1 >= mw { mw - 1 } else { x + 1 };
            let ym = if y == 0 { 0 } else { y - 1 };
            let yp = if y + 1 >= mh { mh - 1 } else { y + 1 };
            // Match numpy.gradient: interior central /2, borders one-sided.
            let dx = if x == 0 || x + 1 >= mw {
                alpha[y * mw + xp] - alpha[y * mw + xm]
            } else {
                0.5 * (alpha[y * mw + xp] - alpha[y * mw + xm])
            };
            let dy = if y == 0 || y + 1 >= mh {
                alpha[yp * mw + x] - alpha[ym * mw + x]
            } else {
                0.5 * (alpha[yp * mw + x] - alpha[ym * mw + x])
            };
            g[y * mw + x] = (dx * dx + dy * dy).sqrt();
        }
    }
    g
}

fn dilate_mask_3x3(mask: &mut [bool], mw: usize, mh: usize) {
    let src = mask.to_vec();
    for y in 0..mh {
        for x in 0..mw {
            let mut on = false;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || xx < 0 || yy as usize >= mh || xx as usize >= mw {
                        continue;
                    }
                    if src[yy as usize * mw + xx as usize] {
                        on = true;
                    }
                }
            }
            mask[y * mw + x] = on;
        }
    }
}

/// Mild Gaussian blur applied only on `mask` pixels (neighbors may be outside).
fn gaussian_masked(roi: &mut [f32], mask: &[bool], mw: usize, mh: usize, rad: i32, sigma: f32) {
    if rad <= 0 || sigma <= 0.0 {
        return;
    }
    let src = roi.to_vec();
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    for y in 0..mh {
        for x in 0..mw {
            let i = y * mw + x;
            if !mask[i] {
                continue;
            }
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || xx < 0 || yy as usize >= mh || xx as usize >= mw {
                        continue;
                    }
                    let w = (-((dx * dx + dy * dy) as f32) * inv_2s2).exp();
                    let o = (yy as usize * mw + xx as usize) * 3;
                    acc[0] += w * src[o];
                    acc[1] += w * src[o + 1];
                    acc[2] += w * src[o + 2];
                    wsum += w;
                }
            }
            if wsum > 1e-6 {
                let o = i * 3;
                roi[o] = acc[0] / wsum;
                roi[o + 1] = acc[1] / wsum;
                roi[o + 2] = acc[2] / wsum;
            }
        }
    }
}

/// Reinject exterior high-frequency grain into mask pixels so a strong
/// footprint fill does not go plastic-smooth versus fabric grain.
fn reinject_exterior_grain(
    roi: &mut [f32],
    blended: &[f32],
    mask: &[bool],
    alpha: &[f32],
    mw: usize,
    mh: usize,
) {
    let strength = GRAIN_STRENGTH.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let mut blur = blended.to_vec();
    let all_true = vec![true; mw * mh];
    let blur_rad = ((GRAIN_BLUR_SIGMA * 2.0).ceil() as i32).max(1);
    gaussian_masked(&mut blur, &all_true, mw, mh, blur_rad, GRAIN_BLUR_SIGMA);
    let mut noise = vec![0f32; mw * mh * 3];
    for i in 0..mw * mh * 3 {
        noise[i] = blended[i] - blur[i];
    }

    let mut ext_sum = [0f32; 3];
    let mut ext_sq = [0f32; 3];
    let mut ext_n = 0f32;
    for i in 0..mw * mh {
        if alpha[i] >= 0.02 {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            let v = noise[o + c];
            ext_sum[c] += v;
            ext_sq[c] += v * v;
        }
        ext_n += 1.0;
    }
    if ext_n < 8.0 {
        return;
    }
    let mut target_std = [0f32; 3];
    for c in 0..3 {
        let mean = ext_sum[c] / ext_n;
        target_std[c] = ((ext_sq[c] / ext_n) - mean * mean)
            .max(0.0)
            .sqrt()
            .max(1e-3);
    }

    let mut synth = noise.clone();
    let mut ext_w = vec![0f32; mw * mh];
    for i in 0..mw * mh {
        if alpha[i] >= 0.02 {
            let o = i * 3;
            synth[o] = 0.0;
            synth[o + 1] = 0.0;
            synth[o + 2] = 0.0;
            ext_w[i] = 0.0;
        } else {
            ext_w[i] = 1.0;
        }
    }

    let rad = ((GRAIN_GUIDE_SIGMA * 2.0).ceil() as i32).max(1);
    let sigma = GRAIN_GUIDE_SIGMA.max(0.1);
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut guided = vec![0f32; mw * mh * 3];
    for y in 0..mh {
        for x in 0..mw {
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || xx < 0 || yy as usize >= mh || xx as usize >= mw {
                        continue;
                    }
                    let j = yy as usize * mw + xx as usize;
                    let w = (-((dx * dx + dy * dy) as f32) * inv_2s2).exp() * ext_w[j];
                    let o = j * 3;
                    acc[0] += w * synth[o];
                    acc[1] += w * synth[o + 1];
                    acc[2] += w * synth[o + 2];
                    wsum += w;
                }
            }
            let i = y * mw + x;
            let o = i * 3;
            if wsum > 1e-6 {
                guided[o] = acc[0] / wsum;
                guided[o + 1] = acc[1] / wsum;
                guided[o + 2] = acc[2] / wsum;
            }
        }
    }

    let mut base = roi.to_vec();
    gaussian_masked(&mut base, mask, mw, mh, blur_rad, GRAIN_BLUR_SIGMA);

    let mut g_sum = [0f32; 3];
    let mut g_sq = [0f32; 3];
    let mut g_n = 0f32;
    for i in 0..mw * mh {
        if !mask[i] {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            let v = guided[o + c];
            g_sum[c] += v;
            g_sq[c] += v * v;
        }
        g_n += 1.0;
    }
    if g_n < 4.0 {
        return;
    }
    let mut scale = [1f32; 3];
    for c in 0..3 {
        let mean = g_sum[c] / g_n;
        let std = ((g_sq[c] / g_n) - mean * mean).max(0.0).sqrt().max(1e-3);
        scale[c] = target_std[c] / std;
    }
    for i in 0..mw * mh {
        if !mask[i] {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            let grain = guided[o + c] * scale[c];
            roi[o + c] = base[o + c] + grain * strength;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::{MarkKind, VideoDetection, VideoMap};

    /// Forward-blend white logo with map alpha onto a solid background.
    fn blend_forward(bg: u8, map: &VideoMap, scale: f32) -> Vec<u8> {
        let n = (map.width as usize) * (map.height as usize);
        let mut out = vec![0u8; n * 4];
        for i in 0..n {
            let a = (map.alpha[i] * scale).clamp(0.0, 1.0) as f64;
            let (lr, lg, lb) = match &map.rgb {
                Some(rgb) => {
                    let o = i * 3;
                    (rgb[o] as f64, rgb[o + 1] as f64, rgb[o + 2] as f64)
                }
                None => (255.0, 255.0, 255.0),
            };
            let o = i * 4;
            out[o] = ((a * lr + (1.0 - a) * bg as f64).round() as u8).min(255);
            out[o + 1] = ((a * lg + (1.0 - a) * bg as f64).round() as u8).min(255);
            out[o + 2] = ((a * lb + (1.0 - a) * bg as f64).round() as u8).min(255);
            out[o + 3] = 255;
        }
        out
    }

    #[test]
    fn synthetic_diamond_patch_recovers_background() {
        // Tiny diamond-ish alpha: center high, corners zero.
        let w = 8u32;
        let h = 8u32;
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
            width: w,
            height: h,
            alpha,
            rgb: None, // white overlay
        };
        let bg = 80u8;
        let scale = 1.0f32;
        // Pad canvas so residual cleanup has an exterior ring.
        let cw = 16u32;
        let ch = 16u32;
        let ox = 4u32;
        let oy = 4u32;
        let mut canvas = vec![bg; (cw as usize) * (ch as usize) * 4];
        for i in 0..(cw * ch) as usize {
            canvas[i * 4 + 3] = 255;
        }
        let patch = blend_forward(bg, &map, scale);
        for py in 0..8usize {
            for px in 0..8usize {
                let src = (py * 8 + px) * 4;
                let dst = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
                canvas[dst..dst + 4].copy_from_slice(&patch[src..src + 4]);
            }
        }
        // Sanity: watermarked diamond center is brighter than bg.
        let center = (((oy as usize) + 3) * cw as usize + (ox as usize) + 3) * 4;
        assert!(
            canvas[center] > bg + 5,
            "blended center should lift, got {} vs bg {}",
            canvas[center],
            bg
        );

        let det = VideoDetection {
            mark: MarkKind::Diamond,
            x: ox,
            y: oy,
            w,
            h,
            score: 1.0,
        };
        remove_on_frame(&mut canvas, cw, ch, &det, &map, scale);

        for py in 0..8usize {
            for px in 0..8usize {
                let a = map.alpha[py * 8 + px];
                if a < 0.02 {
                    continue;
                }
                let o = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
                assert!(
                    (canvas[o] as i16 - bg as i16).abs() <= 3,
                    "pixel ({px},{py}) R={} want ~{bg}",
                    canvas[o]
                );
                assert!((canvas[o + 1] as i16 - bg as i16).abs() <= 3);
                assert!((canvas[o + 2] as i16 - bg as i16).abs() <= 3);
            }
        }
    }

    #[test]
    fn map_to_template_scales_alpha() {
        let map = VideoMap {
            width: 1,
            height: 1,
            alpha: vec![0.5],
            rgb: Some(vec![10, 20, 30]),
        };
        let tpl = map_to_template(&map, 0.5);
        assert_eq!(tpl.width, 1);
        assert_eq!(&tpl.data[..3], &[10, 20, 30]);
        // 0.5 * 0.5 = 0.25 → 64
        assert_eq!(tpl.data[3], 64);
    }

    #[test]
    fn map_to_template_defaults_white() {
        let map = VideoMap {
            width: 1,
            height: 1,
            alpha: vec![1.0],
            rgb: None,
        };
        let tpl = map_to_template(&map, 1.0);
        assert_eq!(&tpl.data[..3], &[255, 255, 255]);
        assert_eq!(tpl.data[3], 255);
    }
    #[test]
    fn alpha_grad_peaks_on_step_edge() {
        // 4×4: left half 0, right half 0.3 → |∇α| peaks on the vertical step.
        let mut alpha = vec![0.0f32; 16];
        for y in 0..4 {
            for x in 2..4 {
                alpha[y * 4 + x] = 0.3;
            }
        }
        let g = alpha_grad_mag(&alpha, 4, 4);
        // Columns 1 and 2 sit on the discontinuity.
        let edge = g[1] + g[2] + g[4 + 1] + g[4 + 2];
        let flat = g[0] + g[3] + g[4 + 0] + g[4 + 3];
        assert!(edge > flat + 0.05, "edge={edge} flat={flat}");
    }

    #[test]
    fn edge_cleanup_reduces_synthetic_silhouette() {
        // Solid bg with a bright diamond-ish overlay; after remove, inject a dark
        // ring on high-∇α pixels and confirm cleanup pulls it toward neighbors.
        let w = 8u32;
        let h = 8u32;
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
            width: w,
            height: h,
            alpha: alpha.clone(),
            rgb: None,
        };
        let bg = 100u8;
        let cw = 16u32;
        let ch = 16u32;
        let ox = 4u32;
        let oy = 4u32;
        let mut canvas = vec![bg; (cw as usize) * (ch as usize) * 4];
        for i in 0..(cw * ch) as usize {
            canvas[i * 4 + 3] = 255;
        }
        let patch = blend_forward(bg, &map, 1.0);
        for py in 0..8usize {
            for px in 0..8usize {
                let src = (py * 8 + px) * 4;
                let dst = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
                canvas[dst..dst + 4].copy_from_slice(&patch[src..src + 4]);
            }
        }
        let det = VideoDetection {
            mark: MarkKind::Diamond,
            x: ox,
            y: oy,
            w,
            h,
            score: 1.0,
        };
        // Blend-only first; snapshot energy on the blended-clean ROI, then paint
        // a dark silhouette so residual survival stays above the ring threshold.
        remove_on_frame_blend_only(&mut canvas, cw, ch, &det, &map, 1.0);
        let before_energy = roi_sil_energy(&canvas, cw, ch, &det, &map).max(1.0);
        let grad = alpha_grad_mag(&map.alpha, 8, 8);
        let mut before_contrast = 0.0f32;
        let mut n = 0u32;
        for py in 0..8usize {
            for px in 0..8usize {
                let i = py * 8 + px;
                if grad[i] <= EDGE_GRAD_THR || map.alpha[i] < EDGE_ALPHA_EPS {
                    continue;
                }
                let o = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
                canvas[o] = 40;
                canvas[o + 1] = 40;
                canvas[o + 2] = 40;
                before_contrast += (bg as f32 - 40.0).abs();
                n += 1;
            }
        }
        assert!(n > 4, "expected an edge ring to paint, got {n}");
        let rb_mask = vec![0u8; (cw * ch) as usize];
        residual_cleanup(&mut canvas, cw, ch, &det, &map, &rb_mask, before_energy);
        let mut after_contrast = 0.0f32;
        for py in 0..8usize {
            for px in 0..8usize {
                let i = py * 8 + px;
                if grad[i] <= EDGE_GRAD_THR || map.alpha[i] < EDGE_ALPHA_EPS {
                    continue;
                }
                let o = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
                after_contrast += (canvas[o] as f32 - bg as f32).abs();
            }
        }
        assert!(
            after_contrast < before_contrast * 0.80,
            "edge cleanup should cut silhouette contrast: before={before_contrast} after={after_contrast}"
        );
    }

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

    fn rgb_std(
        rgba: &[u8],
        cw: u32,
        ox: u32,
        oy: u32,
        mw: usize,
        mh: usize,
        keep: impl Fn(usize) -> bool,
    ) -> f32 {
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
                let l = 0.299 * rgba[o] as f32
                    + 0.587 * rgba[o + 1] as f32
                    + 0.114 * rgba[o + 2] as f32;
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

    /// Pre-drop full-footprint TELEA at old `FOOT_HARD_MIX=0.90` /
    /// `RESID_STRENGTH=0.92`. Test-only; does not restore production FOOT_*.
    /// No FOOT_DILATE: α<0.02 stays the unsmeared exterior reference.
    fn old_full_footprint_telea_mix_090_092(
        rgba: &mut [u8],
        w: u32,
        h: u32,
        det: &VideoDetection,
        map: &VideoMap,
        alpha_scale: f32,
    ) {
        remove_on_frame_blend_only(rgba, w, h, det, map, alpha_scale);
        let mw = map.width as usize;
        let mh = map.height as usize;
        let mut roi = vec![0f32; mw * mh * 3];
        copy_rgba_to_roi(rgba, w, h, det, mw, mh, &mut roi);
        let blended = roi.clone();
        let mut mask = vec![false; mw * mh];
        for i in 0..mw * mh {
            if map.alpha[i] > 0.015 {
                mask[i] = true;
            }
        }
        if !mask.iter().any(|&m| m) {
            return;
        }
        let radius = if mw <= 48 { 5 } else { 7 };
        inpaint_telea(&mut roi, &mask, mw, mh, radius);
        gaussian_masked(
            &mut roi,
            &mask,
            mw,
            mh,
            EDGE_RING_GAUSS_RADIUS,
            EDGE_RING_GAUSS_SIGMA,
        );
        let strength = 0.92f32;
        let hard = 0.90f32;
        let ramp = 0.12f32;
        for i in 0..mw * mh {
            if !mask[i] {
                continue;
            }
            let a = map.alpha[i].clamp(0.0, 1.0);
            let w_mix = (strength * (a / ramp).clamp(0.0, 1.0)).max(hard * strength);
            let o = i * 3;
            for c in 0..3 {
                roi[o + c] = roi[o + c] * w_mix + blended[o + c] * (1.0 - w_mix);
            }
        }
        write_roi_mask_to_rgba(rgba, w, h, det, mw, mh, &roi, &mask);
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
            VideoMap {
                width: 8,
                height: 8,
                alpha: alpha.clone(),
                rgb: None,
            }
        };
        let cw = 16u32;
        let ch = 16u32;
        let ox = 4u32;
        let oy = 4u32;
        let mut blended = checker_canvas(cw, ch);
        for py in 0..8usize {
            for px in 0..8usize {
                let a = map.alpha[py * 8 + px] as f64;
                if a <= 0.0 {
                    continue;
                }
                let dst = (((oy as usize) + py) * cw as usize + (ox as usize) + px) * 4;
                for c in 0..3 {
                    blended[dst + c] =
                        (a * 255.0 + (1.0 - a) * blended[dst + c] as f64).round() as u8;
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
        let mut canvas = blended.clone();
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

        let mut old = blended;
        old_full_footprint_telea_mix_090_092(&mut old, cw, ch, &det, &map, 1.0);
        let old_inner = rgb_std(&old, cw, ox, oy, 8, 8, |i| map.alpha[i] > 0.05);
        let old_outer = rgb_std(&old, cw, ox, oy, 8, 8, |i| map.alpha[i] < 0.02);
        assert!(
            old_outer > 20.0,
            "old-mix exterior should stay unsmeared, std={old_outer}"
        );
        assert!(
            old_inner < 0.50 * old_outer,
            "old 0.90/0.92 full-footprint TELEA must fall below the 50% bar (inner={old_inner} outer={old_outer}); new path inner={inner} outer={outer}"
        );
    }
}
