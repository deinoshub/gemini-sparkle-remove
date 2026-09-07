//! Per-frame reverse-blend using a VideoMap + intensity scale.
//!
//! After reverse-blend, cleanup targets the **full α footprint** (GWT video
//! marks all map pixels as denoise edges — 48×48 = 2304), not only the
//! high-`|∇α|` silhouette ring: exterior-preferring seed + Jacobi fill,
//! alpha-ramped mix, then exterior HF grain reinjection. Classical only.

use crate::blend::reverse_alpha_blend;
use crate::WatermarkTemplate;

use super::detect::VideoDetection;
use super::maps::VideoMap;
use super::telea::inpaint_telea;

/// Same opaque cutoff as the still-image remove path.
const OPAQUE_CUTOFF: f64 = 0.95;
/// Minimum map alpha to include in the edge-ring mask.
const EDGE_ALPHA_EPS: f32 = 0.008;

/// `|∇α|` threshold for the silhouette ring (central-difference magnitude).
const EDGE_GRAD_THR: f32 = 0.022;

/// Binary dilate iterations on the edge mask (expands ~1px into near-edge band).
const EDGE_DILATE: usize = 3;

/// Full-footprint mask: α above this is filled (GWT denoise covers all map px).
const FOOT_ALPHA_THR: f32 = 0.015;

/// Dilate iterations on the full footprint mask.
const FOOT_DILATE: usize = 2;

/// Gaussian seed radius / sigma for filling from known (non-mask) pixels.
#[allow(dead_code)]
const EDGE_SEED_RADIUS: i32 = 7;
#[allow(dead_code)]
const EDGE_SEED_SIGMA: f32 = 2.8;

/// Jacobi diffusion iterations after seeding.
#[allow(dead_code)]
const EDGE_JACOBI_ITERS: usize = 36;

/// Blend filled result toward the Gaussian seed (reduces Jacobi ripple).
#[allow(dead_code)]
const EDGE_SEED_BLEND: f32 = 0.50;

/// Mild Gaussian on the mask after Jacobi (radius / sigma).
const EDGE_RING_GAUSS_RADIUS: i32 = 2;
const EDGE_RING_GAUSS_SIGMA: f32 = 1.4;

/// Alpha-ramped mix of filled vs reverse-blend (higher → more fill).
const RESID_STRENGTH: f32 = 0.92;
const RESID_ALPHA_RAMP: f32 = 0.12;

/// Hard mask mix toward filled (in addition to alpha ramp).
const FOOT_HARD_MIX: f32 = 0.90;

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

/// Reverse-blend the watermark ROI on one RGBA frame, then full-footprint
/// residual cleanup (exterior-guided fill + grain).
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
    let _mask = reverse_alpha_blend(
        w,
        h,
        rgba,
        &tpl,
        det.x as i32,
        det.y as i32,
        OPAQUE_CUTOFF,
    );
    if do_edge_cleanup {
        edge_ring_cleanup(rgba, w, h, det, map);
    }
}


/// Soften residual diamond after reverse-blend via full-footprint fill.
///
/// 1. Mask = high-`|∇α|` ring ∪ dilated α footprint (GWT denoise = all map px).
/// 2. Exterior-preferring Gaussian seed, Jacobi, seed blend, mild blur.
/// 3. Alpha-ramped mix toward fill; reinject exterior HF grain.
fn edge_ring_cleanup(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    det: &VideoDetection,
    map: &VideoMap,
) {
    let mw = map.width as usize;
    let mh = map.height as usize;
    if mw == 0 || mh == 0 {
        return;
    }
    let grad = alpha_grad_mag(&map.alpha, mw, mh);
    let mut mask = vec![false; mw * mh];
    for i in 0..mw * mh {
        if map.alpha[i] >= EDGE_ALPHA_EPS && grad[i] > EDGE_GRAD_THR {
            mask[i] = true;
        }
    }
    for _ in 0..EDGE_DILATE {
        dilate_mask_3x3(&mut mask, mw, mh);
    }
    let mut foot = vec![false; mw * mh];
    for i in 0..mw * mh {
        if map.alpha[i] > FOOT_ALPHA_THR {
            foot[i] = true;
        }
    }
    for _ in 0..FOOT_DILATE {
        dilate_mask_3x3(&mut foot, mw, mh);
    }
    for i in 0..mw * mh {
        mask[i] = mask[i] || foot[i];
    }
    let mut support = vec![false; mw * mh];
    for i in 0..mw * mh {
        if map.alpha[i] > 0.008 {
            support[i] = true;
        }
    }
    dilate_mask_3x3(&mut support, mw, mh);
    dilate_mask_3x3(&mut support, mw, mh);
    for i in 0..mw * mh {
        mask[i] = mask[i] && support[i];
    }
    if !mask.iter().any(|&m| m) {
        return;
    }

    let stride = w as usize;
    let mut roi = vec![0f32; mw * mh * 3];
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

    let blended = roi.clone();
    // Onion-peel Telea-ish fill over the full footprint (python OpenCV TELEA
    // fill; FDnCNN feature uses in-process ncnn when enabled).
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

    let strength = RESID_STRENGTH.clamp(0.0, 1.0);
    let ramp = RESID_ALPHA_RAMP.max(1e-4);
    let hard = FOOT_HARD_MIX.clamp(0.0, 1.0);
    for i in 0..mw * mh {
        let a = map.alpha[i].clamp(0.0, 1.0);
        let mut w_mix = strength * (a / ramp).clamp(0.0, 1.0);
        if mask[i] {
            w_mix = w_mix.max(hard * strength);
        }
        if w_mix <= 1e-6 {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            roi[o + c] = roi[o + c] * w_mix + blended[o + c] * (1.0 - w_mix);
        }
    }

    reinject_exterior_grain(&mut roi, &blended, &mask, &map.alpha, mw, mh);

    for py in 0..mh {
        for px in 0..mw {
            let i = py * mw + px;
            if !mask[i] && map.alpha[i] <= FOOT_ALPHA_THR {
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


/// Replace mask pixels with Gaussian-weighted RGB from nearby non-mask pixels.
/// Prefer low-α / exterior neighbors when available.
#[allow(dead_code)]
fn seed_from_known_prefer_exterior(
    roi: &mut [f32],
    mask: &[bool],
    alpha: &[f32],
    mw: usize,
    mh: usize,
) {
    let src = roi.to_vec();
    let rad = EDGE_SEED_RADIUS;
    let sigma = EDGE_SEED_SIGMA;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    for y in 0..mh {
        for x in 0..mw {
            let i = y * mw + x;
            if !mask[i] {
                continue;
            }
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            let mut acc_any = [0f32; 3];
            let mut wsum_any = 0f32;
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || xx < 0 || yy as usize >= mh || xx as usize >= mw {
                        continue;
                    }
                    let j = yy as usize * mw + xx as usize;
                    if mask[j] {
                        continue;
                    }
                    let w = (-((dx * dx + dy * dy) as f32) * inv_2s2).exp();
                    let o = j * 3;
                    acc_any[0] += w * src[o];
                    acc_any[1] += w * src[o + 1];
                    acc_any[2] += w * src[o + 2];
                    wsum_any += w;
                    if alpha[j] >= 0.02 {
                        continue;
                    }
                    acc[0] += w * src[o];
                    acc[1] += w * src[o + 1];
                    acc[2] += w * src[o + 2];
                    wsum += w;
                }
            }
            let o = i * 3;
            if wsum > 1e-6 {
                roi[o] = acc[0] / wsum;
                roi[o + 1] = acc[1] / wsum;
                roi[o + 2] = acc[2] / wsum;
            } else if wsum_any > 1e-6 {
                roi[o] = acc_any[0] / wsum_any;
                roi[o + 1] = acc_any[1] / wsum_any;
                roi[o + 2] = acc_any[2] / wsum_any;
            }
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

#[allow(dead_code)]
fn jacobi_ring_step(roi: &mut [f32], mask: &[bool], mw: usize, mh: usize) {
    let src = roi.to_vec();
    for y in 0..mh {
        for x in 0..mw {
            let i = y * mw + x;
            if !mask[i] {
                continue;
            }
            let mut acc = [0f32; 3];
            let mut n = 0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let yy = (y as i32 + dy).clamp(0, mh as i32 - 1) as usize;
                    let xx = (x as i32 + dx).clamp(0, mw as i32 - 1) as usize;
                    let o = (yy * mw + xx) * 3;
                    acc[0] += src[o];
                    acc[1] += src[o + 1];
                    acc[2] += src[o + 2];
                    n += 1.0;
                }
            }
            let o = i * 3;
            roi[o] = acc[0] / n;
            roi[o + 1] = acc[1] / n;
            roi[o + 2] = acc[2] / n;
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
/// footprint fill does not go plastic-smooth vs GWT/FDnCNN fabric grain.
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
        target_std[c] = ((ext_sq[c] / ext_n) - mean * mean).max(0.0).sqrt().max(1e-3);
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
        // Blend-only first, then paint a dark silhouette on high-grad pixels.
        remove_on_frame_blend_only(&mut canvas, cw, ch, &det, &map, 1.0);
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
        edge_ring_cleanup(&mut canvas, cw, ch, &det, &map);
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
            after_contrast < before_contrast * 0.55,
            "edge cleanup should cut silhouette contrast: before={before_contrast} after={after_contrast}"
        );
    }

}
