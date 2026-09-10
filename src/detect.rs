//! Locate Gemini's sparkle by asking whether its known overlay explains the pixels.
//!
//! Faithful port of gemini-watermark-remover `src/detect.ts`.

use crate::blend::reverse_alpha_blend;
use crate::template::sparkle_template;
use crate::WatermarkTemplate;

/// Gate 1: removal must destroy the mark's own outline.
const MAX_SILHOUETTE_SURVIVAL: f64 = 0.80;

/// Looser Gate 1 used only on the canonical inset window (88–104).
/// Faint 2K gravel sparkles measure ~0.92; the wide pass stays at 0.80 so
/// off-mark rock fits still fail.
const MAX_SILHOUETTE_SURVIVAL_CANON: f64 = 0.93;

/// Gate 2: leftover silhouette energy vs nearby control patches.
const MAX_SILHOUETTE_VS_CONTROL: f64 = 3.0;

/// Scale search bounds (mark is ~48px in real samples).
const MIN_SIZE: u32 = 42;
const MAX_SIZE: u32 = 56;

/// How far from the expected 96px inset to look.
const MIN_INSET: u32 = 40;
const MAX_INSET: u32 = 116;

/// Opacity scales for version differences.
const ALPHA_SCALES: [f64; 4] = [1.0, 1.25, 1.55, 1.9];

/// NCC locator (independent right/bottom margins). High on purpose so gravel
/// false peaks (posing 1K sandal ~0.61) never become the proposal.
const MIN_NCC: f32 = 0.70;

/// After NCC proposes, reject only if reverse-blend *adds* silhouette edges.
const NCC_SURVIVAL: f64 = 1.0;

/// Half-width of the border kept around a patch so gradients are defined.
const PATCH_MARGIN: i32 = 3;

/// Detected watermark placement and residual silhouette survival.
#[derive(Clone, Debug)]
pub struct Match {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Edge energy left on the mark silhouette after removal / before (lower is better).
    pub residual: f64,
    /// Winning scaled/opacity-adjusted overlay (same as used for the gates).
    pub template: WatermarkTemplate,
}

/// Locate Gemini's sparkle overlay. Returns `None` if no candidate clears both gates.
pub fn match_watermark(width: u32, height: u32, data: &[u8]) -> Option<Match> {
    debug_assert_eq!(data.len(), (width as usize) * (height as usize) * 4);
    // Canonical placement first; wide search only if that fails (older marks).
    search(
        width,
        height,
        data,
        88,
        104,
        MAX_SILHOUETTE_SURVIVAL_CANON,
    )
    .or_else(|| search(width, height, data, MIN_INSET, MAX_INSET, MAX_SILHOUETTE_SURVIVAL))
    .or_else(|| ncc_search(width, height, data))
}

fn search(
    img_w: u32,
    img_h: u32,
    img: &[u8],
    min_inset: u32,
    max_inset: u32,
    max_survival: f64,
) -> Option<Match> {
    let base = sparkle_template();
    let edge = img_w.min(img_h);

    let mut best_x = 0u32;
    let mut best_y = 0u32;
    let mut best_size = 0u32;
    let mut best_tpl: Option<WatermarkTemplate> = None;
    let mut best_after = f64::INFINITY;
    let mut survival = f64::INFINITY;

    for s in MIN_SIZE..=MAX_SIZE {
        if s > edge / 2 {
            break;
        }
        let sized = scale_template(&base, s);

        for &alpha in &ALPHA_SCALES {
            let tpl = if (alpha - 1.0).abs() < f64::EPSILON {
                sized.clone()
            } else {
                with_opacity(&sized, alpha)
            };

            for ins in min_inset..=max_inset {
                // May underflow on tiny images; skip those placements.
                if img_w < ins + s || img_h < ins + s {
                    continue;
                }
                let x = img_w - ins - s;
                let y = img_h - ins - s;

                let before = silhouette_edges(img_w, img_h, img, &tpl, x, y, false);
                if !before.is_finite() || before < 1e-6 {
                    continue;
                }
                let after = silhouette_edges(img_w, img_h, img, &tpl, x, y, true);
                if !after.is_finite() {
                    continue;
                }

                let fraction = after / before;
                if fraction < survival {
                    survival = fraction;
                    best_after = after;
                    best_x = x;
                    best_y = y;
                    best_size = s;
                    best_tpl = Some(tpl.clone());
                }
            }
        }
    }

    let best_tpl = best_tpl?;
    if !survival.is_finite() {
        return None;
    }

    // Gate 1
    if survival > max_survival {
        return None;
    }

    // Gate 2
    let control = control_edges(img_w, img_h, img, &best_tpl, best_x, best_y);
    if control > 1e-6 && best_after / control > MAX_SILHOUETTE_VS_CONTROL {
        return None;
    }

    Some(Match {
        x: best_x,
        y: best_y,
        width: best_size,
        height: best_size,
        residual: survival,
        template: best_tpl,
    })
}

/// Gradient energy along the template silhouette (optionally after reverse-blend).
fn silhouette_edges(
    img_w: u32,
    img_h: u32,
    img: &[u8],
    tpl: &WatermarkTemplate,
    at_x: u32,
    at_y: u32,
    remove_first: bool,
) -> f64 {
    let Some(mut patch) = grab(img_w, img_h, img, tpl.width, at_x, at_y) else {
        return f64::INFINITY;
    };
    if remove_first {
        let pw = patch.width;
        let ph = patch.height;
        let _mask = reverse_alpha_blend(
            pw,
            ph,
            &mut patch.data,
            tpl,
            PATCH_MARGIN,
            PATCH_MARGIN,
            0.95,
        );
    }
    edge_energy(&patch, tpl)
}

fn control_edges(
    img_w: u32,
    img_h: u32,
    img: &[u8],
    tpl: &WatermarkTemplate,
    at_x: u32,
    at_y: u32,
) -> f64 {
    let s = tpl.width as i32;
    let near = s + 8;
    let offsets: [(i32, i32); 6] = [
        (-near, 0),
        (0, -near),
        (-near, -near),
        (-2 * s, 0),
        (0, -2 * s),
        (-2 * s, -2 * s),
    ];

    let mut samples: Vec<f64> = Vec::new();
    for (ox, oy) in offsets {
        let x = at_x as i32 + ox;
        let y = at_y as i32 + oy;
        if x < 0 || y < 0 {
            continue;
        }
        if let Some(patch) = grab(img_w, img_h, img, tpl.width, x as u32, y as u32) {
            samples.push(edge_energy(&patch, tpl));
        }
    }
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() >> 1]
}

struct Patch {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

/// Copy a margin-padded window around the template's footprint.
fn grab(
    img_w: u32,
    img_h: u32,
    img: &[u8],
    size: u32,
    at_x: u32,
    at_y: u32,
) -> Option<Patch> {
    let m = PATCH_MARGIN;
    let pw = size as i32 + m * 2;
    let x0 = at_x as i32 - m;
    let y0 = at_y as i32 - m;
    if x0 < 0 || y0 < 0 || x0 + pw > img_w as i32 || y0 + pw > img_h as i32 {
        return None;
    }

    let pw_u = pw as u32;
    let mut data = vec![0u8; (pw_u as usize) * (pw_u as usize) * 4];
    let iw = img_w as usize;
    for y in 0..pw_u {
        for x in 0..pw_u {
            let si = ((y0 as u32 + y) as usize * iw + (x0 as u32 + x) as usize) * 4;
            let di = (y as usize * pw_u as usize + x as usize) * 4;
            data[di] = img[si];
            data[di + 1] = img[si + 1];
            data[di + 2] = img[si + 2];
            data[di + 3] = 255;
        }
    }
    Some(Patch {
        width: pw_u,
        height: pw_u,
        data,
    })
}

/// Gradient energy weighted by |grad alpha|, over a padded patch.
fn edge_energy(patch: &Patch, tpl: &WatermarkTemplate) -> f64 {
    let s = tpl.width as usize;
    let pw = patch.width as usize;
    let m = PATCH_MARGIN as usize;
    let t = &tpl.data;
    let d = &patch.data;

    let mut energy = 0.0f64;
    let mut weight = 0.0f64;

    for ty in 1..(s - 1) {
        for tx in 1..(s - 1) {
            let a = |xx: usize, yy: usize| -> i32 { t[(yy * s + xx) * 4 + 3] as i32 };
            let ax = a(tx + 1, ty) - a(tx - 1, ty);
            let ay = a(tx, ty + 1) - a(tx, ty - 1);
            let w = ((ax * ax + ay * ay) as f64).sqrt();
            if w < 8.0 {
                continue; // not on the silhouette
            }

            let i = ((ty + m) * pw + (tx + m)) * 4;
            let mut g = 0.0f64;
            for c in 0..3 {
                let gx = d[i + 4 + c] as i32 - d[i - 4 + c] as i32;
                let gy = d[i + pw * 4 + c] as i32 - d[i - pw * 4 + c] as i32;
                g += (gx * gx + gy * gy) as f64;
            }
            energy += g * w;
            weight += w;
        }
    }
    if weight > 0.0 {
        energy / weight
    } else {
        0.0
    }
}

/// Same overlay at a different opacity (alpha scaled, colour unchanged).
fn with_opacity(tpl: &WatermarkTemplate, scale: f64) -> WatermarkTemplate {
    let mut data = tpl.data.clone();
    for i in (3..data.len()).step_by(4) {
        let v = (tpl.data[i] as f64 * scale).round();
        data[i] = if v > 255.0 { 255 } else { v as u8 };
    }
    WatermarkTemplate {
        width: tpl.width,
        height: tpl.height,
        data,
    }
}

/// Bilinear resize of the RGBA template.
fn scale_template(tpl: &WatermarkTemplate, size: u32) -> WatermarkTemplate {
    let sw = tpl.width as i32;
    let sh = tpl.height as i32;
    let src = &tpl.data;
    let size_i = size as i32;
    let mut out = vec![0u8; (size as usize) * (size as usize) * 4];

    for y in 0..size_i {
        let fy = ((y as f64 + 0.5) * sh as f64) / size as f64 - 0.5;
        let y0 = fy.floor() as i32;
        let y0 = y0.clamp(0, sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - y0 as f64;

        for x in 0..size_i {
            let fx = ((x as f64 + 0.5) * sw as f64) / size as f64 - 0.5;
            let x0 = fx.floor() as i32;
            let x0 = x0.clamp(0, sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - x0 as f64;

            let i00 = ((y0 * sw + x0) * 4) as usize;
            let i01 = ((y0 * sw + x1) * 4) as usize;
            let i10 = ((y1 * sw + x0) * 4) as usize;
            let i11 = ((y1 * sw + x1) * 4) as usize;
            let o = ((y * size_i + x) * 4) as usize;

            for c in 0..4 {
                let top = src[i00 + c] as f64 * (1.0 - wx) + src[i01 + c] as f64 * wx;
                let bot = src[i10 + c] as f64 * (1.0 - wx) + src[i11 + c] as f64 * wx;
                out[o + c] = (top * (1.0 - wy) + bot * wy).round() as u8;
            }
        }
    }

    WatermarkTemplate {
        width: size,
        height: size,
        data: out,
    }
}

/// Independent-margin NCC over the bottom-right, confirmed by silhouette gates.
///
/// Silhouette search only walks the diagonal (`inset_x == inset_y`). NCC covers
/// the off-diagonal cases; it never accepts on score alone.
fn ncc_search(img_w: u32, img_h: u32, img: &[u8]) -> Option<Match> {
    let base = sparkle_template();
    let luma = rgba_to_luma(img, img_w, img_h);
    let mut best: Option<(f32, u32, u32, u32, WatermarkTemplate)> = None;

    for s in (MIN_SIZE..=MAX_SIZE).step_by(2) {
        if s > img_w.min(img_h) / 2 {
            break;
        }
        let tpl = scale_template(&base, s);
        let alpha: Vec<f32> = tpl.data.iter().skip(3).step_by(4).map(|&a| a as f32).collect();
        let mean = mean_f32(&alpha);
        let tnorm = centered_norm_f32(&alpha, mean);
        if tnorm < 1e-6 {
            continue;
        }
        for mx in MIN_INSET..=MAX_INSET {
            for my in MIN_INSET..=MAX_INSET {
                if img_w < mx + s || img_h < my + s {
                    continue;
                }
                let x = img_w - mx - s;
                let y = img_h - my - s;
                let score = ncc_at(&luma, img_w, x, y, s, s, &alpha, mean, tnorm);
                if score < MIN_NCC {
                    continue;
                }
                let take = best.as_ref().map(|(b, ..)| score > *b).unwrap_or(true);
                if take {
                    best = Some((score, x, y, s, tpl.clone()));
                }
            }
        }
    }

    let (_, x, y, s, tpl) = best?;
    let before = silhouette_edges(img_w, img_h, img, &tpl, x, y, false);
    let after = silhouette_edges(img_w, img_h, img, &tpl, x, y, true);
    if !before.is_finite() || before < 1e-6 || !after.is_finite() {
        return None;
    }
    let survival = after / before;
    if survival >= NCC_SURVIVAL {
        return None;
    }
    let control = control_edges(img_w, img_h, img, &tpl, x, y);
    if control > 1e-6 && after / control > MAX_SILHOUETTE_VS_CONTROL {
        return None;
    }
    Some(Match {
        x,
        y,
        width: s,
        height: s,
        residual: survival,
        template: tpl,
    })
}

fn rgba_to_luma(rgba: &[u8], width: u32, height: u32) -> Vec<f32> {
    let n = (width as usize) * (height as usize);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        let r = rgba[o] as f32;
        let g = rgba[o + 1] as f32;
        let b = rgba[o + 2] as f32;
        out.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    out
}

fn mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

fn centered_norm_f32(v: &[f32], mean: f32) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        let d = x - mean;
        s += d * d;
    }
    s.sqrt()
}

fn ncc_at(
    luma: &[f32],
    stride: u32,
    x: u32,
    y: u32,
    mw: u32,
    mh: u32,
    alpha: &[f32],
    tpl_mean: f32,
    tpl_norm: f32,
) -> f32 {
    let sw = stride as usize;
    let mw_u = mw as usize;
    let mh_u = mh as usize;

    let mut sum = 0.0f32;
    let n = (mw_u * mh_u) as f32;
    for row in 0..mh_u {
        let base = ((y as usize) + row) * sw + x as usize;
        for col in 0..mw_u {
            sum += luma[base + col];
        }
    }
    let patch_mean = sum / n;

    let mut dot = 0.0f32;
    let mut patch_ss = 0.0f32;
    for row in 0..mh_u {
        let base = ((y as usize) + row) * sw + x as usize;
        let arow = row * mw_u;
        for col in 0..mw_u {
            let pd = luma[base + col] - patch_mean;
            let td = alpha[arow + col] - tpl_mean;
            dot += pd * td;
            patch_ss += pd * pd;
        }
    }
    let patch_norm = patch_ss.sqrt();
    if patch_norm < 1e-6 || tpl_norm < 1e-6 {
        return -1.0;
    }
    dot / (patch_norm * tpl_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_sparkle_on_fixture() {
        let img = image::open("tests/fixtures/sparkle_sample.jpeg")
            .expect("fixture")
            .to_rgba8();
        let (w, h) = (img.width(), img.height());
        let m = match_watermark(w, h, img.as_raw()).expect("should detect sparkle");
        // Prior gwr and this port both land at ~(1255, 647) on 1376×768 (inset ~73, size 48).
        assert!(
            (m.x as i32 - 1255).abs() <= 20,
            "x={} (want ~1255)",
            m.x
        );
        assert!(
            (m.y as i32 - 647).abs() <= 20,
            "y={} (want ~647)",
            m.y
        );
        assert!((42..=56).contains(&m.width), "width={}", m.width);
        assert_eq!(m.width, m.height);
        assert!(m.residual <= 0.80, "residual={}", m.residual);
    }

    #[test]
    fn detects_sparkle_on_2k_gravel_crop() {
        let img = image::open("tests/fixtures/sparkle_gravel_2k.png")
            .expect("fixture")
            .to_rgba8();
        let (w, h) = (img.width(), img.height());
        let m = match_watermark(w, h, img.as_raw()).expect("should detect faint 2K gravel sparkle");
        assert!((m.x as i32 - 119).abs() <= 8, "x={}", m.x);
        assert!((m.y as i32 - 119).abs() <= 8, "y={}", m.y);
        assert!((42..=56).contains(&m.width));
        assert!(m.residual <= 0.93, "residual={}", m.residual);
    }

    #[test]
    fn unmarked_busy_noise_is_not_a_match() {
        // High-frequency luma noise, no composited sparkle.
        let w = 256u32;
        let h = 256u32;
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
        assert!(match_watermark(w, h, &data).is_none());
    }

    #[test]
    fn detects_off_diagonal_sparkle_via_ncc() {
        let w = 400u32;
        let h = 300u32;
        let mut data = vec![0x6a_u8; (w * h * 4) as usize];
        for i in (0..data.len()).step_by(4) {
            data[i + 3] = 255;
        }
        let tpl = crate::sparkle_template();
        let x = w - 96 - tpl.width; // right inset 96
        let y = h - 72 - tpl.height; // bottom inset 72
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
        let m = match_watermark(w, h, &data).expect("NCC+silhouette should find off-diagonal mark");
        assert!((m.x as i32 - x as i32).abs() <= 4, "x={} want {x}", m.x);
        assert!((m.y as i32 - y as i32).abs() <= 4, "y={} want {y}", m.y);
    }
}
