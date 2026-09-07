//! Locate Gemini's sparkle by asking whether its known overlay explains the pixels.
//!
//! Faithful port of gemini-watermark-remover `src/detect.ts`.

use crate::blend::reverse_alpha_blend;
use crate::template::sparkle_template;
use crate::WatermarkTemplate;

/// Gate 1: removal must destroy the mark's own outline.
const MAX_SILHOUETTE_SURVIVAL: f64 = 0.80;

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
    search(width, height, data, 88, 104).or_else(|| search(width, height, data, MIN_INSET, MAX_INSET))
}

fn search(
    img_w: u32,
    img_h: u32,
    img: &[u8],
    min_inset: u32,
    max_inset: u32,
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
    if survival > MAX_SILHOUETTE_SURVIVAL {
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
}
