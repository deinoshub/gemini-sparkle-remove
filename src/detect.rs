//! Locate Gemini's sparkle by asking whether its known overlay explains the pixels.
//!
//! Faithful port of gemini-watermark-remover `src/detect.ts`.

use crate::blend::reverse_alpha_blend;
use crate::template::sparkle_template;
use crate::WatermarkTemplate;

/// Gate 1: removal must destroy the mark's own outline.
const MAX_SILHOUETTE_SURVIVAL: f64 = 0.80;

/// Looser Gate 1 used only on the canonical inset window (88–104).
/// Faint 2K marks on busy texture measure ~0.92–0.96; the wide pass stays at
/// 0.80 so off-mark rock fits still fail.
const MAX_SILHOUETTE_SURVIVAL_CANON: f64 = 0.98;

/// Gate 2: leftover silhouette energy vs nearby control patches.
const MAX_SILHOUETTE_VS_CONTROL: f64 = 3.0;

/// Strong Gate 1 (outline mostly gone) may sit on a locally busier patch than
/// its neighbors (wood crack under the sparkle, Gate 2 ~3.46). Documented
/// rock impostor is ~3.9, so stay below that.
const STRONG_SURVIVAL: f64 = 0.70;
const STRONG_GATE2: f64 = 3.75;

/// Below this, Gate 2 is skipped: leftover JPEG ringing vs empty neighbors is
/// not an impostor signal. Wood fixture is ~0.52 so it still uses Gate 2.
/// Jazz-on-rope measured 0.293 — 0.007 under this cutoff.
const VERY_STRONG_SURVIVAL: f64 = 0.30;

/// Control below this on the strong path is uninformative (dark BR JPEG ~1).
/// Do not use this floor on the normal Gate 2 line (canonical max 0.98).
const CONTROL_INFORMATIVE: f64 = 8.0;

/// Scale search bounds (mark is ~48px in real samples).
const MIN_SIZE: u32 = 42;
const MAX_SIZE: u32 = 56;

/// How far from the expected 96px inset to look.
const MIN_INSET: u32 = 40;
const MAX_INSET: u32 = 116;

/// Opacity scales. Values below 1.0 cover faint 2K-on-sky / snow marks where
/// scale 1.0 over-subtracts (survival > 1).
const ALPHA_SCALES: [f64; 7] = [0.55, 0.70, 0.85, 1.0, 1.25, 1.55, 1.9];

/// NCC locator (independent right/bottom margins). High on purpose so gravel
/// false peaks (posing 1K sandal ~0.61) never become the proposal.
const MIN_NCC: f32 = 0.70;

/// After NCC proposes, reject only if reverse-blend *adds* silhouette edges.
const NCC_SURVIVAL: f64 = 1.0;

/// NCC Gate 2 may use the strong-path 3.75 limit when survival is this or lower.
const NCC_RELAX_SURVIVAL: f64 = 0.85;

/// How many lowest-survival silhouette placements to try through the gates.
const TOP_K: usize = 5;

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InsetWalk {
    Diagonal,
    Independent,
}

#[derive(Clone, Debug)]
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

fn search(
    img_w: u32,
    img_h: u32,
    img: &[u8],
    min_inset: u32,
    max_inset: u32,
    max_survival: f64,
    walk: InsetWalk,
) -> Option<Match> {
    let base = sparkle_template();
    let edge = img_w.min(img_h);

    let mut cands: Vec<SilhouetteCand> = Vec::new();

    let mut walk_xy = |mx: u32, my: u32, tpl: &WatermarkTemplate, s: u32| {
        if img_w < mx + s || img_h < my + s {
            return;
        }
        let x = img_w - mx - s;
        let y = img_h - my - s;

        let before = silhouette_edges(img_w, img_h, img, tpl, x, y, false);
        if !before.is_finite() || before < 1e-6 {
            return;
        }
        let after = silhouette_edges(img_w, img_h, img, tpl, x, y, true);
        if !after.is_finite() {
            return;
        }

        let fraction = after / before;
        if fraction.is_finite() {
            consider_cand(
                &mut cands,
                SilhouetteCand {
                    x,
                    y,
                    size: s,
                    survival: fraction,
                    after,
                    template: tpl.clone(),
                },
                img_w,
                img_h,
            );
        }
    };

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

    choose_match(cands, img_w, img_h, img, max_survival)
}

fn strong_path_ok(survival: f64, gate2: f64, control: f64) -> bool {
    if survival <= VERY_STRONG_SURVIVAL {
        return true;
    }
    survival <= STRONG_SURVIVAL && (gate2 <= STRONG_GATE2 || control < CONTROL_INFORMATIVE)
}

fn gates_ok(survival: f64, after: f64, control: f64, max_survival: f64) -> bool {
    if !survival.is_finite() {
        return false;
    }
    let gate2 = if control > 1e-6 { after / control } else { 0.0 };
    if strong_path_ok(survival, gate2, control) {
        return true;
    }
    if survival > max_survival {
        return false;
    }
    control <= 1e-6 || gate2 <= MAX_SILHOUETTE_VS_CONTROL
}

fn ncc_gates_ok(survival: f64, gate2: f64, control: f64) -> bool {
    if !survival.is_finite() {
        return false;
    }
    if strong_path_ok(survival, gate2, control) {
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
fn grab(img_w: u32, img_h: u32, img: &[u8], size: u32, at_x: u32, at_y: u32) -> Option<Patch> {
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
/// Wide silhouette search stays diagonal. NCC covers off-diagonal placements
/// outside the canonical window; it never accepts on score alone.
fn ncc_search(img_w: u32, img_h: u32, img: &[u8]) -> Option<Match> {
    let base = sparkle_template();
    let luma = rgba_to_luma(img, img_w, img_h);
    let mut best: Option<(f32, u32, u32, u32, WatermarkTemplate)> = None;

    for s in (MIN_SIZE..=MAX_SIZE).step_by(2) {
        if s > img_w.min(img_h) / 2 {
            break;
        }
        let tpl = scale_template(&base, s);
        let alpha: Vec<f32> = tpl
            .data
            .iter()
            .skip(3)
            .step_by(4)
            .map(|&a| a as f32)
            .collect();
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
    let control = control_edges(img_w, img_h, img, &tpl, x, y);
    let gate2 = if control > 1e-6 { after / control } else { 0.0 };
    if survival >= NCC_SURVIVAL || !ncc_gates_ok(survival, gate2, control) {
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
        assert!((m.x as i32 - 1255).abs() <= 20, "x={} (want ~1255)", m.x);
        assert!((m.y as i32 - 647).abs() <= 20, "y={} (want ~647)", m.y);
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
    fn detects_sparkle_on_2k_wood_crop() {
        // Strong Gate 1 (~0.52) but Gate 2 ~3.46 because the mark sits on a wood crack.
        let img = image::open("tests/fixtures/sparkle_wood_2k.png")
            .expect("fixture")
            .to_rgba8();
        let (w, h) = (img.width(), img.height());
        let m = match_watermark(w, h, img.as_raw()).expect("should detect sparkle on wood grain");
        assert!((m.x as i32 - 119).abs() <= 8, "x={}", m.x);
        assert!((m.y as i32 - 119).abs() <= 8, "y={}", m.y);
        assert!((42..=56).contains(&m.width));
        assert!(m.residual <= 0.70, "residual={}", m.residual);
    }

    #[test]
    fn detects_sparkle_on_2k_asphalt_crop() {
        // Faint mark on wet asphalt: Gate 1 ~0.96, Gate 2 ~1.14.
        let img = image::open("tests/fixtures/sparkle_asphalt_2k.png")
            .expect("fixture")
            .to_rgba8();
        let (w, h) = (img.width(), img.height());
        let m = match_watermark(w, h, img.as_raw()).expect("should detect faint asphalt sparkle");
        assert!((m.x as i32 - 119).abs() <= 8, "x={}", m.x);
        assert!((m.y as i32 - 119).abs() <= 8, "y={}", m.y);
        assert!((42..=56).contains(&m.width));
        assert!(m.residual <= 0.98, "residual={}", m.residual);
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

    fn paint_sparkle(data: &mut [u8], w: u32, x: u32, y: u32) {
        paint_template(data, w, x, y, &crate::sparkle_template());
    }

    fn paint_template(data: &mut [u8], w: u32, x: u32, y: u32, tpl: &crate::WatermarkTemplate) {
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

    fn fill_rgb(data: &mut [u8], r: u8, g: u8, b: u8) {
        for i in 0..data.len() / 4 {
            data[i * 4] = r;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = b;
            data[i * 4 + 3] = 255;
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

    #[test]
    fn top_k_skips_failing_lowest_survival() {
        let tpl = crate::sparkle_template();
        let loser = SilhouetteCand {
            x: 80,
            y: 80,
            size: tpl.width,
            survival: 0.50,
            after: 1.0e6,
            template: tpl.clone(),
        };
        let winner = SilhouetteCand {
            x: 40,
            y: 40,
            size: tpl.width,
            survival: 0.60,
            after: 1.0,
            template: tpl,
        };
        // Fake image large enough for control_edges on winner (x=40,y=40,s=48).
        // XOR texture keeps control above CONTROL_INFORMATIVE; uniform grey
        // would take the strong-path empty-control pass (loser survival 0.50).
        let w = 200u32;
        let h = 200u32;
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = 80u8.wrapping_add(((x.wrapping_mul(13)) ^ (y.wrapping_mul(7))) as u8);
                data[i] = v;
                data[i + 1] = v.saturating_sub(9);
                data[i + 2] = v.saturating_add(11);
                data[i + 3] = 255;
            }
        }
        // Loser at (80,80) so grab() succeeds; after=1e6 so Gate 2 fails even
        // if control is modest (XOR keeps control well above CONTROL_INFORMATIVE).
        // Winner after=1.0 passes Gate 2.
        let got = choose_match(vec![loser, winner.clone()], w, h, &data, 0.98)
            .expect("second candidate must pass after first fails gates");
        assert_eq!(got.x, 40);
        assert_eq!(got.y, 40);
        assert!((got.residual - 0.60).abs() < 1e-9);
    }

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

    #[test]
    fn very_strong_survival_passes_despite_empty_control() {
        // Diamond-like: outline gone (surv 0.02) but JPEG-dark neighbors make
        // Gate 2 = 60/0.8 = 75. Must not reject.
        assert!(gates_ok(0.02, 60.0, 0.8, MAX_SILHOUETTE_SURVIVAL_CANON));
        assert!(ncc_gates_ok(0.02, 75.0, 0.8));
    }

    #[test]
    fn very_strong_survival_passes_with_informative_control() {
        // Frost/PCB-like: surv 0.25, control ~70, Gate 2 ~5.7 (> 3.75).
        assert!(gates_ok(0.25, 400.0, 70.0, MAX_SILHOUETTE_SURVIVAL_CANON));
        assert!(ncc_gates_ok(0.25, 5.71, 70.0));
    }

    #[test]
    fn rock_impostor_still_fails_silhouette_gates() {
        assert!(!gates_ok(0.80, 3.9, 1.0, MAX_SILHOUETTE_SURVIVAL));
    }

    #[test]
    fn strong_path_accepts_uninformative_control() {
        // surv 0.50 is above VERY_STRONG, below STRONG; Gate 2 exploded.
        assert!(gates_ok(0.50, 60.0, 0.8, MAX_SILHOUETTE_SURVIVAL_CANON));
        assert!(ncc_gates_ok(0.50, 75.0, 0.8));
    }

    #[test]
    fn normal_path_does_not_treat_jpeg_dark_as_empty_control() {
        // Raising the 1e-6 floor to CONTROL_INFORMATIVE on this line would
        // let survival ~0.75 blobs on black through canonical max 0.98.
        assert!(!gates_ok(0.75, 60.0, 0.8, MAX_SILHOUETTE_SURVIVAL_CANON));
        assert!(!ncc_gates_ok(0.75, 75.0, 0.8));
    }

    fn noise_canvas(w: u32, h: u32) -> Vec<u8> {
        // XOR lattice around mid-grey. Full-range XOR (unmarked_busy_noise) makes
        // a peak-α≈0.31 overlay smooth the patch (survival > 1), so bound AMP.
        let mut data = vec![0u8; (w * h * 4) as usize];
        const AMP: u8 = 24;
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let n = (((x.wrapping_mul(37)) ^ (y.wrapping_mul(91))) % (AMP as u32 * 2)) as u8;
                let v = 0x6a_u8.saturating_add(n).saturating_sub(AMP);
                data[i] = v;
                data[i + 1] = v.saturating_sub(7);
                data[i + 2] = v.saturating_add(5);
                data[i + 3] = 255;
            }
        }
        data
    }

    #[test]
    fn faint_sparkle_on_bright_sky_is_found() {
        // 2K-on-sky: real mark is ~0.55 of the template. Scale 1.0 punches a
        // dark hole (survival > 1) so ALPHA_SCALES must include fainter values.
        let w = 400u32;
        let h = 300u32;
        let s = crate::SPARKLE_SIZE;
        let x = w - 89 - s;
        let y = h - 89 - s;
        let mut data = vec![0u8; (w * h * 4) as usize];
        fill_rgb(&mut data, 180, 210, 235);
        let faint = with_opacity(&crate::sparkle_template(), 0.55);
        paint_template(&mut data, w, x, y, &faint);
        let m = match_watermark(w, h, &data).expect("faint sparkle on bright sky");
        assert!((m.x as i32 - x as i32).abs() <= 4, "x={} want {x}", m.x);
        assert!((m.y as i32 - y as i32).abs() <= 4, "y={} want {y}", m.y);
    }

    #[test]
    fn noise_without_overlay_is_not_found() {
        let data = noise_canvas(400, 300);
        assert!(match_watermark(400, 300, &data).is_none());
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
}
