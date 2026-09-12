//! Multi-frame diamond / Veo probe via normalized cross-correlation (NCC).
//!
//! The 720p Gemini diamond is scored against the embedded alpha map in a
//! bottom-right margin prior. Multi-frame probe keeps the highest-scoring
//! consistent hit.

use super::maps::{
    diamond_map_1080p_standard, diamond_map_720p_compact, diamond_map_720p_standard, VideoMap,
};
use super::MarkKind;

/// Minimum NCC accepted as a detection (rejects cleaned / empty BR ~0.50).
const MIN_NCC: f32 = 0.55;

/// Bottom-right margin search bounds on a 720p reference frame (canonical ≈ 96).
const REF_MARGIN_LO: u32 = 48;
const REF_MARGIN_HI: u32 = 128;
const REF_HEIGHT: u32 = 720;

/// Detected watermark placement and NCC confidence.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoDetection {
    pub mark: MarkKind,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// Normalized cross-correlation in roughly \[-1, 1\]; higher is better.
    pub score: f32,
}

/// Detect a watermark on a single RGBA frame using the supplied maps.
///
/// Searches bottom-right geometry priors (scaled from 720p) and returns the
/// best NCC hit above [`MIN_NCC`], or `None`.
pub fn detect_on_frame(
    rgba: &[u8],
    width: u32,
    height: u32,
    maps: &[VideoMap],
) -> Option<VideoDetection> {
    if width == 0 || height == 0 || maps.is_empty() {
        return None;
    }
    let expected = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?;
    if rgba.len() != expected {
        return None;
    }

    let luma = rgba_to_luma(rgba, width, height);
    let (m_lo, m_hi) = margin_range(height);

    let mut best: Option<VideoDetection> = None;

    for map in maps {
        if map.width == 0 || map.height == 0 {
            continue;
        }
        if map.alpha.len() != (map.width as usize) * (map.height as usize) {
            continue;
        }
        if map.width > width || map.height > height {
            continue;
        }

        let tpl_mean = mean_f32(&map.alpha);
        let tpl_norm = centered_norm_f32(&map.alpha, tpl_mean);
        if tpl_norm < 1e-6 {
            continue;
        }

        let max_mx = width.saturating_sub(map.width);
        let max_my = height.saturating_sub(map.height);
        let mx_lo = m_lo.min(max_mx);
        let mx_hi = m_hi.min(max_mx);
        let my_lo = m_lo.min(max_my);
        let my_hi = m_hi.min(max_my);
        if mx_lo > mx_hi || my_lo > my_hi {
            continue;
        }

        for my in my_lo..=my_hi {
            for mx in mx_lo..=mx_hi {
                let x = width - mx - map.width;
                let y = height - my - map.height;
                let score = ncc_at(
                    &luma, width, x, y, map.width, map.height, &map.alpha, tpl_mean, tpl_norm,
                );
                if score < MIN_NCC {
                    continue;
                }
                let better = match &best {
                    None => true,
                    Some(b) => score > b.score,
                };
                if better {
                    best = Some(VideoDetection {
                        mark: MarkKind::Diamond,
                        x,
                        y,
                        w: map.width,
                        h: map.height,
                        score,
                    });
                }
            }
        }
    }

    best
}

/// Probe several RGBA frames and return the strongest detection.
///
/// Tries 720p (48×48) and 1080p (72×72) diamond maps. Prefer this over a single frame 0 —
/// some frames occlude the mark (low NCC) while others lock cleanly near the
/// geometry prior `(1136, 576)` on 1280×720.
pub fn detect_from_probe_frames(frames: &[(u32, u32, Vec<u8>)]) -> Option<VideoDetection> {
    if frames.is_empty() {
        return None;
    }

    let mut maps = vec![diamond_map_720p_standard(), diamond_map_1080p_standard()];
    if let Some(compact) = diamond_map_720p_compact() {
        maps.push(compact);
    }

    let mut best: Option<VideoDetection> = None;
    for (width, height, rgba) in frames {
        if let Some(det) = detect_on_frame(rgba, *width, *height, &maps) {
            let take = match &best {
                None => true,
                Some(b) => det.score > b.score,
            };
            if take {
                best = Some(det);
            }
        }
    }
    best
}

fn margin_range(frame_height: u32) -> (u32, u32) {
    let scale = frame_height as f32 / REF_HEIGHT as f32;
    let lo = ((REF_MARGIN_LO as f32) * scale).round() as u32;
    let hi = ((REF_MARGIN_HI as f32) * scale).round() as u32;
    let lo = lo.max(1);
    let hi = hi.max(lo);
    (lo, hi)
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

    // Patch mean
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
    use crate::video::maps::diamond_map_720p_standard;

    fn load_mid_frame() -> (u32, u32, Vec<u8>) {
        let img = image::open("tests/fixtures/video/mid_frame_720p.png")
            .expect("mid-frame fixture at tests/fixtures/video/mid_frame_720p.png")
            .to_rgba8();
        (img.width(), img.height(), img.into_raw())
    }

    #[test]
    fn detect_diamond_on_720p_mid_frame() {
        let (w, h, rgba) = load_mid_frame();
        assert_eq!((w, h), (1280, 720));
        let map = diamond_map_720p_standard();
        let det = detect_on_frame(&rgba, w, h, &[map]).expect("should detect diamond");
        assert_eq!(det.mark, MarkKind::Diamond);
        assert_eq!(det.w, 48);
        assert_eq!(det.h, 48);
        // Geometry prior ~(1136, 576); allow small snap slack.
        assert!((det.x as i32 - 1136).abs() <= 4, "x={} want ~1136", det.x);
        assert!((det.y as i32 - 576).abs() <= 4, "y={} want ~576", det.y);
        assert!(det.score >= MIN_NCC, "score {} below MIN_NCC", det.score);
        assert!(
            det.score > 0.7,
            "expected strong mid-frame NCC, got {}",
            det.score
        );
    }

    #[test]
    fn detect_from_probe_frames_prefers_strong_hit() {
        let (w, h, rgba) = load_mid_frame();
        // Synthetic weak frame: solid gray (no diamond) — should not win.
        let gray = vec![40u8; (w as usize) * (h as usize) * 4];
        let det = detect_from_probe_frames(&[(w, h, gray), (w, h, rgba)])
            .expect("probe should lock on mid frame");
        assert_eq!(det.mark, MarkKind::Diamond);
        assert!((det.x as i32 - 1136).abs() <= 4);
        assert!((det.y as i32 - 576).abs() <= 4);
        assert!(det.score > 0.7);
    }

    #[test]
    fn detect_rejects_empty_maps() {
        let (w, h, rgba) = load_mid_frame();
        assert!(detect_on_frame(&rgba, w, h, &[]).is_none());
    }
}
