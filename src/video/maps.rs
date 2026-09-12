//! Embedded video watermark alpha (+ optional RGB) maps.
//!
//! Ships 720p (48×48) and 1080p (72×72) Gemini diamond maps extracted from
//! GeminiWatermarkTool-Video and validated against sample clip ROIs.
//!
//! Reverse-blend uses a **white** overlay (see [`diamond_map_720p_standard`]):
//! the on-disk RGB companion is an alpha-shaped grayscale preview, not logo chrome.

use std::sync::OnceLock;

/// Alpha (+ optional straight RGB) overlay map for reverse-blend on video frames.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoMap {
    pub width: u32,
    pub height: u32,
    /// Per-pixel opacity in [0, 1], row-major (`width * height`).
    pub alpha: Vec<f32>,
    /// Optional straight (non-premultiplied) RGB, length `width * height * 3`.
    ///
    /// When `None`, [`crate::video::frame::map_to_template`] uses pure white —
    /// correct for Gemini diamond chrome. Do not attach the grayscale preview
    /// bin as RGB; that under-removes (~50% of GWT BR mean-diff).
    pub rgb: Option<Vec<u8>>,
}

const DIAMOND_720P_W: u32 = 48;
const DIAMOND_720P_H: u32 = 48;

/// Little-endian f32 alpha, row-major 48×48 (values in [0, 1]).
const DIAMOND_ALPHA_720P: &[u8] = include_bytes!("../../assets/video/diamond_alpha_720p_48x48.f32");

/// Grayscale diamond preview from GWT (48×48×3), alpha-shaped mid-gray.
/// Kept embedded so the asset stays in the build; **not** used as blend RGB.
#[allow(dead_code)]
const DIAMOND_RGB_PREVIEW_720P: &[u8] =
    include_bytes!("../../assets/video/diamond_rgb_720p_48x48.bin");

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "f32 map byte length must be a multiple of 4"
    );
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

/// 720p standard Gemini diamond map (48×48).
///
/// Geometry prior on 1280×720: bottom-right inset ≈ 72px → top-left near
/// `(1136, 576)` (matches GWT sample hit).
///
/// `rgb` is `None` so reverse-blend uses white. Calibrated operating scale is
/// ≈ 0.97–1.0 vs GWT on `/workspace/video-in/input.mp4`.
pub fn diamond_map_720p_standard() -> VideoMap {
    static CACHE: OnceLock<VideoMap> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let expected = (DIAMOND_720P_W as usize) * (DIAMOND_720P_H as usize);
            debug_assert_eq!(DIAMOND_ALPHA_720P.len(), expected * 4);
            debug_assert_eq!(DIAMOND_RGB_PREVIEW_720P.len(), expected * 3);
            let alpha = decode_f32_le(DIAMOND_ALPHA_720P);
            debug_assert_eq!(alpha.len(), expected);
            VideoMap {
                width: DIAMOND_720P_W,
                height: DIAMOND_720P_H,
                alpha,
                rgb: None,
            }
        })
        .clone()
}

/// 720p compact diamond map (44×44), if an authentic map is available.
///
/// Phase-1: not extracted from GWT (no 44×44 resource found); returns `None`.
pub fn diamond_map_720p_compact() -> Option<VideoMap> {
    None
}

const DIAMOND_1080P_W: u32 = 72;
const DIAMOND_1080P_H: u32 = 72;

/// Little-endian f32 alpha, row-major 72×72 (values in [0, 1]).
///
/// Derived from the same GWT 96×96 grayscale diamond PNG as the 720p map,
/// LANCZOS-scaled to 72×72 (1.5× the 48×48 720p operating size).
const DIAMOND_ALPHA_1080P: &[u8] =
    include_bytes!("../../assets/video/diamond_alpha_1080p_72x72.f32");

/// 1080p standard Gemini diamond map (72×72).
///
/// Geometry prior on 1920×1080: bottom-right margin ≈ 144px → top-left
/// `(1704, 864)` (matches GWT `1080p landscape relocated` hit, NCC ≈ 0.98).
///
/// `rgb` is `None` so reverse-blend uses white (same as 720p).
pub fn diamond_map_1080p_standard() -> VideoMap {
    static CACHE: OnceLock<VideoMap> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let expected = (DIAMOND_1080P_W as usize) * (DIAMOND_1080P_H as usize);
            debug_assert_eq!(DIAMOND_ALPHA_1080P.len(), expected * 4);
            let alpha = decode_f32_le(DIAMOND_ALPHA_1080P);
            debug_assert_eq!(alpha.len(), expected);
            VideoMap {
                width: DIAMOND_1080P_W,
                height: DIAMOND_1080P_H,
                alpha,
                rgb: None,
            }
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diamond_map_720p_standard_size() {
        let m = diamond_map_720p_standard();
        assert_eq!(m.width, 48);
        assert_eq!(m.height, 48);
        assert_eq!(m.alpha.len(), 48 * 48);
        let max_a = m.alpha.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_a > 0.05 && max_a < 0.95,
            "expected max alpha in (0.05, 0.95), got {max_a}"
        );
        // Blend uses white (preview RGB bin is not logo chrome).
        assert!(m.rgb.is_none(), "standard map must use white overlay");
        assert_eq!(DIAMOND_RGB_PREVIEW_720P.len(), 48 * 48 * 3);
        // Alpha finite and in [0, 1].
        assert!(m
            .alpha
            .iter()
            .all(|a| a.is_finite() && *a >= 0.0 && *a <= 1.0));
    }

    #[test]
    fn diamond_map_720p_compact_unavailable() {
        assert!(diamond_map_720p_compact().is_none());
    }

    #[test]
    fn diamond_map_1080p_standard_size() {
        let m = diamond_map_1080p_standard();
        assert_eq!(m.width, 72);
        assert_eq!(m.height, 72);
        assert_eq!(m.alpha.len(), 72 * 72);
        let max_a = m.alpha.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_a > 0.05 && max_a < 0.95,
            "expected max alpha in (0.05, 0.95), got {max_a}"
        );
        assert!(m.rgb.is_none(), "1080p map must use white overlay");
        assert!(m
            .alpha
            .iter()
            .all(|a| a.is_finite() && *a >= 0.0 && *a <= 1.0));
    }
}
