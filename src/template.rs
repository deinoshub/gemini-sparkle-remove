//! Embedded Gemini sparkle watermark template (48×48 RGBA).

use std::sync::OnceLock;

use crate::WatermarkTemplate;

/// Side length of the sparkle mark in pixels.
pub const SPARKLE_SIZE: u32 = 48;

/// Absolute placement geometry for the sparkle (independent of frame size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparkleGeometry {
    pub size_px: u32,
    pub inset_px: u32,
}

/// Fixed 48px mark, inset 96px from the right and bottom edges.
pub const SPARKLE_GEOMETRY: SparkleGeometry = SparkleGeometry {
    size_px: 48,
    inset_px: 96,
};

const SPARKLE_RGBA: &[u8] = include_bytes!("../assets/sparkle_rgba.bin");

/// Decode the embedded template to RGBA. Cached; safe to call per image.
pub fn sparkle_template() -> WatermarkTemplate {
    static CACHE: OnceLock<WatermarkTemplate> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            debug_assert_eq!(
                SPARKLE_RGBA.len(),
                (SPARKLE_SIZE as usize) * (SPARKLE_SIZE as usize) * 4
            );
            WatermarkTemplate {
                width: SPARKLE_SIZE,
                height: SPARKLE_SIZE,
                data: SPARKLE_RGBA.to_vec(),
            }
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkle_template_is_48x48_rgba() {
        let tpl = sparkle_template();
        assert_eq!(tpl.width, 48);
        assert_eq!(tpl.height, 48);
        assert_eq!(tpl.data.len(), 48 * 48 * 4);
        assert_eq!(SPARKLE_SIZE, 48);
        assert_eq!(SPARKLE_GEOMETRY.size_px, 48);
        assert_eq!(SPARKLE_GEOMETRY.inset_px, 96);
        // Peak opacity ~0.31 → alpha byte ≤ ~79; bound with headroom.
        let max_a = tpl
            .data
            .iter()
            .skip(3)
            .step_by(4)
            .copied()
            .max()
            .unwrap_or(0);
        assert!(max_a <= 90, "expected peak sparkle alpha ≤ 90, got {max_a}");
    }
}
