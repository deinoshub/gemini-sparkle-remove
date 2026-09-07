//! Remove Gemini visible sparkle watermarks from RGBA buffers.
//!
//! Crate name: `gemini-sparkle-remove` (import as `gemini_sparkle_remove`).
//! Pure Rust; optional `video` / `video-fdncnn` features need ffmpeg tools
//! (build-downloaded or PATH) and a per-target cmake-built libncnn — never Python.

pub mod blend;
pub mod cli;
pub mod detect;
pub mod native_link;
pub mod template;

#[cfg(feature = "video")]
pub mod video;

/// Overlay template as composited onto the image:
/// `observed = alpha * rgb + (1 - alpha) * original`.
/// RGBA buffer; alpha is overlay opacity, RGB is straight (non-premultiplied) overlay color.
#[derive(Clone, Debug)]
pub struct WatermarkTemplate {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub use blend::reverse_alpha_blend;
pub use template::{sparkle_template, SparkleGeometry, SPARKLE_GEOMETRY, SPARKLE_SIZE};

pub use detect::{match_watermark, Match};

/// Mutable RGBA view (`data.len() == width * height * 4`).
pub struct RgbaImage<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a mut [u8],
}

/// Outcome of [`remove_gemini_sparkle`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoveResult {
    Removed { x: u32, y: u32 },
    NotFound,
}

const OPAQUE_CUTOFF: f64 = 0.95;

/// Auto-detect the Gemini sparkle and reverse-blend it in place.
///
/// If no mark clears the detect gates, returns [`RemoveResult::NotFound`] and
/// leaves the buffer unchanged.
pub fn remove_gemini_sparkle(img: &mut RgbaImage<'_>) -> RemoveResult {
    assert_eq!(
        img.data.len(),
        (img.width as usize) * (img.height as usize) * 4
    );
    let Some(m) = match_watermark(img.width, img.height, img.data) else {
        return RemoveResult::NotFound;
    };
    let _mask = reverse_alpha_blend(
        img.width,
        img.height,
        img.data,
        &m.template,
        m.x as i32,
        m.y as i32,
        OPAQUE_CUTOFF,
    );
    RemoveResult::Removed { x: m.x, y: m.y }
}

/// Force reverse-blend of the canonical 48×48 sparkle at `(x, y)` (top-left).
pub fn remove_at(img: &mut RgbaImage<'_>, x: u32, y: u32) {
    assert_eq!(
        img.data.len(),
        (img.width as usize) * (img.height as usize) * 4
    );
    let tpl = sparkle_template();
    let _mask = reverse_alpha_blend(
        img.width,
        img.height,
        img.data,
        &tpl,
        x as i32,
        y as i32,
        OPAQUE_CUTOFF,
    );
}
