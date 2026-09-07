//! Optional video watermark removal (feature = `"video"`).
//!
//! Phase-1: public types, errors, embedded 720p/1080p diamond maps, NCC detect,
//! adaptive alpha, per-frame reverse-blend, and ffmpeg `remove_video` pipeline.

mod alpha;
mod detect;
mod error;
mod frame;
mod maps;
mod pipeline;
mod telea;
#[cfg(feature = "video-fdncnn")]
mod ncnn_ffi;
#[cfg(feature = "video-fdncnn")]
mod fdncnn;

pub use alpha::{estimate_alpha, refine_alpha_bisection, FRAME_ALPHA_CAP};
pub use detect::{detect_from_probe_frames, detect_on_frame, VideoDetection};
pub use error::{Result, VideoError};
pub use frame::{map_to_template, remove_on_frame};
pub use maps::{diamond_map_1080p_standard, diamond_map_720p_compact, diamond_map_720p_standard, VideoMap};
pub use pipeline::{ffmpeg_available, remove_video};

/// Which watermark family to look for / remove.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkKind {
    /// Probe and pick diamond or Veo automatically.
    Auto,
    /// Gemini diamond / sparkle-style mark.
    Diamond,
    /// Veo-style mark.
    Veo,
}

/// Options for the video removal pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoRemoveOptions {
    /// Mark family to target.
    pub mark: MarkKind,
    /// Use legacy / compact geometry when true.
    pub legacy: bool,
    /// Force a fixed alpha scale (maps to GWT `--veo-alpha`); `None` = adaptive.
    pub force_alpha: Option<f32>,
}

impl Default for VideoRemoveOptions {
    fn default() -> Self {
        Self {
            mark: MarkKind::Auto,
            legacy: false,
            force_alpha: None,
        }
    }
}

/// Summary returned after a successful video remove run.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoRemoveResult {
    /// Frames that received reverse-blend.
    pub frames_processed: u32,
    /// Frames skipped (e.g. detect miss / no-op).
    pub frames_skipped: u32,
    /// Resolved mark kind after Auto detection when applicable.
    pub mark: MarkKind,
    /// Detected region `(x, y, w, h)` when known.
    pub region: (u32, u32, u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_options_default() {
        let opts = VideoRemoveOptions::default();
        assert_eq!(opts.mark, MarkKind::Auto);
        assert!(!opts.legacy);
        assert!(opts.force_alpha.is_none());
    }
}
