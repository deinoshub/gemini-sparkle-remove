//! Errors for the optional video watermark removal pipeline.

use std::fmt;

/// Errors produced while detecting or removing video watermarks.
#[derive(Debug)]
pub enum VideoError {
    /// ffmpeg / ffprobe CLI failure (message includes stderr when available).
    Ffmpeg(String),
    /// Local filesystem or temp-dir I/O failure.
    Io(std::io::Error),
    /// Mark detection failed or produced an unusable result.
    Detect(String),
    /// Caller-supplied path / options / container was invalid.
    InvalidInput(String),
}

/// Result alias for video APIs.
pub type Result<T> = std::result::Result<T, VideoError>;

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::Ffmpeg(msg) => write!(f, "ffmpeg error: {msg}"),
            VideoError::Io(err) => write!(f, "I/O error: {err}"),
            VideoError::Detect(msg) => write!(f, "detection error: {msg}"),
            VideoError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for VideoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VideoError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VideoError {
    fn from(err: std::io::Error) -> Self {
        VideoError::Io(err)
    }
}
