use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Image,
    Video,
}

/// Classify a path as still image vs video from its extension.
pub fn classify_input(path: &Path) -> Result<InputKind, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff" => Ok(InputKind::Image),
        "mp4" | "mov" | "mkv" | "webm" | "avi" => Ok(InputKind::Video),
        other => {
            if other.is_empty() {
                Err("input path has no extension".into())
            } else {
                Err(format!("unsupported input extension '{other}'"))
            }
        }
    }
}
