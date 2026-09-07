use gemini_unmark::cli::{classify_input, InputKind};
use std::path::Path;

#[test]
fn still_image_extensions_are_image() {
    for p in ["shot.png", "a.JPG", "b.jpeg", "c.webp", "d.bmp", "e.tif", "f.tiff"] {
        assert_eq!(
            classify_input(Path::new(p)).unwrap(),
            InputKind::Image,
            "{p}"
        );
    }
}

#[test]
fn video_extensions_are_video() {
    for p in ["clip.mp4", "a.MOV", "b.mkv", "c.webm", "d.avi"] {
        assert_eq!(
            classify_input(Path::new(p)).unwrap(),
            InputKind::Video,
            "{p}"
        );
    }
}

#[test]
fn unknown_extension_is_error() {
    let err = classify_input(Path::new("notes.txt")).unwrap_err();
    assert!(err.contains("txt") || err.contains("unsupported"), "{err}");
}
