//! Reproduce liquid-diamond α footprint bug → rust-crate-test/bugfix-liquid/
use gemini_sparkle_remove::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    assert!(ffmpeg_available());
    let att = "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments";
    let out_dir = PathBuf::from("/workspace/rust-crate-test/bugfix-liquid");
    fs::create_dir_all(&out_dir).unwrap();

    let clips = [
        (
            "8e9d4f7735ac7fd5b9146c9a125904214eaf5bf278369cc79c632e801837d53b.mp4",
            "8e9d4f77",
        ),
        (
            "06d214682d2c97b047e6fde40a88559d89e088adeeced3ab01f261edefac9d42.mp4",
            "06d21468",
        ),
    ];
    let opts = VideoRemoveOptions::default();
    for (name, short) in clips {
        let inp = format!("{att}/{name}");
        let out = out_dir.join(format!("{short}_rust_fixed.mp4"));
        eprintln!("=== {short} ===");
        let t0 = Instant::now();
        match remove_video(&inp, &out, &opts) {
            Ok(r) => eprintln!(
                "OK {short} frames={} mark={:?} region={:?} {:.2}s",
                r.frames_processed,
                r.mark,
                r.region,
                t0.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("FAIL {short}: {e}"),
        }
    }
    for src in [
        "/workspace/rust-crate-test/quality-v3/8e9d4f77_gwt.mp4",
        "/workspace/rust-crate-test/bugfix-roi/8e9d4f77_gwt.mp4",
    ] {
        let gwt = PathBuf::from(src);
        if gwt.is_file() {
            let _ = fs::copy(&gwt, out_dir.join("8e9d4f77_gwt.mp4"));
            break;
        }
    }
}
