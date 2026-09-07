//! Process the two new2 user clips with remove_video.
use gemini_sparkle_remove::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::path::Path;
use std::time::Instant;

fn main() {
    assert!(ffmpeg_available(), "ffmpeg/ffprobe required");
    let pairs = [
        (
            "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments/f45c36e451687c894b8e6db53027e734f5670b874baa2293f1e51a47d723f87f.mp4",
            "/workspace/rust-crate-test/new2v2/f45c36e4_rust.mp4",
            "720p",
        ),
        (
            "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments/06d214682d2c97b047e6fde40a88559d89e088adeeced3ab01f261edefac9d42.mp4",
            "/workspace/rust-crate-test/new2v2/06d21468_rust.mp4",
            "1080p",
        ),
    ];
    let opts = VideoRemoveOptions::default();
    for (inp, out, label) in pairs {
        println!("=== {label} {inp} ===");
        let t = Instant::now();
        match remove_video(Path::new(inp), Path::new(out), &opts) {
            Ok(r) => println!(
                "OK frames={} mark={:?} region={:?} elapsed={:.1}s -> {out}",
                r.frames_processed,
                r.mark,
                r.region,
                t.elapsed().as_secs_f64()
            ),
            Err(e) => println!("FAIL elapsed={:.1}s: {e}", t.elapsed().as_secs_f64()),
        }
    }
}
