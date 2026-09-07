//! Process both user clips with video-fdncnn → rust-crate-test/fdncnn/.
use gemini_sparkle_remove::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::path::Path;
use std::time::Instant;

fn main() {
    assert!(ffmpeg_available(), "ffmpeg/ffprobe required");
    let out_dir = Path::new("/workspace/rust-crate-test/fdncnn");
    let pairs = [
        (
            "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments/f45c36e451687c894b8e6db53027e734f5670b874baa2293f1e51a47d723f87f.mp4",
            "f45c36e4_rust_fdn.mp4",
            "720p",
        ),
        (
            "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments/06d214682d2c97b047e6fde40a88559d89e088adeeced3ab01f261edefac9d42.mp4",
            "06d21468_rust_fdn.mp4",
            "1080p",
        ),
    ];
    let opts = VideoRemoveOptions::default();
    for (inp, name, label) in pairs {
        let out = out_dir.join(name);
        println!("=== {label} → {} ===", out.display());
        let t = Instant::now();
        match remove_video(Path::new(inp), &out, &opts) {
            Ok(r) => println!(
                "OK frames={} mark={:?} region={:?} elapsed={:.1}s",
                r.frames_processed,
                r.mark,
                r.region,
                t.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("FAIL: {e}"),
        }
    }
}
