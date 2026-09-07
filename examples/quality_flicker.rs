//! quality-flicker: process 8e9d4f77 with temporal-stable FDnCNN path.
use gemini_unmark::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::path::Path;
use std::time::Instant;

fn main() {
    assert!(ffmpeg_available());
    let inp = "/home/box/sand-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments/8e9d4f7735ac7fd5b9146c9a125904214eaf5bf278369cc79c632e801837d53b.mp4";
    let out = Path::new("/workspace/rust-crate-test/quality-flicker/8e9d4f77_rust_fixed.mp4");
    let opts = VideoRemoveOptions::default();
    let t = Instant::now();
    match remove_video(Path::new(inp), out, &opts) {
        Ok(r) => println!(
            "OK frames={} mark={:?} region={:?} elapsed={:.2}s",
            r.frames_processed,
            r.mark,
            r.region,
            t.elapsed().as_secs_f64()
        ),
        Err(e) => {
            eprintln!("FAIL: {e}");
            std::process::exit(1);
        }
    }
}
