use gemini_unmark::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    assert!(ffmpeg_available());
    let args: Vec<String> = std::env::args().collect();
    let inp = &args[1];
    let out = PathBuf::from(&args[2]);
    let alpha: f32 = args[3].parse().unwrap();
    let mut opts = VideoRemoveOptions::default();
    opts.force_alpha = Some(alpha);
    let t = Instant::now();
    let r = remove_video(Path::new(inp), &out, &opts).unwrap();
    eprintln!("force_alpha={alpha} frames={} region={:?} {:.2}s", r.frames_processed, r.region, t.elapsed().as_secs_f64());
}
