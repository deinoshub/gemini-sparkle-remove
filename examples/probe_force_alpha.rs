//! Run `remove_video` with a fixed alpha scale (skips adaptive lock).
//!
//! ```text
//! cargo run --release --example probe_force_alpha --features video -- <input.mp4> <output.mp4> <alpha>
//! ```

use gemini_unmark::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    if !ffmpeg_available() {
        eprintln!("ffmpeg/ffprobe not available");
        return ExitCode::FAILURE;
    }
    let mut args = env::args().skip(1);
    let (Some(inp), Some(out), Some(alpha_s)) = (args.next(), args.next(), args.next()) else {
        eprintln!("usage: probe_force_alpha <input.mp4> <output.mp4> <alpha>");
        return ExitCode::from(2);
    };
    let Ok(alpha) = alpha_s.parse::<f32>() else {
        eprintln!("invalid alpha: {alpha_s}");
        return ExitCode::from(2);
    };
    let mut opts = VideoRemoveOptions::default();
    opts.force_alpha = Some(alpha);
    let t = Instant::now();
    match remove_video(Path::new(&inp), &PathBuf::from(&out), &opts) {
        Ok(r) => {
            eprintln!(
                "force_alpha={alpha} frames={} region={:?} {:.2}s",
                r.frames_processed,
                r.region,
                t.elapsed().as_secs_f64()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("remove_video failed: {e}");
            ExitCode::FAILURE
        }
    }
}
