//! Time `remove_video` on a clip.
//!
//! ```text
//! cargo run --release --example bench_time --features video -- <input.mp4> <output.mp4>
//! ```

use gemini_unmark::video::{remove_video, VideoRemoveOptions};
use std::env;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let (Some(input), Some(output)) = (args.next(), args.next()) else {
        eprintln!("usage: bench_time <input.mp4> <output.mp4>");
        return ExitCode::from(2);
    };
    let t0 = Instant::now();
    let r = match remove_video(&input, &output, &VideoRemoveOptions::default()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("remove_video failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "ok frames={} skipped={:?} region={:?} elapsed_sec={:.3} fps={:.2}",
        r.frames_processed,
        r.frames_skipped,
        r.region,
        dt,
        r.frames_processed as f64 / dt
    );
    ExitCode::SUCCESS
}
