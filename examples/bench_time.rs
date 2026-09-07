
use gemini_unmark::video::{remove_video, VideoRemoveOptions};
use std::time::Instant;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let input = &args[1];
    let output = &args[2];
    let t0 = Instant::now();
    let r = remove_video(input, output, &VideoRemoveOptions::default()).expect("remove");
    let dt = t0.elapsed();
    println!("ok frames={} skipped={:?} region={:?} elapsed_sec={:.3} fps={:.2}",
        r.frames_processed, r.frames_skipped, r.region,
        dt.as_secs_f64(), r.frames_processed as f64 / dt.as_secs_f64());
}
