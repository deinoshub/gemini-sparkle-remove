//! Remove watermark from the sample clip and print BR metrics vs original / GWT.
use gemini_unmark::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::path::Path;
use std::process::Command;

fn main() {
    assert!(ffmpeg_available(), "ffmpeg/ffprobe required");
    let input = Path::new("/workspace/video-in/input.mp4");
    let output = Path::new("/workspace/rust-crate-test/video/cleaned_smooth.mp4");
    std::fs::create_dir_all(output.parent().unwrap()).unwrap();
    let opts = VideoRemoveOptions::default();
    let r = remove_video(input, output, &opts).expect("remove_video");
    println!(
        "ok frames={} mark={:?} region={:?} -> {}",
        r.frames_processed,
        r.mark,
        r.region,
        output.display()
    );

    let mid = 120u32;
    let tmp = tempfile::tempdir().unwrap();
    for (label, src) in [
        ("orig", input),
        ("out", output),
        ("gwt", Path::new("/workspace/video-out/cleaned.mp4")),
    ] {
        let png = tmp.path().join(format!("{label}.png"));
        let st = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                src.to_str().unwrap(),
                "-vf",
                &format!("select=eq(n\\,{mid})"),
                "-vframes",
                "1",
                png.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(st.success());
    }
    let (bx, by, bw, bh) = r.region;
    let orig = image::open(tmp.path().join("orig.png")).unwrap().to_rgba8();
    let out = image::open(tmp.path().join("out.png")).unwrap().to_rgba8();
    let gwt = image::open(tmp.path().join("gwt.png")).unwrap().to_rgba8();
    let d_orig = mean_abs(orig.as_raw(), out.as_raw(), orig.width(), bx, by, bw, bh);
    let d_gwt_base = mean_abs(orig.as_raw(), gwt.as_raw(), orig.width(), bx, by, bw, bh);
    let d_vs_gwt = mean_abs(out.as_raw(), gwt.as_raw(), orig.width(), bx, by, bw, bh);
    println!(
        "BR mid-frame: vs_orig={d_orig:.2} GWT_vs_orig={d_gwt_base:.2} ours_vs_GWT={d_vs_gwt:.2} pct_of_GWT={:.1}%",
        100.0 * d_orig / d_gwt_base
    );
}

fn mean_abs(a: &[u8], b: &[u8], stride: u32, x: u32, y: u32, rw: u32, rh: u32) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u64;
    let sw = stride as usize;
    for py in 0..rh as usize {
        for px in 0..rw as usize {
            let o = ((y as usize + py) * sw + (x as usize + px)) * 4;
            for c in 0..3 {
                sum += (a[o + c] as f64 - b[o + c] as f64).abs();
                n += 1;
            }
        }
    }
    sum / n as f64
}
