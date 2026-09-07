//! Batch-run video-fdncnn on all user test clips → rust-crate-test/batch-all/
use gemini_sparkle_remove::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

fn id8(path: &str) -> &str {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| &s[..8.min(s.len())])
        .unwrap_or("unknown")
}

fn probe_meta(path: &str) -> (u32, u32, f64, u64) {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration,size",
            "-of",
            "csv=p=0",
            path,
        ])
        .output()
        .expect("ffprobe");
    let s = String::from_utf8_lossy(&out.stdout);
    // ffprobe may emit two lines: stream then format, or combined depending on version
    let mut w = 0u32;
    let mut h = 0u32;
    let mut dur = 0.0f64;
    let mut size = 0u64;
    for line in s.lines() {
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(a), Ok(b)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                if a > 16 && b > 16 {
                    w = a;
                    h = b;
                }
            }
        }
        for p in &parts {
            if let Ok(f) = p.parse::<f64>() {
                if f > 0.5 && f < 36000.0 && dur == 0.0 {
                    // prefer duration-looking floats
                    if f < 10000.0 {
                        dur = f;
                    }
                }
            }
            if let Ok(n) = p.parse::<u64>() {
                if n > 10_000 {
                    size = n;
                }
            }
        }
    }
    // fallback size from filesystem
    if size == 0 {
        size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    // duration from format if still 0
    if dur == 0.0 {
        let out2 = Command::new("ffprobe")
            .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
            .output()
            .ok();
        if let Some(o) = out2 {
            dur = String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse()
                .unwrap_or(0.0);
        }
    }
    if w == 0 || h == 0 {
        let out3 = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                path,
            ])
            .output()
            .ok();
        if let Some(o) = out3 {
            let line = String::from_utf8_lossy(&o.stdout);
            let parts: Vec<&str> = line.trim().split(',').collect();
            if parts.len() >= 2 {
                w = parts[0].parse().unwrap_or(0);
                h = parts[1].parse().unwrap_or(0);
            }
        }
    }
    (w, h, dur, size)
}

fn main() {
    assert!(ffmpeg_available(), "ffmpeg/ffprobe required");
    let att = "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments";
    let out_dir = PathBuf::from("/workspace/rust-crate-test/batch-all");
    fs::create_dir_all(&out_dir).unwrap();

    let clips: Vec<String> = [
        "a53c4db195f3e37c5e555ec267b4f3951492942b55f19ad8810364336dd727b7.mp4",
        "57a3334c2bace6213805d6bff9b256f57e5d58f79126713db2eff713a8a815e0.mp4",
        "c2dd4416b212e6fd123505e863b0bf41fb61a8ab8de861b8d23a8f0777bd16ad.mp4",
        "14dafdc0b0aeb88b9e50eb506dfaf0cf7ef37a2c6687d25b44c9881e73f24ffa.mp4",
        "74cdea4ab97f67da8bc160e99ab71c5afde42391573fe3c71baf4526f2fdccc8.mp4",
        "7146dcd4b1c4bdb97a4ce59a63b69375aa2dd29563aa0d7c8bb010f657a0415c.mp4",
        "f45c36e451687c894b8e6db53027e734f5670b874baa2293f1e51a47d723f87f.mp4",
        "06d214682d2c97b047e6fde40a88559d89e088adeeced3ab01f261edefac9d42.mp4",
        "8e9d4f7735ac7fd5b9146c9a125904214eaf5bf278369cc79c632e801837d53b.mp4",
    ]
    .iter()
    .map(|n| format!("{att}/{n}"))
    .filter(|p| Path::new(p).exists())
    .collect();

    println!("BATCH clips={} out={}", clips.len(), out_dir.display());
    let opts = VideoRemoveOptions::default();
    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "| id8 | res | dur_s | region | elapsed_s | fps | size_in | size_out | status |".into(),
    );
    rows.push("|-----|-----|-------|--------|-----------|-----|---------|----------|--------|".into());

    for inp in &clips {
        let short = id8(inp);
        let out = out_dir.join(format!("{short}_rust_fdn.mp4"));
        let (w, h, dur, size_in) = probe_meta(inp);
        println!("=== {short} {w}x{h} dur={dur:.2}s size_in={size_in} → {} ===", out.display());
        let t = Instant::now();
        let status = match remove_video(Path::new(inp), &out, &opts) {
            Ok(r) => {
                let elapsed = t.elapsed().as_secs_f64();
                let fps = if elapsed > 0.0 {
                    r.frames_processed as f64 / elapsed
                } else {
                    0.0
                };
                let size_out = fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
                let region = format!("{:?}", r.region);
                println!(
                    "OK frames={} skipped={} mark={:?} region={} elapsed={:.3}s fps={:.2}",
                    r.frames_processed, r.frames_skipped, r.mark, region, elapsed, fps
                );
                // machine line for post
                println!(
                    "ROW\t{short}\t{w}x{h}\t{dur:.3}\t{}\t{:.3}\t{:.2}\t{size_in}\t{size_out}\tok\tframes={}\tskipped={}",
                    region.replace('\t', " "),
                    elapsed,
                    fps,
                    r.frames_processed,
                    r.frames_skipped
                );
                rows.push(format!(
                    "| {short} | {w}x{h} | {dur:.2} | `{}` | {elapsed:.1} | {fps:.2} | {size_in} | {size_out} | ok |",
                    region
                ));
                "ok"
            }
            Err(e) => {
                let elapsed = t.elapsed().as_secs_f64();
                eprintln!("FAIL {short}: {e}");
                println!(
                    "ROW\t{short}\t{w}x{h}\t{dur:.3}\t-\t{elapsed:.3}\t0\t{size_in}\t0\tfail\t{e}"
                );
                rows.push(format!(
                    "| {short} | {w}x{h} | {dur:.2} | - | {elapsed:.1} | - | {size_in} | - | FAIL: {e} |"
                ));
                "fail"
            }
        };
        let _ = status;
    }

    let summary = format!(
        "# Batch FDnCNN (in-process) — all user test videos\n\n\
         Crate: gemini-sparkle-remove\n\
         Feature: `video-fdncnn` (NcnnDenoiser in-process)\n\
         Output: `/workspace/rust-crate-test/batch-all/`\n\
         Date: 2026-09-07 Asia/Saigon\n\n\
         {}\n",
        rows.join("\n")
    );
    fs::write(out_dir.join("SUMMARY.md"), &summary).unwrap();
    println!("Wrote {}", out_dir.join("SUMMARY.md").display());
}
