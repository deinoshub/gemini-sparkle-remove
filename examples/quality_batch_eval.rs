//! quality-batch-eval: FDnCNN parity (43ed8bf defaults) on diverse originals.
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

fn probe_wh(path: &str) -> (u32, u32) {
    let out = Command::new("ffprobe")
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
        .expect("ffprobe");
    let s = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = s.trim().split(',').collect();
    let w = parts.first().and_then(|p| p.parse().ok()).unwrap_or(0);
    let h = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0);
    (w, h)
}

fn main() {
    assert!(ffmpeg_available(), "ffmpeg/ffprobe required");
    let att = "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments";
    let out_dir = PathBuf::from("/workspace/rust-crate-test/quality-batch-eval");
    fs::create_dir_all(&out_dir).unwrap();

    // Skip 8e9d (copied from quality-fdncnn-parity). Prefer ≥5 diverse clips.
    let clips: &[&str] = &[
        "06d214682d2c97b047e6fde40a88559d89e088adeeced3ab01f261edefac9d42.mp4", // 1080p fabric hard
        "f45c36e451687c894b8e6db53027e734f5670b874baa2293f1e51a47d723f87f.mp4", // 720p
        "57a3334c2bace6213805d6bff9b256f57e5d58f79126713db2eff713a8a815e0.mp4", // 720p
        "c2dd4416b212e6fd123505e863b0bf41fb61a8ab8de861b8d23a8f0777bd16ad.mp4", // 1080p soft skin
        "14dafdc0b0aeb88b9e50eb506dfaf0cf7ef37a2c6687d25b44c9881e73f24ffa.mp4", // 720p extra
    ];

    let opts = VideoRemoveOptions::default();
    let mut log = String::from("quality-batch-eval rust runs\n");
    for name in clips {
        let inp = format!("{att}/{name}");
        if !Path::new(&inp).exists() {
            eprintln!("MISSING {inp}");
            continue;
        }
        let short = id8(&inp);
        let out = out_dir.join(format!("{short}_rust.mp4"));
        let (w, h) = probe_wh(&inp);
        println!("=== {short} {w}x{h} → {} ===", out.display());
        let t = Instant::now();
        match remove_video(Path::new(&inp), &out, &opts) {
            Ok(r) => {
                let elapsed = t.elapsed().as_secs_f64();
                let fps = if elapsed > 0.0 {
                    r.frames_processed as f64 / elapsed
                } else {
                    0.0
                };
                let line = format!(
                    "OK\t{short}\t{w}x{h}\tframes={}\tregion={:?}\telapsed={elapsed:.3}\tfps={fps:.2}\n",
                    r.frames_processed, r.region
                );
                print!("{line}");
                log.push_str(&line);
            }
            Err(e) => {
                let elapsed = t.elapsed().as_secs_f64();
                let line = format!("FAIL\t{short}\t{elapsed:.3}\t{e}\n");
                eprint!("{line}");
                log.push_str(&line);
            }
        }
    }
    fs::write(out_dir.join("rust_run.log"), log).ok();
}
