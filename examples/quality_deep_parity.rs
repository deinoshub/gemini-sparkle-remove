//! quality-deep-parity: weak3 + smoke vs 43ed8bf / GWT
use gemini_sparkle_remove::video::{ffmpeg_available, remove_video, VideoRemoveOptions};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn id8(path: &str) -> &str {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| &s[..8.min(s.len())])
        .unwrap_or("unknown")
}

fn main() {
    assert!(ffmpeg_available());
    let att = "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments";
    let out_dir = PathBuf::from("/workspace/rust-crate-test/quality-deep-parity");
    fs::create_dir_all(&out_dir).unwrap();
    let clips: &[&str] = &[
        "c2dd4416b212e6fd123505e863b0bf41fb61a8ab8de861b8d23a8f0777bd16ad.mp4",
        "14dafdc0b0aeb88b9e50eb506dfaf0cf7ef37a2c6687d25b44c9881e73f24ffa.mp4",
        "57a3334c2bace6213805d6bff9b256f57e5d58f79126713db2eff713a8a815e0.mp4",
        "8e9d4f7735ac7fd5b9146c9a125904214eaf5bf278369cc79c632e801837d53b.mp4",
        "06d214682d2c97b047e6fde40a88559d89e088adeeced3ab01f261edefac9d42.mp4",
    ];
    let opts = VideoRemoveOptions::default();
    let mut log = String::from("quality-deep-parity rust runs\n");
    for name in clips {
        let inp = format!("{att}/{name}");
        let short = id8(&inp);
        let out = out_dir.join(format!("{short}_new.mp4"));
        println!("=== {short} → {} ===", out.display());
        let t = Instant::now();
        match remove_video(Path::new(&inp), &out, &opts) {
            Ok(r) => {
                let elapsed = t.elapsed().as_secs_f64();
                let fps = r.frames_processed as f64 / elapsed.max(1e-6);
                let line = format!(
                    "OK\t{short}\tframes={}\tregion={:?}\telapsed={elapsed:.3}\tfps={fps:.2}\n",
                    r.frames_processed, r.region
                );
                print!("{line}");
                log.push_str(&line);
            }
            Err(e) => {
                let line = format!("FAIL\t{short}\t{e}\n");
                eprint!("{line}");
                log.push_str(&line);
            }
        }
    }
    fs::write(out_dir.join("rust_run.log"), log).ok();
}
