//! Quality-v3: worst clips + one easy → rust-crate-test/quality-v3/
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
    let out_dir = PathBuf::from("/workspace/rust-crate-test/quality-v3");
    fs::create_dir_all(&out_dir).unwrap();

    // Worst textured fails + one easier clip (57a3334c).
    let clips = [
        "8e9d4f7735ac7fd5b9146c9a125904214eaf5bf278369cc79c632e801837d53b.mp4", // fabric bands — worst plastic box
        "c2dd4416b212e6fd123505e863b0bf41fb61a8ab8de861b8d23a8f0777bd16ad.mp4", // skin/soft — tip ghosts
        "06d214682d2c97b047e6fde40a88559d89e088adeeced3ab01f261edefac9d42.mp4", // fabric + veo ref
        "57a3334c2bace6213805d6bff9b256f57e5d58f79126713db2eff713a8a815e0.mp4", // easier
        "f45c36e451687c894b8e6db53027e734f5670b874baa2293f1e51a47d723f87f.mp4", // veo/gwt ref
    ];

    let opts = VideoRemoveOptions::default();
    let mut log = String::new();
    for name in clips {
        let inp = format!("{att}/{name}");
        let short = id8(&inp).to_string();
        let out = out_dir.join(format!("{short}_rust_new.mp4"));
        eprintln!("=== {short} ===");
        let t0 = Instant::now();
        match remove_video(&inp, &out, &opts) {
            Ok(r) => {
                let elapsed = t0.elapsed().as_secs_f64();
                let fps = r.frames_processed as f64 / elapsed.max(1e-6);
                let line = format!(
                    "OK {short} frames={} mark={:?} region={:?} elapsed={elapsed:.3}s fps={fps:.2}\n",
                    r.frames_processed, r.mark, r.region
                );
                eprint!("{line}");
                log.push_str(&line);
            }
            Err(e) => {
                let line = format!("FAIL {short}: {e}\n");
                eprint!("{line}");
                log.push_str(&line);
            }
        }
    }
    fs::write(out_dir.join("run.log"), &log).unwrap();
    eprintln!("Wrote {}", out_dir.join("run.log").display());
}
