//! Build script:
//! - feature `video`: download static ffmpeg+ffprobe for TARGET into
//!   `third_party/ffmpeg/<version>/<triple>/` and emit `GSR_FFMPEG` / `GSR_FFPROBE`.
//! - feature `system-ffmpeg` (with `video`): skip download; runtime uses
//!   `GSR_FFMPEG`/`GSR_FFPROBE` then PATH only. Env `GSR_SKIP_FFMPEG_DOWNLOAD=1`
//!   is an extra override that also skips download.
//! - feature `video-fdncnn`: link bundled static libncnn + C++ Option shim.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Cache key / documented bundle id (BtbN master “latest” GPL assets, or
/// evermeet/osxexperts pins for macOS). Update COMMIT.txt + README when bumping.
const FFMPEG_BUNDLE_VERSION: &str = "btbn-master-2026-09";

fn main() {
    if env::var("CARGO_FEATURE_VIDEO").is_ok() {
        setup_ffmpeg();
    }

    if env::var("CARGO_FEATURE_VIDEO_FDNCNN").is_ok() {
        setup_ncnn();
    }
}

fn setup_ncnn() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ncnn = manifest.join("third_party/ncnn");
    let lib = ncnn.join("lib/libncnn.a");
    if !lib.is_file() {
        panic!(
            "missing {} — run scripts/build_ncnn_static.sh first",
            lib.display()
        );
    }

    println!(
        "cargo:rerun-if-changed={}",
        ncnn.join("lib/libncnn.a").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        ncnn.join("shim/gwr_ncnn_opt.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        ncnn.join("shim/gwr_ncnn_opt.h").display()
    );

    cc::Build::new()
        .cpp(true)
        .file(ncnn.join("shim/gwr_ncnn_opt.cpp"))
        .include(ncnn.join("include"))
        .include(ncnn.join("shim"))
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("gwr_ncnn_opt");

    println!(
        "cargo:rustc-link-search=native={}",
        ncnn.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=ncnn");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn setup_ffmpeg() {
    println!("cargo:rerun-if-env-changed=GSR_SKIP_FFMPEG_DOWNLOAD");
    println!("cargo:rerun-if-changed=GSR_FFMPEG");
    println!("cargo:rerun-if-changed=GSR_FFPROBE");

    // Feature `system-ffmpeg`: never download and never emit vendored paths.
    // Runtime uses GSR_FFMPEG/GSR_FFPROBE then PATH only.
    if env::var("CARGO_FEATURE_SYSTEM_FFMPEG").is_ok() {
        println!(
            "cargo:warning=feature `system-ffmpeg` — skipping ffmpeg download;              runtime uses GSR_FFMPEG/GSR_FFPROBE then PATH"
        );
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target = env::var("TARGET").unwrap_or_else(|_| env::var("HOST").unwrap());
    let cache_root = manifest
        .join("third_party/ffmpeg")
        .join(FFMPEG_BUNDLE_VERSION)
        .join(&target);

    let (ffmpeg_name, ffprobe_name) = if target.contains("windows") {
        ("ffmpeg.exe", "ffprobe.exe")
    } else {
        ("ffmpeg", "ffprobe")
    };
    let ffmpeg_path = cache_root.join(ffmpeg_name);
    let ffprobe_path = cache_root.join(ffprobe_name);

    let skip = env::var("GSR_SKIP_FFMPEG_DOWNLOAD").ok().as_deref() == Some("1");

    if ffmpeg_path.is_file() && ffprobe_path.is_file() {
        emit_ffmpeg_env(&ffmpeg_path, &ffprobe_path);
        return;
    }

    if skip {
        println!(
            "cargo:warning=GSR_SKIP_FFMPEG_DOWNLOAD=1 — not downloading ffmpeg;              runtime will use PATH or fail if missing"
        );
        return;
    }

    let Some(spec) = ffmpeg_download_spec(&target) else {
        println!(
            "cargo:warning=no static ffmpeg bundle mapped for target {target}; \
             install ffmpeg/ffprobe on PATH or set GSR_FFMPEG / GSR_FFPROBE"
        );
        return;
    };

    if let Err(e) = download_ffmpeg_bundle(&cache_root, &spec, ffmpeg_name, ffprobe_name) {
        println!(
            "cargo:warning=ffmpeg download failed ({e}); \
             runtime will fall back to PATH (set GSR_SKIP_FFMPEG_DOWNLOAD=1 to silence)"
        );
        return;
    }

    if ffmpeg_path.is_file() && ffprobe_path.is_file() {
        emit_ffmpeg_env(&ffmpeg_path, &ffprobe_path);
    }
}

fn emit_ffmpeg_env(ffmpeg: &Path, ffprobe: &Path) {
    println!("cargo:rustc-env=GSR_FFMPEG={}", ffmpeg.display());
    println!("cargo:rustc-env=GSR_FFPROBE={}", ffprobe.display());
    println!("cargo:rerun-if-changed={}", ffmpeg.display());
    println!("cargo:rerun-if-changed={}", ffprobe.display());
}

struct DownloadSpec {
    /// Human label for logs / COMMIT docs.
    label: &'static str,
    /// One or more URLs to fetch (archive or single binary zip).
    urls: Vec<&'static str>,
    kind: ArchiveKind,
}

#[derive(Clone, Copy)]
enum ArchiveKind {
    /// BtbN-style: tar.xz / zip containing `bin/ffmpeg` (+ ffprobe).
    BtbN,
    /// evermeet: separate zips each containing a single `ffmpeg` or `ffprobe` binary.
    EvermeetPair,
    /// osxexperts-style: one zip with both binaries at root.
    ZipRootPair,
}

fn ffmpeg_download_spec(target: &str) -> Option<DownloadSpec> {
    // BtbN FFmpeg-Builds — GPL static (includes libx264 for encode).
    // Floating “latest” n7.1 assets; cache dir pins FFMPEG_BUNDLE_VERSION.
    // Docs: https://github.com/BtbN/FFmpeg-Builds
    match target {
        "x86_64-unknown-linux-gnu" => Some(DownloadSpec {
            label: "BtbN master linux64-gpl",
            urls: vec![
                "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
            ],
            kind: ArchiveKind::BtbN,
        }),
        "aarch64-unknown-linux-gnu" => Some(DownloadSpec {
            label: "BtbN master linuxarm64-gpl",
            urls: vec![
                "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
            ],
            kind: ArchiveKind::BtbN,
        }),
        "x86_64-pc-windows-msvc" | "x86_64-pc-windows-gnu" => Some(DownloadSpec {
            label: "BtbN master win64-gpl",
            urls: vec![
                "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
            ],
            kind: ArchiveKind::BtbN,
        }),
        // evermeet.cx — static Intel macOS builds (ffmpeg + ffprobe separate zips).
        "x86_64-apple-darwin" => Some(DownloadSpec {
            label: "evermeet.cx ffmpeg/ffprobe 7.1.1",
            urls: vec![
                "https://evermeet.cx/ffmpeg/ffmpeg-7.1.1.zip",
                "https://evermeet.cx/ffmpeg/ffprobe-7.1.1.zip",
            ],
            kind: ArchiveKind::EvermeetPair,
        }),
        // Apple Silicon static build (osxexperts).
        "aarch64-apple-darwin" => Some(DownloadSpec {
            label: "osxexperts.net ffmpeg 7.1 arm64",
            urls: vec!["https://www.osxexperts.net/ffmpeg71arm.zip"],
            kind: ArchiveKind::ZipRootPair,
        }),
        _ => None,
    }
}

fn download_ffmpeg_bundle(
    cache_root: &Path,
    spec: &DownloadSpec,
    ffmpeg_name: &str,
    ffprobe_name: &str,
) -> Result<(), String> {
    fs::create_dir_all(cache_root).map_err(|e| format!("mkdir {}: {e}", cache_root.display()))?;
    let staging = cache_root.join("_staging");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging).map_err(|e| format!("mkdir staging: {e}"))?;

    println!("cargo:warning=downloading ffmpeg tools ({})…", spec.label);

    match spec.kind {
        ArchiveKind::BtbN => {
            let url = spec.urls[0];
            let archive = staging.join("bundle");
            download_url(url, &archive)?;
            extract_archive(&archive, &staging)?;
            let ffmpeg_src = find_named_file(&staging, ffmpeg_name)
                .ok_or_else(|| format!("archive missing {ffmpeg_name}"))?;
            let ffprobe_src = find_named_file(&staging, ffprobe_name)
                .ok_or_else(|| format!("archive missing {ffprobe_name}"))?;
            fs::copy(&ffmpeg_src, cache_root.join(ffmpeg_name))
                .map_err(|e| format!("copy ffmpeg: {e}"))?;
            fs::copy(&ffprobe_src, cache_root.join(ffprobe_name))
                .map_err(|e| format!("copy ffprobe: {e}"))?;
        }
        ArchiveKind::EvermeetPair => {
            if spec.urls.len() < 2 {
                return Err("evermeet spec needs two URLs".into());
            }
            let fz = staging.join("ffmpeg.zip");
            let pz = staging.join("ffprobe.zip");
            download_url(spec.urls[0], &fz)?;
            download_url(spec.urls[1], &pz)?;
            run_cmd("unzip", &["-o", "-q", fz.to_str().unwrap(), "-d", staging.to_str().unwrap()])?;
            run_cmd("unzip", &["-o", "-q", pz.to_str().unwrap(), "-d", staging.to_str().unwrap()])?;
            let ffmpeg_src = find_named_file(&staging, "ffmpeg")
                .ok_or_else(|| "evermeet zip missing ffmpeg".to_string())?;
            let ffprobe_src = find_named_file(&staging, "ffprobe")
                .ok_or_else(|| "evermeet zip missing ffprobe".to_string())?;
            fs::copy(&ffmpeg_src, cache_root.join(ffmpeg_name))
                .map_err(|e| format!("copy ffmpeg: {e}"))?;
            fs::copy(&ffprobe_src, cache_root.join(ffprobe_name))
                .map_err(|e| format!("copy ffprobe: {e}"))?;
        }
        ArchiveKind::ZipRootPair => {
            let url = spec.urls[0];
            let archive = staging.join("bundle.zip");
            download_url(url, &archive)?;
            run_cmd(
                "unzip",
                &[
                    "-o",
                    "-q",
                    archive.to_str().unwrap(),
                    "-d",
                    staging.to_str().unwrap(),
                ],
            )?;
            let ffmpeg_src = find_named_file(&staging, "ffmpeg")
                .or_else(|| find_named_file(&staging, ffmpeg_name))
                .ok_or_else(|| "zip missing ffmpeg".to_string())?;
            let ffprobe_src = find_named_file(&staging, "ffprobe")
                .or_else(|| find_named_file(&staging, ffprobe_name))
                .ok_or_else(|| {
                    "zip missing ffprobe (osxexperts may ship ffmpeg only — use PATH ffprobe)"
                        .to_string()
                })?;
            fs::copy(&ffmpeg_src, cache_root.join(ffmpeg_name))
                .map_err(|e| format!("copy ffmpeg: {e}"))?;
            fs::copy(&ffprobe_src, cache_root.join(ffprobe_name))
                .map_err(|e| format!("copy ffprobe: {e}"))?;
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for name in [ffmpeg_name, ffprobe_name] {
            let p = cache_root.join(name);
            if p.is_file() {
                let mut perms = fs::metadata(&p).unwrap().permissions();
                perms.set_mode(0o755);
                let _ = fs::set_permissions(&p, perms);
            }
        }
    }

    let _ = fs::remove_dir_all(&staging);
    Ok(())
}

fn download_url(url: &str, dest: &Path) -> Result<(), String> {
    // Prefer curl (follows GitHub redirects); wget as fallback.
    match run_cmd(
        "curl",
        &[
            "-fL",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-o",
            dest.to_str().unwrap(),
            url,
        ],
    ) {
        Ok(()) => Ok(()),
        Err(curl_err) => run_cmd(
            "wget",
            &["-q", "-O", dest.to_str().unwrap(), url],
        )
        .map_err(|wget_err| format!("download {url}: curl: {curl_err}; wget: {wget_err}")),
    }
}

fn extract_archive(archive: &Path, dest: &Path) -> Result<(), String> {
    let name = archive.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let path = archive.to_str().unwrap();
    let dest_s = dest.to_str().unwrap();
    if name.ends_with(".tar.xz") || path.ends_with("bundle") {
        // Detect by magic / try tar xz first for BtbN linux; zip for windows.
        let xz = Command::new("tar")
            .args(["-xJf", path, "-C", dest_s])
            .status();
        if matches!(xz, Ok(s) if s.success()) {
            return Ok(());
        }
        let gz = Command::new("tar")
            .args(["-xzf", path, "-C", dest_s])
            .status();
        if matches!(gz, Ok(s) if s.success()) {
            return Ok(());
        }
        return run_cmd("unzip", &["-o", "-q", path, "-d", dest_s]);
    }
    if name.ends_with(".zip") {
        return run_cmd("unzip", &["-o", "-q", path, "-d", dest_s]);
    }
    // Unknown extension: try tar xz then unzip (BtbN linux archive saved as “bundle”).
    let xz = Command::new("tar")
        .args(["-xJf", path, "-C", dest_s])
        .status();
    if matches!(xz, Ok(s) if s.success()) {
        return Ok(());
    }
    run_cmd("unzip", &["-o", "-q", path, "-d", dest_s])
}

fn find_named_file(root: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).ok()?;
        for ent in entries.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|s| s.to_str()) == Some(name) {
                return Some(p);
            }
        }
    }
    None
}

fn run_cmd(bin: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(bin)
        .args(args)
        .status()
        .map_err(|e| format!("spawn {bin}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{bin} {:?} failed: {status}", args))
    }
}
