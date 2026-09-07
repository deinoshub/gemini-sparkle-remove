//! Build script:
//! - feature `video`: download static ffmpeg+ffprobe for TARGET into
//!   `third_party/ffmpeg/<version>/<triple>/` and emit `GUM_FFMPEG` / `GUM_FFPROBE`.
//! - feature `system-ffmpeg` (with `video`): skip download; runtime uses
//!   `GUM_FFMPEG`/`GUM_FFPROBE` then PATH only. Env `GUM_SKIP_FFMPEG_DOWNLOAD=1`
//!   is an extra override that also skips download.
//! - feature `video-fdncnn`: download official Tencent/ncnn static zip for
//!   TARGET (cmake-from-source fallback) and link the C++ Option shim.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

include!("src/native_link.rs");

/// Cache key / documented bundle id (BtbN master “latest” GPL assets, or
/// evermeet/osxexperts pins for macOS). Update COMMIT.txt + README when bumping.
const FFMPEG_BUNDLE_VERSION: &str = "btbn-master-2026-09";

/// Pinned Tencent/ncnn release tag. Override with env `NCNN_REV`.
const NCNN_REV: &str = "20260526";

fn main() {
    if env::var("CARGO_FEATURE_VIDEO").is_ok() {
        setup_ffmpeg();
    }

    if env::var("CARGO_FEATURE_VIDEO_FDNCNN").is_ok() {
        setup_ncnn();
    }
}

fn setup_ncnn() {
    println!("cargo:rerun-if-env-changed=NCNN_SRC");
    println!("cargo:rerun-if-env-changed=NCNN_REV");
    println!("cargo:rerun-if-env-changed=CMAKE");
    println!("cargo:rerun-if-env-changed=GUM_NCNN_FROM_SOURCE");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target = env::var("TARGET").unwrap_or_else(|_| env::var("HOST").unwrap());
    let spec = ncnn_link_spec(&target);
    let ncnn = manifest.join("third_party/ncnn");

    println!(
        "cargo:rerun-if-changed={}",
        ncnn.join("shim/gwr_ncnn_opt.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        ncnn.join("shim/gwr_ncnn_opt.h").display()
    );

    let from_source = env::var("GUM_NCNN_FROM_SOURCE").ok().as_deref() == Some("1");
    if !from_source {
        if let Some(pre) = ensure_ncnn_prebuilt(&manifest, &target) {
            compile_ncnn_shim(&ncnn, &[&pre.include]);
            link_ncnn_prebuilt(&pre, &target);
            return;
        }
    }

    let src = env::var("NCNN_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("third_party/ncnn-src"));
    let build_dir = src.join(format!("build-{target}"));
    let lib_path = find_named_under(&build_dir, spec.lib_filename).unwrap_or_else(|| {
        ensure_ncnn_built(&src, &build_dir);
        find_named_under(&build_dir, spec.lib_filename).unwrap_or_else(|| {
            panic!(
                "ncnn static lib {} not found under {} after cmake build",
                spec.lib_filename,
                build_dir.display()
            )
        })
    });

    let include_src = src.join("src");
    let include_gen = lib_path
        .parent()
        .map(|p| {
            if p.file_name().and_then(|s| s.to_str()) == Some("Release")
                || p.file_name().and_then(|s| s.to_str()) == Some("Debug")
            {
                p.parent().unwrap_or(p).to_path_buf()
            } else {
                p.to_path_buf()
            }
        })
        .unwrap_or_else(|| build_dir.join("src"));

    compile_ncnn_shim(&ncnn, &[&include_src, &include_gen]);
    let lib_dir = lib_path.parent().expect("ncnn lib parent");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ncnn");
    for extra in spec.extra_libs {
        emit_link_lib(extra.kind, extra.name);
    }
    println!("cargo:rerun-if-changed={}", lib_path.display());
}

struct NcnnPrebuilt {
    include: PathBuf,
    /// Directory to pass as `-L` (native) or framework search path.
    search: PathBuf,
    framework: bool,
}

fn compile_ncnn_shim(ncnn: &Path, includes: &[&Path]) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(ncnn.join("shim/gwr_ncnn_opt.cpp"))
        .include(ncnn.join("shim"))
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-Wno-unused-parameter");
    for include in includes {
        build.include(include);
    }
    build.compile("gwr_ncnn_opt");
}

fn emit_link_lib(kind: &str, name: &str) {
    if kind.is_empty() {
        println!("cargo:rustc-link-lib={name}");
    } else {
        println!("cargo:rustc-link-lib={kind}={name}");
    }
}

fn link_ncnn_prebuilt(pre: &NcnnPrebuilt, target: &str) {
    if pre.framework {
        println!("cargo:rustc-link-search=framework={}", pre.search.display());
        println!("cargo:rustc-link-lib=framework=ncnn");
        println!("cargo:rustc-link-lib=framework=openmp");
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-search=native={}", pre.search.display());
        println!("cargo:rustc-link-lib=static=ncnn");
        // Official zips are Vulkan-enabled; gpu.cpp needs the bundled glslang.
        for name in [
            "glslang",
            "MachineIndependent",
            "GenericCodeGen",
            "OSDependent",
            "SPIRV",
            "glslang-default-resource-limits",
        ] {
            println!("cargo:rustc-link-lib=static={name}");
        }
        if target.contains("windows") {
            if !target.contains("gnu") {
                println!("cargo:rustc-link-lib=dylib=vcomp");
            }
        } else {
            println!("cargo:rustc-link-lib=gomp");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=dl");
        }
    }
}

fn ncnn_prebuilt_asset(target: &str, rev: &str) -> Option<String> {
    if target == "x86_64-unknown-linux-gnu" {
        Some(format!("ncnn-{rev}-ubuntu-2404.zip"))
    } else if target.contains("apple-darwin") {
        Some(format!("ncnn-{rev}-macos.zip"))
    } else if target == "x86_64-pc-windows-msvc" || target == "aarch64-pc-windows-msvc" {
        Some(format!("ncnn-{rev}-windows-vs2022.zip"))
    } else {
        None
    }
}

fn ensure_ncnn_prebuilt(manifest: &Path, target: &str) -> Option<NcnnPrebuilt> {
    let rev = env::var("NCNN_REV").unwrap_or_else(|_| NCNN_REV.to_string());
    let asset = ncnn_prebuilt_asset(target, &rev)?;
    let dest = manifest
        .join("third_party/ncnn-prebuilt")
        .join(&rev)
        .join(target);
    if locate_prebuilt(&dest, target).is_none() {
        if let Err(e) = download_ncnn_prebuilt(&dest, &rev, &asset) {
            println!("cargo:warning=ncnn prebuilt download failed ({e}); falling back to cmake");
            return None;
        }
    }
    locate_prebuilt(&dest, target)
}

fn download_ncnn_prebuilt(dest: &Path, rev: &str, asset: &str) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| format!("mkdir {}: {e}", dest.display()))?;
    let url = format!("https://github.com/Tencent/ncnn/releases/download/{rev}/{asset}");
    let zip = dest.join("_dl.zip");
    println!("cargo:warning=downloading official ncnn {asset}…");
    download_url(&url, &zip)?;
    let dest_s = dest.to_str().unwrap();
    let zip_s = zip.to_str().unwrap();
    let unzipped = run_cmd("unzip", &["-o", "-q", zip_s, "-d", dest_s]).or_else(|_| {
        let cmd = format!(
            "Expand-Archive -Force '{}' '{}'",
            zip.display(),
            dest.display()
        );
        run_cmd("powershell", &["-NoProfile", "-Command", &cmd])
    });
    unzipped?;
    let _ = fs::remove_file(&zip);
    Ok(())
}

fn locate_prebuilt(root: &Path, target: &str) -> Option<NcnnPrebuilt> {
    if !root.exists() {
        return None;
    }
    if target.contains("apple") {
        let fw = find_dir_named(root, "ncnn.framework")?;
        let include = find_named_under(&fw, "c_api.h")?.parent()?.to_path_buf();
        let search = fw.parent()?.to_path_buf();
        return Some(NcnnPrebuilt {
            include,
            search,
            framework: true,
        });
    }
    let search_root = if target.contains("windows") {
        let arch = if target.starts_with("aarch64") {
            "arm64"
        } else {
            "x64"
        };
        find_dir_named(root, arch).unwrap_or_else(|| root.to_path_buf())
    } else {
        root.to_path_buf()
    };
    let include = find_named_under(&search_root, "c_api.h")?
        .parent()?
        .to_path_buf();
    let lib_name = if target.contains("windows") {
        "ncnn.lib"
    } else {
        "libncnn.a"
    };
    let lib = find_named_under(&search_root, lib_name)?;
    Some(NcnnPrebuilt {
        include,
        search: lib.parent()?.to_path_buf(),
        framework: false,
    })
}

fn find_dir_named(root: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut n = 0usize;
    while let Some(dir) = stack.pop() {
        n += 1;
        if n > 256 {
            break;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for ent in entries.flatten() {
            let p = ent.path();
            if p.is_dir() {
                if p.file_name().and_then(|s| s.to_str()) == Some(name) {
                    return Some(p);
                }
                stack.push(p);
            }
        }
    }
    None
}

fn find_named_under(root: &Path, filename: &str) -> Option<PathBuf> {
    if !root.exists() {
        return None;
    }
    let mut stack = vec![root.to_path_buf()];
    let mut depth = 0usize;
    while let Some(dir) = stack.pop() {
        depth += 1;
        if depth > 256 {
            break;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for ent in entries.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.file_name().and_then(|s| s.to_str()) == Some(filename) {
                return Some(p);
            }
        }
    }
    None
}

fn ensure_ncnn_built(src: &Path, build_dir: &Path) {
    ensure_ncnn_checkout(src);
    let cmake = cmake_bin();
    println!(
        "cargo:warning=building static CPU ncnn ({}) in {}",
        env::var("NCNN_REV").unwrap_or_else(|_| NCNN_REV.to_string()),
        build_dir.display()
    );
    fs::create_dir_all(build_dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", build_dir.display()));
    run_cmd(
        cmake.to_str().unwrap(),
        &[
            "-S",
            src.to_str().unwrap(),
            "-B",
            build_dir.to_str().unwrap(),
            "-DNCNN_BUILD_TOOLS=OFF",
            "-DNCNN_BUILD_EXAMPLES=OFF",
            "-DNCNN_VULKAN=OFF",
            "-DNCNN_SHARED_LIB=OFF",
            "-DNCNN_OPENMP=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
    )
    .unwrap_or_else(|e| panic!("cmake configure ncnn: {e}"));
    let jobs = env::var("CMAKE_BUILD_PARALLEL_LEVEL").unwrap_or_else(|_| "2".to_string());
    run_cmd(
        cmake.to_str().unwrap(),
        &[
            "--build",
            build_dir.to_str().unwrap(),
            "--config",
            "Release",
            "--parallel",
            &jobs,
        ],
    )
    .unwrap_or_else(|e| panic!("cmake build ncnn: {e}"));
}

fn ensure_ncnn_checkout(src: &Path) {
    if src.join("CMakeLists.txt").is_file() {
        return;
    }
    let rev = env::var("NCNN_REV").unwrap_or_else(|_| NCNN_REV.to_string());
    if let Some(parent) = src.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| panic!("mkdir {}: {e}", parent.display()));
    }
    let dest = src.to_str().unwrap();
    println!("cargo:warning=cloning Tencent/ncnn {rev} → {dest}");
    run_cmd(
        "git",
        &[
            "clone",
            "--depth",
            "1",
            "--branch",
            &rev,
            "https://github.com/Tencent/ncnn.git",
            dest,
        ],
    )
    .unwrap_or_else(|e| panic!("git clone ncnn: {e}"));
}

fn cmake_bin() -> PathBuf {
    if let Ok(p) = env::var("CMAKE") {
        return PathBuf::from(p);
    }
    for candidate in [
        "cmake",
        "/opt/homebrew/bin/cmake",
        "/usr/local/bin/cmake",
        "C:\\Program Files\\CMake\\bin\\cmake.exe",
    ] {
        let p = Path::new(candidate);
        if candidate == "cmake" {
            if Command::new("cmake")
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return PathBuf::from("cmake");
            }
            continue;
        }
        if p.is_file() {
            return p.to_path_buf();
        }
    }
    panic!("cmake not found — install CMake or set CMAKE to its path");
}

fn setup_ffmpeg() {
    println!("cargo:rerun-if-env-changed=GUM_SKIP_FFMPEG_DOWNLOAD");
    println!("cargo:rerun-if-env-changed=GUM_FFMPEG");
    println!("cargo:rerun-if-env-changed=GUM_FFPROBE");

    // Feature `system-ffmpeg`: never download and never emit vendored paths.
    // Runtime uses GUM_FFMPEG/GUM_FFPROBE then PATH only.
    if env::var("CARGO_FEATURE_SYSTEM_FFMPEG").is_ok() {
        println!(
            "cargo:warning=feature `system-ffmpeg` — skipping ffmpeg download;              runtime uses GUM_FFMPEG/GUM_FFPROBE then PATH"
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

    let skip = env::var("GUM_SKIP_FFMPEG_DOWNLOAD").ok().as_deref() == Some("1");

    if ffmpeg_path.is_file() && ffprobe_path.is_file() {
        emit_ffmpeg_env(&ffmpeg_path, &ffprobe_path);
        return;
    }

    if skip {
        println!(
            "cargo:warning=GUM_SKIP_FFMPEG_DOWNLOAD=1 — not downloading ffmpeg;              runtime will use PATH or fail if missing"
        );
        return;
    }

    let Some(spec) = ffmpeg_download_spec(&target) else {
        println!(
            "cargo:warning=no static ffmpeg bundle mapped for target {target}; \
             install ffmpeg/ffprobe on PATH or set GUM_FFMPEG / GUM_FFPROBE"
        );
        return;
    };

    if let Err(e) = download_ffmpeg_bundle(&cache_root, &spec, ffmpeg_name, ffprobe_name) {
        println!(
            "cargo:warning=ffmpeg download failed ({e}); \
             runtime will fall back to PATH (set GUM_SKIP_FFMPEG_DOWNLOAD=1 to silence)"
        );
        return;
    }

    if ffmpeg_path.is_file() && ffprobe_path.is_file() {
        emit_ffmpeg_env(&ffmpeg_path, &ffprobe_path);
    }
}

fn emit_ffmpeg_env(ffmpeg: &Path, ffprobe: &Path) {
    println!("cargo:rustc-env=GUM_FFMPEG={}", ffmpeg.display());
    println!("cargo:rustc-env=GUM_FFPROBE={}", ffprobe.display());
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
    // Inherit stdio so cmake progress shows in CI; do not buffer the full log
    // (that OOMs Ubuntu runners when ncnn compiles in parallel with rav1e).
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
