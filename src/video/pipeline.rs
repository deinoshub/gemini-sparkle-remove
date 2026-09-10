//! ffmpeg CLI pipeline: probe → rawvideo decode → detect → adaptive remove (rayon) → encode.
//!
//! Frames stay in memory as RGBA (no PNG extract/re-encode). FDnCNN (feature
//! `video-fdncnn`) runs in-process via bundled libncnn — no Python runtime.
//!
//! ffmpeg/ffprobe resolution order: `GUM_FFMPEG`/`GUM_FFPROBE` env (runtime
//! override) → build-time vendored paths (`option_env!`, skipped when feature
//! `system-ffmpeg`) → binaries named `ffmpeg`/`ffprobe` on `PATH`.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;

use rayon::prelude::*;

use super::alpha::refine_alpha_bisection;
use super::detect::{detect_from_probe_frames, VideoDetection};
#[cfg(not(feature = "video-fdncnn"))]
use super::frame::remove_on_frame;
#[cfg(feature = "video-fdncnn")]
use super::frame::remove_on_frame_blend_only;
use super::maps::{diamond_map_1080p_standard, diamond_map_720p_compact, diamond_map_720p_standard, VideoMap};
use super::{MarkKind, Result, VideoError, VideoRemoveOptions, VideoRemoveResult};

/// Number of evenly spaced frames used for multi-frame detect probe.
const PROBE_FRAME_COUNT: usize = 5;

/// Remove Gemini/Veo-style watermarks from a video file (mp4 → mp4).
///
/// Requires `ffmpeg` and `ffprobe` (build-time vendored under
/// `third_party/ffmpeg/…` unless feature `system-ffmpeg`, or on `PATH`).
/// Audio is copied from the input
/// (`-c:a copy`) when present. Decode/encode use ffmpeg **rawvideo** pipes so
/// frames are processed in memory (no PNG frame extract/re-encode).
pub fn remove_video(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    opts: &VideoRemoveOptions,
) -> Result<VideoRemoveResult> {
    let input = input.as_ref();
    let output = output.as_ref();

    if !input.is_file() {
        return Err(VideoError::InvalidInput(format!(
            "input is not a file: {}",
            input.display()
        )));
    }
    if matches!(opts.mark, MarkKind::Veo) {
        return Err(VideoError::Detect(
            "Veo mark removal is not available in phase 1".into(),
        ));
    }

    require_ffmpeg_tools()?;

    let probe = probe_input(input)?;
    if probe.width == 0 || probe.height == 0 {
        return Err(VideoError::InvalidInput(
            "ffprobe reported zero width/height".into(),
        ));
    }

    let mut frames = decode_frames_raw(input, probe.width, probe.height)?;
    let frame_count = frames.len() as u32;
    if frame_count == 0 {
        return Err(VideoError::InvalidInput(
            "no frames decoded from input".into(),
        ));
    }

    let probe_frames = select_probe_frames(&frames, probe.width, probe.height)?;
    let det = detect_from_probe_frames(&probe_frames).ok_or_else(|| {
        VideoError::Detect("no watermark detected on probe frames".into())
    })?;

    let map = select_map(&det, opts)?;

    // Phase 1: alpha intensity.
    // GWT video locks a per-shot constant ("seed-only") after dynamic seed;
    // we mirror that by refining on evenly spaced probe frames and taking the
    // median, then applying one scale to every frame (no per-frame flicker).
    let alpha_scale = if let Some(forced) = opts.force_alpha {
        forced.clamp(0.05, 2.0)
    } else {
        seed_alpha_locked(&frames, probe.width, probe.height, &det, &map)
    };
    eprintln!(
        "seed_alpha_locked scale={:.4} region=({},{},{},{})",
        alpha_scale, det.x, det.y, det.w, det.h
    );

    // Phase 2: reverse-blend (+ classical footprint fill only when FDnCNN is
    // unavailable). With `video-fdncnn`, GWT semantics are blend → FDnCNN; a
    // pre-denoise TELEA fill destroys under-mark texture and yields plastic ROI.
    let width = probe.width;
    let height = probe.height;
    frames.par_iter_mut().for_each(|rgba| {
        #[cfg(feature = "video-fdncnn")]
        remove_on_frame_blend_only(rgba, width, height, &det, &map, alpha_scale);
        #[cfg(not(feature = "video-fdncnn"))]
        remove_on_frame(rgba, width, height, &det, &map, alpha_scale);
    });
    let frames_processed = frames.len() as u32;

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    // Residual cleanup: with `video-fdncnn`, in-process NcnnDenoiser only
    // (GWT video is blend→AI; classical TELEA already ran inside remove_on_frame
    // when FDnCNN is disabled).
    #[cfg(feature = "video-fdncnn")]
    {
        if !fdncnn_postpass_raw(&mut frames, probe.width, probe.height, &det, &map) {
            eprintln!(
                "fdncnn_postpass_rust failed — check assets/video/fdncnn + third_party/ncnn"
            );
        }
    }

    encode_frames_raw(&frames, input, output, &probe)?;

    Ok(VideoRemoveResult {
        frames_processed,
        frames_skipped: 0,
        mark: det.mark,
        region: (det.x, det.y, det.w, det.h),
    })
}


/// GWT-style per-shot constant alpha: refine on a few evenly spaced frames,
/// return the median (stable across the clip).
///
/// Soft clips often lock ~0.90–0.93 while GWT operates near ~1.0, which leaves
/// a bright tip residual. Blind force-1.0 helps those but regresses already
/// well-locked smoke clips (e.g. 8e9d ≈ 0.98). Only lift clearly under-locked
/// seeds; leave near-1.0 locks alone.
fn seed_alpha_locked(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
) -> f32 {
    /// Below this median the seed is treated as under-locked vs GWT ~1.0.
    const UNDER_LOCK_THR: f32 = 0.95;
    /// Target scale for under-locked soft clips (GWT-like).
    const UNDER_LOCK_TARGET: f32 = 1.0;

    let n = frames.len();
    if n == 0 {
        return UNDER_LOCK_TARGET;
    }
    let samples = 5.min(n);
    let mut vals = Vec::with_capacity(samples);
    let mut prev: Option<f32> = None;
    for i in 0..samples {
        let idx = if samples == 1 {
            0
        } else {
            i * (n - 1) / (samples - 1)
        };
        let s = refine_alpha_bisection(&frames[idx], width, height, det, map, prev);
        prev = Some(s);
        vals.push(s);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = vals[vals.len() / 2];
    if median < UNDER_LOCK_THR {
        UNDER_LOCK_TARGET
    } else {
        median
    }
}

fn select_map(det: &VideoDetection, opts: &VideoRemoveOptions) -> Result<VideoMap> {
    if opts.legacy {
        if let Some(m) = diamond_map_720p_compact() {
            return Ok(m);
        }
    }
    // Prefer map matching detected geometry.
    if det.w == 44 && det.h == 44 {
        if let Some(m) = diamond_map_720p_compact() {
            return Ok(m);
        }
    }
    if det.w == 72 && det.h == 72 {
        return Ok(diamond_map_1080p_standard());
    }
    match det.mark {
        MarkKind::Diamond | MarkKind::Auto => Ok(diamond_map_720p_standard()),
        MarkKind::Veo => Err(VideoError::Detect(
            "Veo maps not available in phase 1".into(),
        )),
    }
}

#[derive(Clone, Debug)]
struct ProbeInfo {
    width: u32,
    height: u32,
    fps: f64,
    #[allow(dead_code)]
    duration: f64,
    has_audio: bool,
}

fn ffmpeg_tool_path(kind: &str) -> PathBuf {
    // 1) Runtime override
    let env_key = if kind == "ffmpeg" {
        "GUM_FFMPEG"
    } else {
        "GUM_FFPROBE"
    };
    if let Ok(p) = std::env::var(env_key) {
        let pb = PathBuf::from(&p);
        if pb.is_file() {
            return pb;
        }
    }
    // 2) Build-time vendored path from build.rs (not used with `system-ffmpeg`)
    #[cfg(not(feature = "system-ffmpeg"))]
    {
        let build_path = if kind == "ffmpeg" {
            option_env!("GUM_FFMPEG")
        } else {
            option_env!("GUM_FFPROBE")
        };
        if let Some(p) = build_path {
            let pb = PathBuf::from(p);
            if pb.is_file() {
                return pb;
            }
        }
    }
    // 3) PATH
    PathBuf::from(kind)
}

fn ffmpeg_bin() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| ffmpeg_tool_path("ffmpeg")).clone()
}

fn ffprobe_bin() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| ffmpeg_tool_path("ffprobe")).clone()
}


fn missing_ffmpeg_hint() -> &'static str {
    #[cfg(feature = "system-ffmpeg")]
    {
        "Feature `system-ffmpeg` is enabled (no build-time download).          Install ffmpeg/ffprobe on PATH, or set GUM_FFMPEG / GUM_FFPROBE to absolute paths."
    }
    #[cfg(not(feature = "system-ffmpeg"))]
    {
        "Build with feature `video` to download static tools, install on PATH,          or set GUM_FFMPEG / GUM_FFPROBE. Offline: GUM_SKIP_FFMPEG_DOWNLOAD=1 skips download;          or use `--features video,system-ffmpeg` for PATH-only."
    }
}

fn require_tool(bin: &Path, label: &str) -> Result<()> {
    let status = Command::new(bin)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| {
            VideoError::Ffmpeg(format!(
                "failed to spawn {label} ({}): {e}. {}",
                bin.display(),
                missing_ffmpeg_hint(),
            ))
        })?;
    if !status.success() {
        return Err(VideoError::Ffmpeg(format!(
            "{label} not usable at {} (exit {status})",
            bin.display()
        )));
    }
    Ok(())
}

fn require_ffmpeg_tools() -> Result<()> {
    require_tool(&ffmpeg_bin(), "ffmpeg")?;
    require_tool(&ffprobe_bin(), "ffprobe")?;
    Ok(())
}

fn probe_input(input: &Path) -> Result<ProbeInfo> {
    let input_s = path_str(input)?;

    let v = ffprobe_csv(&[
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "csv=p=0",
        input_s,
    ])?;
    // csv: width,height,r_frame_rate,avg_frame_rate  e.g. 1280,720,24/1,24/1
    let parts: Vec<&str> = v.trim().split(',').collect();
    if parts.len() < 3 {
        return Err(VideoError::Ffmpeg(format!(
            "unexpected ffprobe video csv: {v:?}"
        )));
    }
    let width: u32 = parts[0]
        .parse()
        .map_err(|_| VideoError::Ffmpeg(format!("bad width in {v:?}")))?;
    let height: u32 = parts[1]
        .parse()
        .map_err(|_| VideoError::Ffmpeg(format!("bad height in {v:?}")))?;
    let fps = parse_fps(parts[2]).or_else(|| parts.get(3).and_then(|s| parse_fps(s)));
    let fps = fps.ok_or_else(|| VideoError::Ffmpeg(format!("bad fps in {v:?}")))?;

    let dur_s = ffprobe_csv(&[
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        input_s,
    ])?;
    let duration: f64 = dur_s.trim().parse().unwrap_or(0.0);

    let audio_out = Command::new(ffprobe_bin())
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            input_s,
        ])
        .output()
        .map_err(|e| VideoError::Ffmpeg(format!("ffprobe audio: {e}")))?;
    let has_audio =
        audio_out.status.success() && !String::from_utf8_lossy(&audio_out.stdout).trim().is_empty();

    Ok(ProbeInfo {
        width,
        height,
        fps,
        duration,
        has_audio,
    })
}

fn parse_fps(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() || s == "0/0" || s == "N/A" {
        return None;
    }
    if let Some((a, b)) = s.split_once('/') {
        let num: f64 = a.parse().ok()?;
        let den: f64 = b.parse().ok()?;
        if den == 0.0 {
            return None;
        }
        Some(num / den)
    } else {
        s.parse().ok()
    }
}

fn ffprobe_csv(args: &[&str]) -> Result<String> {
    let out = Command::new(ffprobe_bin())
        .args(args)
        .output()
        .map_err(|e| VideoError::Ffmpeg(format!("ffprobe spawn: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(VideoError::Ffmpeg(format!(
            "ffprobe failed: {}",
            err.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn frame_nbytes(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| VideoError::InvalidInput("frame size overflow".into()))
}

/// Decode all video frames as RGBA via ffmpeg rawvideo stdout pipe.
fn decode_frames_raw(input: &Path, width: u32, height: u32) -> Result<Vec<Vec<u8>>> {
    let input_s = path_str(input)?;
    let nbytes = frame_nbytes(width, height)?;

    let mut child = Command::new(ffmpeg_bin())
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_s,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| VideoError::Ffmpeg(format!("ffmpeg decode spawn: {e}")))?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| VideoError::Ffmpeg("ffmpeg decode missing stdout".into()))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| VideoError::Ffmpeg("ffmpeg decode missing stderr".into()))?;

    let err_thr = thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });

    let mut frames = Vec::new();
    loop {
        let mut buf = vec![0u8; nbytes];
        match read_full(&mut stdout, &mut buf) {
            Ok(true) => frames.push(buf),
            Ok(false) => break,
            Err(e) => {
                let _ = child.kill();
                let err = err_thr.join().unwrap_or_default();
                return Err(VideoError::Ffmpeg(format!(
                    "ffmpeg decode read: {e}; stderr={}",
                    err.trim()
                )));
            }
        }
    }
    drop(stdout);

    let status = child
        .wait()
        .map_err(|e| VideoError::Ffmpeg(format!("ffmpeg decode wait: {e}")))?;
    let err = err_thr.join().unwrap_or_default();
    if !status.success() {
        return Err(VideoError::Ffmpeg(format!(
            "ffmpeg decode failed: {}",
            err.trim()
        )));
    }
    Ok(frames)
}

/// Read exactly `buf.len()` bytes, or return Ok(false) on clean EOF before any byte.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..])? {
            0 => {
                if got == 0 {
                    return Ok(false);
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("short rawvideo frame: got {got} of {}", buf.len()),
                ));
            }
            n => got += n,
        }
    }
    Ok(true)
}

fn select_probe_frames(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
) -> Result<Vec<(u32, u32, Vec<u8>)>> {
    let frame_count = frames.len() as u32;
    let n = PROBE_FRAME_COUNT.min(frames.len()).max(1);
    let nbytes = frame_nbytes(width, height)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let idx = if n == 1 {
            0usize
        } else {
            (((i as u32) * (frame_count - 1)) / ((n as u32) - 1)) as usize
        };
        let rgba = &frames[idx];
        if rgba.len() != nbytes {
            return Err(VideoError::InvalidInput(format!(
                "frame {idx} size {} != expected {nbytes}",
                rgba.len()
            )));
        }
        out.push((width, height, rgba.clone()));
    }
    Ok(out)
}

/// FDnCNN postpass: in-process ncnn only (no Python).
#[cfg(feature = "video-fdncnn")]
fn fdncnn_postpass_raw(
    frames: &mut [Vec<u8>],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
) -> bool {
    if super::fdncnn::fdncnn_postpass_inprocess(frames, width, height, det, map) {
        eprintln!(
            "fdncnn_postpass_rust frames={} sigma=75 strength=180% pad=64 region=({},{},{},{})",
            frames.len(),
            det.x,
            det.y,
            map.width,
            map.height
        );
        return true;
    }
    false
}

fn encode_frames_raw(
    frames: &[Vec<u8>],
    input: &Path,
    output: &Path,
    probe: &ProbeInfo,
) -> Result<()> {
    let input_s = path_str(input)?;
    let output_s = path_str(output)?;
    let fps = format!("{:.6}", probe.fps);
    let size = format!("{}x{}", probe.width, probe.height);
    let nbytes = frame_nbytes(probe.width, probe.height)?;

    let mut args: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-y".into(),
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "rgba".into(),
        "-s".into(),
        size,
        "-framerate".into(),
        fps,
        "-i".into(),
        "pipe:0".into(),
        "-i".into(),
        input_s.to_string(),
        "-map".into(),
        "0:v:0".into(),
    ];
    if probe.has_audio {
        args.extend([
            "-map".into(),
            "1:a:0".into(),
            "-c:a".into(),
            "copy".into(),
        ]);
    }
    args.extend([
        "-c:v".into(),
        "libx264".into(),
        "-pix_fmt".into(),
        "yuv420p".into(),
        "-crf".into(),
        "18".into(),
        "-movflags".into(),
        "+faststart".into(),
        output_s.to_string(),
    ]);

    let mut child = Command::new(ffmpeg_bin())
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| VideoError::Ffmpeg(format!("ffmpeg encode spawn: {e}")))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| VideoError::Ffmpeg("ffmpeg encode missing stdin".into()))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| VideoError::Ffmpeg("ffmpeg encode missing stderr".into()))?;
    let err_thr = thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });

    for (i, frame) in frames.iter().enumerate() {
        if frame.len() != nbytes {
            let _ = child.kill();
            let _ = err_thr.join();
            return Err(VideoError::InvalidInput(format!(
                "encode frame {i} size {} != {nbytes}",
                frame.len()
            )));
        }
        if let Err(e) = stdin.write_all(frame) {
            let _ = child.kill();
            let err = err_thr.join().unwrap_or_default();
            return Err(VideoError::Ffmpeg(format!(
                "ffmpeg encode write: {e}; stderr={}",
                err.trim()
            )));
        }
    }
    drop(stdin);

    let status = child
        .wait()
        .map_err(|e| VideoError::Ffmpeg(format!("ffmpeg encode wait: {e}")))?;
    let err = err_thr.join().unwrap_or_default();
    if !status.success() {
        return Err(VideoError::Ffmpeg(format!(
            "ffmpeg encode failed: {}",
            err.trim()
        )));
    }
    if !output.is_file() {
        return Err(VideoError::Ffmpeg(
            "ffmpeg encode produced no output file".into(),
        ));
    }
    Ok(())
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| VideoError::InvalidInput(format!("non-UTF8 path: {}", path.display())))
}

/// Whether `ffmpeg` and `ffprobe` are usable (vendored build paths or PATH).
pub fn ffmpeg_available() -> bool {
    require_ffmpeg_tools().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_input() -> PathBuf {
        PathBuf::from("/workspace/video-in/input.mp4")
    }

    fn gwt_cleaned() -> PathBuf {
        PathBuf::from("/workspace/video-out/cleaned.mp4")
    }

    fn br_mean_abs_diff(a: &[u8], b: &[u8], stride: u32, x: u32, y: u32, rw: u32, rh: u32) -> f64 {
        let mut sum = 0.0f64;
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
        if n == 0 {
            0.0
        } else {
            sum / n as f64
        }
    }

    fn extract_one_png(video: &Path, frame_index: u32, out_png: &Path) -> Result<()> {
        require_ffmpeg_tools()?;
        let vs = path_str(video)?;
        let os = path_str(out_png)?;
        let sel = format!("select=eq(n\\,{frame_index})");
        let out = Command::new(ffmpeg_bin())
            .args([
                "-y", "-i", vs, "-vf", &sel, "-vframes", "1", "-pix_fmt", "rgba", os,
            ])
            .output()
            .map_err(|e| VideoError::Ffmpeg(format!("extract one: {e}")))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(VideoError::Ffmpeg(format!(
                "extract one failed: {}",
                err.trim()
            )));
        }
        Ok(())
    }

    fn probe_duration(path: &Path) -> Result<f64> {
        let s = ffprobe_csv(&[
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path_str(path)?,
        ])?;
        s.trim()
            .parse()
            .map_err(|_| VideoError::Ffmpeg(format!("bad duration {s:?}")))
    }

    #[test]
    fn remove_video_sample() {
        if !ffmpeg_available() {
            eprintln!("skip remove_video_sample: ffmpeg/ffprobe not available");
            return;
        }
        let input = sample_input();
        if !input.is_file() {
            eprintln!("skip remove_video_sample: missing {}", input.display());
            return;
        }

        let out_dir = tempfile::tempdir().expect("temp out");
        let output = out_dir.path().join("cleaned_rs.mp4");
        let opts = VideoRemoveOptions::default();

        let result = remove_video(&input, &output, &opts).expect("remove_video");
        assert!(
            result.frames_processed >= 200,
            "expected ~240 frames, got {}",
            result.frames_processed
        );
        assert_eq!(result.mark, MarkKind::Diamond);
        assert_eq!(result.region.2, 48);
        assert_eq!(result.region.3, 48);
        assert!(
            (result.region.0 as i32 - 1136).abs() <= 4,
            "x={}",
            result.region.0
        );
        assert!(
            (result.region.1 as i32 - 576).abs() <= 4,
            "y={}",
            result.region.1
        );

        let dur = probe_duration(&output).expect("duration");
        assert!(
            (9.5..10.6).contains(&dur),
            "duration {dur} not ~10s"
        );

        // Audio passthrough present.
        let audio = Command::new(ffprobe_bin())
            .args([
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                path_str(&output).unwrap(),
            ])
            .output()
            .expect("ffprobe audio");
        assert!(
            audio.status.success()
                && !String::from_utf8_lossy(&audio.stdout).trim().is_empty(),
            "output should retain audio"
        );

        // BR zoom mean-diff vs original mid-frame — watermark removal must change ROI.
        let mid = 120u32;
        let before_png = out_dir.path().join("before_mid.png");
        let after_png = out_dir.path().join("after_mid.png");
        extract_one_png(&input, mid, &before_png).expect("before");
        extract_one_png(&output, mid, &after_png).expect("after");
        let before = image::open(&before_png).unwrap().to_rgba8();
        let after = image::open(&after_png).unwrap().to_rgba8();
        let (bx, by, bw, bh) = result.region;
        let diff = br_mean_abs_diff(
            before.as_raw(),
            after.as_raw(),
            before.width(),
            bx,
            by,
            bw,
            bh,
        );
        assert!(
            diff > 1.5,
            "BR mean-abs-diff vs original too small ({diff}); watermark may remain"
        );

        // Loose compare to GWT baseline BR (same mid frame): our change should be
        // in the same ballpark as GWT's (at least ~80% of GWT mean-diff).
        let gwt = gwt_cleaned();
        if gwt.is_file() {
            let gwt_png = out_dir.path().join("gwt_mid.png");
            extract_one_png(&gwt, mid, &gwt_png).expect("gwt mid");
            let gwt_img = image::open(&gwt_png).unwrap().to_rgba8();
            let gwt_diff = br_mean_abs_diff(
                before.as_raw(),
                gwt_img.as_raw(),
                before.width(),
                bx,
                by,
                bw,
                bh,
            );
            assert!(
                diff >= gwt_diff * 0.8,
                "ours BR diff {diff} << GWT {gwt_diff}"
            );
            eprintln!(
                "remove_video_sample OK: frames={} dur={dur:.3}s BR_diff={diff:.2} GWT_BR_diff={gwt_diff:.2} region={:?}",
                result.frames_processed, result.region
            );
        } else {
            eprintln!(
                "remove_video_sample OK: frames={} dur={dur:.3}s BR_diff={diff:.2} (no GWT baseline)",
                result.frames_processed
            );
        }
    }

    fn count_video_frames(path: &Path) -> Result<u32> {
        let s = ffprobe_csv(&[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            path_str(path)?,
        ])?;
        s.trim()
            .parse()
            .map_err(|_| VideoError::Ffmpeg(format!("bad frame count {s:?}")))
    }

    fn lavfi_audio_mp4(path: &Path, audio_secs: &str) -> Result<()> {
        let out = Command::new(ffmpeg_bin())
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("sine=f=440:d={audio_secs}"),
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x96:r=24:d=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                path_str(path)?,
            ])
            .output()
            .map_err(|e| VideoError::Ffmpeg(format!("lavfi src: {e}")))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(VideoError::Ffmpeg(format!(
                "lavfi src failed: {}",
                err.trim()
            )));
        }
        Ok(())
    }

    #[test]
    fn encode_keeps_all_frames_when_audio_is_shorter() {
        if !ffmpeg_available() {
            eprintln!("skip encode_keeps_all_frames_when_audio_is_shorter: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().expect("temp");
        let src = dir.path().join("src.mp4");
        lavfi_audio_mp4(&src, "1").expect("lavfi src");
        let w = 160u32;
        let h = 96u32;
        let n = 48u32;
        let nbytes = frame_nbytes(w, h).unwrap();
        let frames = vec![vec![0u8; nbytes]; n as usize];
        let probe = ProbeInfo {
            width: w,
            height: h,
            fps: 24.0,
            duration: 2.0,
            has_audio: true,
        };
        let out = dir.path().join("out.mp4");
        encode_frames_raw(&frames, &src, &out, &probe).expect("encode");
        let got = count_video_frames(&out).expect("count");
        assert_eq!(got, n, "expected {n} video frames, got {got}");
    }
}
