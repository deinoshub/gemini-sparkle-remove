# gemini-unmark

[![CI](https://github.com/deinoshub/gemini-unmark/actions/workflows/ci.yml/badge.svg)](https://github.com/deinoshub/gemini-unmark/actions/workflows/ci.yml)

Rust library that removes Gemini visible sparkle watermarks from RGBA buffers via reverse alpha-blend.

Pure Rust runtime (no Python). MIT licensed.

Port of [long66666/gemini-watermark-remover](https://github.com/long66666/gemini-watermark-remover), plus GWT/Veo-style video ideas (diamond reverse-blend, adaptive alpha, FDnCNN ROI cleanup).

Repository: [deinoshub/gemini-unmark](https://github.com/deinoshub/gemini-unmark).

## Requirements

| Path | Needs |
|------|--------|
| Default (still images) | Rust toolchain (`cargo`) |
| Feature `video` | Rust + **ffmpeg** / **ffprobe** (downloaded at build time for common targets by default, or on `PATH`) |
| Feature `video` + `system-ffmpeg` | Same APIs; **no** build-time download — uses `GUM_FFMPEG`/`GUM_FFPROBE` then `PATH` only |
| Feature `video-fdncnn` | Above + C++ compiler. Downloads official Tencent/ncnn zip for TARGET (tag **`20260526`**). CMake only as fallback. |
| Feature `cli` | `video-fdncnn` + `system-ffmpeg` + `clap`. Binary: `gunmark`. |

No `python3`, OpenCV, or system ncnn Python bindings.

### Build-time ffmpeg/ffprobe

When `video` (or `video-fdncnn`) is enabled **without** `system-ffmpeg`, `build.rs` downloads a **static** ffmpeg+ffprobe build for the Cargo **TARGET** triple into:

`third_party/ffmpeg/<bundle-version>/<target-triple>/`

(gitignored; cached by version). Documented bundle id: **`btbn-master-2026-09`**.

| Target | Source |
|--------|--------|
| `x86_64-unknown-linux-gnu` | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) `ffmpeg-master-latest-linux64-gpl.tar.xz` |
| `aarch64-unknown-linux-gnu` | BtbN `ffmpeg-master-latest-linuxarm64-gpl.tar.xz` |
| `x86_64-pc-windows-msvc` / `gnu` | BtbN `ffmpeg-master-latest-win64-gpl.zip` |
| `x86_64-apple-darwin` | [evermeet.cx](https://evermeet.cx/ffmpeg/) ffmpeg/ffprobe **7.1.1** zips |
| `aarch64-apple-darwin` | [osxexperts.net](https://www.osxexperts.net/) `ffmpeg71arm.zip` |

Opt out of download (keep download as default for `video`):

```bash
cargo build --features video                  # downloads tools (default)
cargo build --features video,system-ffmpeg    # system PATH / GUM_* only
cargo build --features video-fdncnn,system-ffmpeg
cargo build --release --features cli --bin gunmark
```

Also: set `GUM_SKIP_FFMPEG_DOWNLOAD=1` to skip download without the feature (still uses an existing cache if present). Provide tools on `PATH`, or set `GUM_FFMPEG` / `GUM_FFPROBE` to absolute binary paths. If tools are missing, `remove_video` returns a clear error.

Runtime resolution order: env override → build-time vendored paths (unless `system-ffmpeg`) → `PATH`.

## Usage

### Path dependency

```toml
gemini-unmark = { path = "../gemini-unmark" }

# video (downloads ffmpeg tools at build when online):
gemini-unmark = { path = "../gemini-unmark", features = ["video"] }

# video without downloading ffmpeg (system PATH / GUM_* only):
gemini-unmark = { path = "../gemini-unmark", features = ["video", "system-ffmpeg"] }

# + in-process FDnCNN (cmake-builds ncnn for the target):
gemini-unmark = { path = "../gemini-unmark", features = ["video-fdncnn"] }
```

### CLI (feature = `"cli"`)

Needs **ffmpeg** / **ffprobe** on `PATH` (or `GUM_FFMPEG` / `GUM_FFPROBE`) and CMake the first time ncnn is built.

```bash
cargo run --release --features cli -- input.png output.png
cargo run --release --features cli -- input.mp4 output.mp4
cargo run --release --features cli -- input.mp4 output.mp4 --mark diamond --force-alpha 0.8
```

GitHub Releases attach prebuilt binaries for Linux x86_64, macOS aarch64, and Windows x86_64.

### Buffer API

RGBA, 4 bytes/pixel, row-major (`data.len() == width * height * 4`). You own encode/decode.

```rust
use gemini_unmark::{remove_at, remove_gemini_sparkle, RemoveResult, RgbaImage};

let mut img = RgbaImage {
    width,
    height,
    data: &mut rgba_bytes,
};
match remove_gemini_sparkle(&mut img) {
    RemoveResult::Removed { x, y } => { /* mark cleared at (x, y) */ }
    RemoveResult::NotFound => { /* buffer unchanged */ }
}

remove_at(&mut img, x, y);
```

### Video API (feature = `"video"`)

```rust
use gemini_unmark::video::{
    remove_video, MarkKind, VideoRemoveOptions,
};

let opts = VideoRemoveOptions {
    mark: MarkKind::Auto,   // or Diamond
    legacy: false,
    force_alpha: None,      // None = adaptive; Some(s) forces scale
};

let result = remove_video("input.mp4", "cleaned.mp4", &opts)?;
```

Build/test:

```bash
cargo test
cargo test --features video
cargo test --features video,system-ffmpeg
cargo test --features video-fdncnn,system-ffmpeg   # downloads official ncnn zip
cargo test --features cli
```

## CI and releases

GitHub Actions (`.github/workflows/ci.yml`) on `main` and pull requests:

| Job | Runner | What it runs |
|-----|--------|----------------|
| `test` | Linux, macOS, Windows | `cargo test` (default features) |
| `video` | Linux, macOS, Windows | download official ncnn zip, `cargo test --features video-fdncnn,system-ffmpeg`, release-build CLI |

The video job installs **system** `ffmpeg` / **ffprobe** and enables `system-ffmpeg` (no bundled ffmpeg download). FDnCNN uses the official ncnn release zip for that OS.

To cut a GitHub Release, push a tag matching `Cargo.toml` `version`:

```bash
git tag v0.2.3
git push origin v0.2.3
```

`.github/workflows/release.yml` packages the crate and attaches Linux/macOS/Windows CLI binaries plus `SHA256SUMS.txt` (no crates.io publish).

## License

MIT
