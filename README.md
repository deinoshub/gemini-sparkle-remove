# gemini-sparkle-remove

[![CI](https://github.com/deinoshub/gemini-sparkle-remove/actions/workflows/ci.yml/badge.svg)](https://github.com/deinoshub/gemini-sparkle-remove/actions/workflows/ci.yml)

Rust library that removes Gemini visible sparkle watermarks from RGBA buffers via reverse alpha-blend.

Pure Rust runtime (no Python). MIT licensed.

Port of [long66666/gemini-watermark-remover](https://github.com/long66666/gemini-watermark-remover), plus GWT/Veo-style video ideas (diamond reverse-blend, adaptive alpha, FDnCNN ROI cleanup).

Repository: [deinoshub/gemini-sparkle-remove](https://github.com/deinoshub/gemini-sparkle-remove).

## Requirements

| Path | Needs |
|------|--------|
| Default (still images) | Rust toolchain (`cargo`) |
| Feature `video` | Rust + **ffmpeg** / **ffprobe** (downloaded at build time for common targets by default, or on `PATH`) |
| Feature `video` + `system-ffmpeg` | Same APIs; **no** build-time download — uses `GSR_FFMPEG`/`GSR_FFPROBE` then `PATH` only |
| Feature `video-fdncnn` | Above + bundled `third_party/ncnn` (`libncnn.a`; rebuild via `scripts/build_ncnn_static.sh`) |

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
cargo build --features video,system-ffmpeg    # system PATH / GSR_* only
cargo build --features video-fdncnn,system-ffmpeg
```

Also: set `GSR_SKIP_FFMPEG_DOWNLOAD=1` to skip download without the feature (still uses an existing cache if present). Provide tools on `PATH`, or set `GSR_FFMPEG` / `GSR_FFPROBE` to absolute binary paths. If tools are missing, `remove_video` returns a clear error.

Runtime resolution order: env override → build-time vendored paths (unless `system-ffmpeg`) → `PATH`.

## Usage

### Path dependency

```toml
gemini-sparkle-remove = { path = "../gemini-sparkle-remove" }

# video (downloads ffmpeg tools at build when online):
gemini-sparkle-remove = { path = "../gemini-sparkle-remove", features = ["video"] }

# video without downloading ffmpeg (system PATH / GSR_* only):
gemini-sparkle-remove = { path = "../gemini-sparkle-remove", features = ["video", "system-ffmpeg"] }

# + in-process FDnCNN (needs third_party/ncnn):
gemini-sparkle-remove = { path = "../gemini-sparkle-remove", features = ["video-fdncnn"] }
```

### Buffer API

RGBA, 4 bytes/pixel, row-major (`data.len() == width * height * 4`). You own encode/decode.

```rust
use gemini_sparkle_remove::{remove_at, remove_gemini_sparkle, RemoveResult, RgbaImage};

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
use gemini_sparkle_remove::video::{
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
cargo test --features video-fdncnn   # needs third_party/ncnn
```

## CI and releases

GitHub Actions (`.github/workflows/ci.yml`) on `main` and pull requests:

| Job | Runner | What it runs |
|-----|--------|----------------|
| `test` | Linux, macOS, Windows | `cargo test` (default features) |
| `video` | Linux | `cargo test --all-features` and `cargo build --release --examples --all-features` |

The video job installs **system** `ffmpeg` / `ffprobe` and enables `system-ffmpeg`, so build-time ffmpeg download is skipped. `video-fdncnn` uses the bundled `third_party/ncnn` static lib (ELF x86_64; Linux CI only).

To cut a GitHub Release, push a tag matching `Cargo.toml` `version`:

```bash
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/release.yml` re-runs default + all-features tests, `cargo package`s the crate (no crates.io publish), and attaches the `.crate` plus `SHA256SUMS.txt`.

## License

MIT
