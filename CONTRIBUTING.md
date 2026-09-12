# Contributing

## Build and test

Default (still images, no ffmpeg/ncnn):

```bash
cargo test
cargo fmt
```

Video (system ffmpeg / ffprobe on `PATH`, or `GUM_FFMPEG` / `GUM_FFPROBE`):

```bash
cargo test --features video,system-ffmpeg
cargo test --features video-fdncnn,system-ffmpeg
cargo test --features cli
```

CI runs the same matrix. `video-fdncnn` downloads the official Tencent/ncnn
zip for the Cargo target the first time (cached under `third_party/ncnn-prebuilt/`).

Optional env for the skippable full-clip video test:

| Variable | Role |
|----------|------|
| `GUM_SAMPLE_VIDEO` | Path to a watermarked sample `.mp4` |
| `GUM_REFERENCE_VIDEO` | Optional cleaned reference `.mp4` for BR mean-diff |

Do not commit `third_party/ncnn-src`, `third_party/ncnn-prebuilt`, or
`third_party/ffmpeg` binaries (gitignored; downloaded at build time).

## CLI

```bash
cargo run --release --features cli --bin gunmark -- <input> <output>
```

## Examples

```bash
cargo run --release --example bench_time --features video -- <input.mp4> <output.mp4>
cargo run --release --example probe_force_alpha --features video -- <input.mp4> <output.mp4> <alpha>
```

## Style

Run `cargo fmt` before sending a change. Public rustdoc stays English.
Keep comments that explain a non-obvious constraint; skip implementation diary.
