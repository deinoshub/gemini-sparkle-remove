# Build-time ffmpeg/ffprobe cache

Static tools are downloaded by `build.rs` when feature `video` is enabled (and
`system-ffmpeg` is **not**) into:

`<bundle-version>/<target-triple>/ffmpeg` (+ `ffprobe`)

Current bundle id: **`btbn-master-2026-09`**.

| Target | Source URL |
|--------|------------|
| Linux x86_64 | https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz |
| Linux aarch64 | …/ffmpeg-master-latest-linuxarm64-gpl.tar.xz |
| Windows x86_64 | …/ffmpeg-master-latest-win64-gpl.zip |
| macOS x86_64 | https://evermeet.cx/ffmpeg/ffmpeg-7.1.1.zip + ffprobe-7.1.1.zip |
| macOS aarch64 | https://www.osxexperts.net/ffmpeg71arm.zip |

Binaries are **gitignored** (multi-100MB).

- Default: `cargo build --features video` downloads tools.
- Opt out: `cargo build --features video,system-ffmpeg` (PATH / `GSR_FFMPEG`/`GSR_FFPROBE` only).
- Env override: `GSR_SKIP_FFMPEG_DOWNLOAD=1` skips download (still uses cache if already present).
