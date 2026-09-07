use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use gemini_unmark::cli::{classify_input, InputKind};
use gemini_unmark::video::{remove_video, MarkKind, VideoRemoveOptions};
use gemini_unmark::{remove_gemini_sparkle, RemoveResult, RgbaImage};

#[derive(Parser, Debug)]
#[command(
    name = "gunmark",
    about = "Remove Gemini visible sparkle watermarks from images and videos"
)]
struct Args {
    /// Input image or video
    input: PathBuf,
    /// Output path (same type as input)
    output: PathBuf,
    /// Video mark family (ignored for still images)
    #[arg(long, value_enum, default_value_t = MarkArg::Auto)]
    mark: MarkArg,
    /// Use compact/legacy diamond geometry (video)
    #[arg(long, default_value_t = false)]
    legacy: bool,
    /// Force a fixed video alpha scale (omit for adaptive)
    #[arg(long)]
    force_alpha: Option<f32>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MarkArg {
    Auto,
    Diamond,
    Veo,
}

impl From<MarkArg> for MarkKind {
    fn from(v: MarkArg) -> Self {
        match v {
            MarkArg::Auto => MarkKind::Auto,
            MarkArg::Diamond => MarkKind::Diamond,
            MarkArg::Veo => MarkKind::Veo,
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    match classify_input(&args.input)? {
        InputKind::Image => process_image(&args.input, &args.output),
        InputKind::Video => process_video(args),
    }
}

fn process_image(input: &Path, output: &Path) -> Result<(), String> {
    let img = image::open(input).map_err(|e| format!("open {}: {e}", input.display()))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut buf = rgba.into_raw();
    let result = {
        let mut view = RgbaImage {
            width: w,
            height: h,
            data: &mut buf,
        };
        remove_gemini_sparkle(&mut view)
    };
    match result {
        RemoveResult::Removed { x, y } => {
            eprintln!("removed sparkle at ({x}, {y})");
        }
        RemoveResult::NotFound => {
            eprintln!("no sparkle detected; writing unchanged buffer");
        }
    }
    image::RgbaImage::from_raw(w, h, buf)
        .ok_or_else(|| "internal: rgba buffer size mismatch".to_string())?
        .save(output)
        .map_err(|e| format!("write {}: {e}", output.display()))?;
    Ok(())
}

fn process_video(args: &Args) -> Result<(), String> {
    let opts = VideoRemoveOptions {
        mark: args.mark.into(),
        legacy: args.legacy,
        force_alpha: args.force_alpha,
    };
    let result = remove_video(&args.input, &args.output, &opts)
        .map_err(|e| format!("remove_video: {e}"))?;
    eprintln!(
        "ok frames={} skipped={} mark={:?} region={:?}",
        result.frames_processed, result.frames_skipped, result.mark, result.region
    );
    Ok(())
}
