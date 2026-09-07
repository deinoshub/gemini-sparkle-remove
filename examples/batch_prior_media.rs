//! Batch-test prior conversation media with this crate.
use gemini_unmark::{remove_at, remove_gemini_sparkle, RemoveResult, RgbaImage};
use image::ImageReader;
use std::path::{Path, PathBuf};

#[cfg(feature = "video")]
use gemini_unmark::video::{remove_video, VideoRemoveOptions};

fn process_image(src: &Path, dst: &Path) -> String {
    let img = ImageReader::open(src)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
    let (w, h) = img.dimensions();
    let mut buf = img.into_raw();
    let mut view = RgbaImage {
        width: w,
        height: h,
        data: &mut buf,
    };
    let msg = match remove_gemini_sparkle(&mut view) {
        RemoveResult::Removed { x, y } => format!("Removed at ({x},{y})"),
        RemoveResult::NotFound => {
            // force BR inset 96 like gwr geometry if detect misses (e.g. 1c3d9c64)
            if w >= 48 + 96 && h >= 48 + 96 {
                let x = w - 96;
                let y = h - 96;
                remove_at(&mut view, x, y);
                format!("NotFound → remove_at ({x},{y})")
            } else {
                "NotFound (no force)".into()
            }
        }
    };
    let rgba = image::RgbaImage::from_raw(w, h, buf).unwrap();
    image::DynamicImage::ImageRgba8(rgba)
        .to_rgb8()
        .save(dst)
        .unwrap();
    msg
}

fn main() {
    let att = PathBuf::from(
        "/home/box/agent-data/agents/45f20ddc-5a49-4f0e-a717-b99340b95659/attachments",
    );
    let out_img = PathBuf::from("/workspace/rust-crate-test/images");
    std::fs::create_dir_all(&out_img).unwrap();

    let originals = [
        "29b8e42b83838685eb3d2a6087ae79b822bd69c6dc1eb9645f27e68c3f9769e2.jpeg",
        "9617fa8a76a8a88cf8ae1108d3bdf42b2c932c2a625a10afbc64a91476061a6c.jpeg",
        "1c3d9c64bf0eeb6fa602ab16df0146aedfedad224dc3599c41f71cf5cd590cb6.jpeg",
        "6b444d927a52d938387326b66877d71434081db51ea3b6d7eff39b3b6bcf0325.jpeg",
        "4cc8b656e467f60e2ff80e4cb26f6ce00536d8be5540485e10f288f70db7efbb.jpeg",
        "0ba2f089c58387743bb42079ec013e3ef95062c0c4c566ccf14e224790808c5d.jpeg",
    ];

    println!("=== IMAGES ===");
    for name in originals {
        let src = att.join(name);
        let short = &name[..8];
        let dst = out_img.join(format!("{short}_rust.jpeg"));
        let msg = process_image(&src, &dst);
        println!("{short}: {msg} -> {}", dst.display());
    }

    #[cfg(feature = "video")]
    {
        println!("=== VIDEO ===");
        let vin = Path::new("/workspace/video-in/input.mp4");
        let vout = Path::new("/workspace/rust-crate-test/video/cleaned_rust.mp4");
        std::fs::create_dir_all(vout.parent().unwrap()).unwrap();
        let opts = VideoRemoveOptions::default();
        match remove_video(vin, vout, &opts) {
            Ok(r) => println!(
                "video: ok frames={} skipped={} mark={:?} region={:?} -> {}",
                r.frames_processed, r.frames_skipped, r.mark, r.region, vout.display()
            ),
            Err(e) => eprintln!("video: ERR {e}"),
        }
    }
}
