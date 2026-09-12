//! Remove Gemini visible sparkle watermarks from RGBA buffers.
//!
//! Crate name: `gemini-unmark` (import as `gemini_unmark`).
//! Pure Rust; optional `video` / `video-fdncnn` features need ffmpeg tools
//! (build-downloaded or PATH) and a per-target cmake-built libncnn — never Python.

pub mod blend;
pub mod cli;
pub mod detect;
pub mod native_link;
pub mod telea;
pub mod template;

#[cfg(feature = "video")]
pub mod video;

/// Overlay template as composited onto the image:
/// `observed = alpha * rgb + (1 - alpha) * original`.
/// RGBA buffer; alpha is overlay opacity, RGB is straight (non-premultiplied) overlay color.
#[derive(Clone, Debug)]
pub struct WatermarkTemplate {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub use blend::reverse_alpha_blend;
pub use template::{sparkle_template, SparkleGeometry, SPARKLE_GEOMETRY, SPARKLE_SIZE};

pub use detect::{match_watermark, Match};

/// Mutable RGBA view (`data.len() == width * height * 4`).
pub struct RgbaImage<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a mut [u8],
}

/// Outcome of [`remove_gemini_sparkle`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoveResult {
    Removed { x: u32, y: u32 },
    NotFound,
}

const OPAQUE_CUTOFF: f64 = 0.95;

/// Reverse-blend leftover is still a visible star on smooth BR (sky ~control 2,
/// survival ~0.40). Inpaint the alpha footprint in that case only — busy
/// texture (wood/castle) has high control and must stay reverse-blend-only.
const SMOOTH_CONTROL: f64 = 40.0;
const RESIDUAL_SURVIVAL: f64 = 0.20;
const INPAINT_ALPHA: u8 = 1;
const INPAINT_RADIUS: i32 = 4;
const INPAINT_PAD: i32 = 10;
const INPAINT_DILATE: i32 = 4;

/// Auto-detect the Gemini sparkle and reverse-blend it in place.
///
/// If no mark clears the detect gates, returns [`RemoveResult::NotFound`] and
/// leaves the buffer unchanged.
pub fn remove_gemini_sparkle(img: &mut RgbaImage<'_>) -> RemoveResult {
    assert_eq!(
        img.data.len(),
        (img.width as usize) * (img.height as usize) * 4
    );
    let Some(m) = match_watermark(img.width, img.height, img.data) else {
        return RemoveResult::NotFound;
    };
    let _mask = reverse_alpha_blend(
        img.width,
        img.height,
        img.data,
        &m.template,
        m.x as i32,
        m.y as i32,
        OPAQUE_CUTOFF,
    );
    residual_inpaint_smooth(img, &m);
    RemoveResult::Removed { x: m.x, y: m.y }
}

fn residual_inpaint_smooth(img: &mut RgbaImage<'_>, m: &Match) {
    if m.residual <= RESIDUAL_SURVIVAL || m.control >= SMOOTH_CONTROL {
        return;
    }
    let tw = m.template.width as i32;
    let th = m.template.height as i32;
    if tw == 0 || th == 0 {
        return;
    }
    let pad = INPAINT_PAD;
    let rw = tw + pad * 2;
    let rh = th + pad * 2;
    let x0 = m.x as i32 - pad;
    let y0 = m.y as i32 - pad;
    let img_w = img.width as i32;
    let img_h = img.height as i32;
    let rw_u = rw as usize;
    let rh_u = rh as usize;
    let mut roi = vec![0f32; rw_u * rh_u * 3];
    let mut seed = vec![false; rw_u * rh_u];
    let mut any = false;
    for ry in 0..rh {
        let y = y0 + ry;
        if y < 0 || y >= img_h {
            continue;
        }
        for rx in 0..rw {
            let x = x0 + rx;
            if x < 0 || x >= img_w {
                continue;
            }
            let ii = ((y as usize) * img.width as usize + x as usize) * 4;
            let oi = (ry as usize * rw_u + rx as usize) * 3;
            roi[oi] = img.data[ii] as f32;
            roi[oi + 1] = img.data[ii + 1] as f32;
            roi[oi + 2] = img.data[ii + 2] as f32;
            let tx = rx - pad;
            let ty = ry - pad;
            if tx >= 0 && ty >= 0 && tx < tw && ty < th {
                let a = m.template.data[(ty as usize * tw as usize + tx as usize) * 4 + 3];
                if a >= INPAINT_ALPHA {
                    seed[ry as usize * rw_u + rx as usize] = true;
                    any = true;
                }
            }
        }
    }
    if !any {
        return;
    }
    let mask = dilate_mask(&seed, rw_u, rh_u, INPAINT_DILATE);
    telea::inpaint_telea(&mut roi, &mask, rw_u, rh_u, INPAINT_RADIUS);
    for ry in 0..rh {
        let y = y0 + ry;
        if y < 0 || y >= img_h {
            continue;
        }
        for rx in 0..rw {
            if !mask[ry as usize * rw_u + rx as usize] {
                continue;
            }
            let x = x0 + rx;
            if x < 0 || x >= img_w {
                continue;
            }
            let ii = ((y as usize) * img.width as usize + x as usize) * 4;
            let oi = (ry as usize * rw_u + rx as usize) * 3;
            img.data[ii] = roi[oi].round().clamp(0.0, 255.0) as u8;
            img.data[ii + 1] = roi[oi + 1].round().clamp(0.0, 255.0) as u8;
            img.data[ii + 2] = roi[oi + 2].round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn dilate_mask(seed: &[bool], w: usize, h: usize, r: i32) -> Vec<bool> {
    let mut out = seed.to_vec();
    if r <= 0 {
        return out;
    }
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !seed[y as usize * w + x as usize] {
                continue;
            }
            for dy in -r..=r {
                for dx in -r..=r {
                    let yy = y + dy;
                    let xx = x + dx;
                    if yy < 0 || xx < 0 || yy >= h as i32 || xx >= w as i32 {
                        continue;
                    }
                    out[yy as usize * w + xx as usize] = true;
                }
            }
        }
    }
    out
}

/// Force reverse-blend of the canonical 48×48 sparkle at `(x, y)` (top-left).
pub fn remove_at(img: &mut RgbaImage<'_>, x: u32, y: u32) {
    assert_eq!(
        img.data.len(),
        (img.width as usize) * (img.height as usize) * 4
    );
    let tpl = sparkle_template();
    let _mask = reverse_alpha_blend(
        img.width,
        img.height,
        img.data,
        &tpl,
        x as i32,
        y as i32,
        OPAQUE_CUTOFF,
    );
}
