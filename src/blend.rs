//! Reverse alpha-blend for Gemini sparkle watermark recovery.
//!
//! Undo a standard alpha-blend overlay in place:
//!   observed = a * overlayColor + (1 - a) * original
//!   original = (observed - a * overlayColor) / (1 - a)
//!
//! Pixels where the overlay is (near-)opaque carry almost no information
//! about the original and cannot be inverted; those are returned as a mask
//! so the caller can inpaint them instead.

use crate::WatermarkTemplate;

/// Reverse-blend `tpl` onto `img` at `(at_x, at_y)` in place.
///
/// Returns a residual mask of unrecoverable pixels (1 = needs inpainting),
/// same dimensions as the full image (`img_w * img_h`).
pub fn reverse_alpha_blend(
    img_w: u32,
    img_h: u32,
    img: &mut [u8],
    tpl: &WatermarkTemplate,
    at_x: i32,
    at_y: i32,
    opaque_cutoff: f64,
) -> Vec<u8> {
    let iw = img_w as usize;
    let ih = img_h as usize;
    debug_assert_eq!(img.len(), iw * ih * 4);

    let mut mask = vec![0u8; iw * ih];
    let tw = tpl.width as usize;
    let th = tpl.height as usize;
    let t = &tpl.data;

    for ty in 0..th {
        let y = at_y + ty as i32;
        if y < 0 || y as usize >= ih {
            continue;
        }
        let y = y as usize;
        for tx in 0..tw {
            let x = at_x + tx as i32;
            if x < 0 || x as usize >= iw {
                continue;
            }
            let x = x as usize;

            let ti = (ty * tw + tx) * 4;
            let a = t[ti + 3] as f64 / 255.0;
            if a == 0.0 {
                continue;
            }

            let ii = (y * iw + x) * 4;
            if a >= opaque_cutoff {
                mask[y * iw + x] = 1;
                continue;
            }
            let inv = 1.0 / (1.0 - a);
            img[ii] = clamp((img[ii] as f64 - a * t[ti] as f64) * inv);
            img[ii + 1] = clamp((img[ii + 1] as f64 - a * t[ti + 1] as f64) * inv);
            img[ii + 2] = clamp((img[ii + 2] as f64 - a * t[ti + 2] as f64) * inv);
        }
    }
    mask
}

fn clamp(v: f64) -> u8 {
    if v < 0.0 {
        0
    } else if v > 255.0 {
        255
    } else {
        v.round() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2×2 buffer, 1×1 overlay α≈0.5, recover RGB≈100.
    #[test]
    fn reverses_known_alpha_blend() {
        // Original RGB = 100; overlay white (255) at α = 128/255 ≈ 0.5
        // observed = round(a*255 + (1-a)*100) = 178
        let mut img = Vec::with_capacity(16);
        for _ in 0..4 {
            img.extend_from_slice(&[100, 100, 100, 255]);
        }
        // Composite overlay at (0,0): observed ≈ 178
        img[0] = 178;
        img[1] = 178;
        img[2] = 178;

        let tpl = WatermarkTemplate {
            width: 1,
            height: 1,
            data: vec![255, 255, 255, 128], // white, α≈0.5
        };

        let mask = reverse_alpha_blend(2, 2, &mut img, &tpl, 0, 0, 0.95);

        assert_eq!(mask.len(), 4);
        assert_eq!(mask[0], 0);
        // Recovered RGB ≈ 100 (allow ±1 for rounding)
        assert!((img[0] as i16 - 100).abs() <= 1, "R={}", img[0]);
        assert!((img[1] as i16 - 100).abs() <= 1, "G={}", img[1]);
        assert!((img[2] as i16 - 100).abs() <= 1, "B={}", img[2]);
        // Untouched pixel at (1,0)
        assert_eq!(&img[4..8], &[100, 100, 100, 255]);
    }
}
