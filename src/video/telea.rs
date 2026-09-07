//! Compact Telea-style inpaint for the watermark ROI (no OpenCV).
//!
//! Onion-peel FMM: repeatedly fill mask pixels that border known pixels,
//! using a Gaussian of nearby known samples. Approximates OpenCV TELEA
//! well enough for the diamond footprint after reverse-blend.

/// Inpaint `roi` (RGB planar interleaved, `mw*mh*3`) where `mask` is true.
///
/// Known pixels (`mask == false`) are left unchanged and used as sources.
pub fn inpaint_telea(roi: &mut [f32], mask: &[bool], mw: usize, mh: usize, radius: i32) {
    if mw == 0 || mh == 0 || radius <= 0 {
        return;
    }
    debug_assert_eq!(roi.len(), mw * mh * 3);
    debug_assert_eq!(mask.len(), mw * mh);

    let mut unknown = mask.to_vec();
    if !unknown.iter().any(|&u| u) {
        return;
    }

    let rad = radius.max(1);
    let sigma = (rad as f32) * 0.55;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);

    // Max iterations: one peel per pixel depth.
    let max_passes = mw + mh + 8;
    for _ in 0..max_passes {
        let mut frontier: Vec<usize> = Vec::new();
        for y in 0..mh {
            for x in 0..mw {
                let i = y * mw + x;
                if !unknown[i] {
                    continue;
                }
                // Adjacent to a known pixel?
                let mut border = false;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dy == 0 && dx == 0 {
                            continue;
                        }
                        let yy = y as i32 + dy;
                        let xx = x as i32 + dx;
                        if yy < 0 || xx < 0 || yy as usize >= mh || xx as usize >= mw {
                            continue;
                        }
                        if !unknown[yy as usize * mw + xx as usize] {
                            border = true;
                        }
                    }
                }
                if border {
                    frontier.push(i);
                }
            }
        }
        if frontier.is_empty() {
            // Isolated unknowns: fill from any known in expanded radius.
            for y in 0..mh {
                for x in 0..mw {
                    let i = y * mw + x;
                    if !unknown[i] {
                        continue;
                    }
                    if fill_from_known(roi, &unknown, mw, mh, x, y, rad * 3, inv_2s2) {
                        unknown[i] = false;
                    }
                }
            }
            break;
        }

        // Snapshot so simultaneous frontier uses consistent known set.
        let src = roi.to_vec();
        let unk_snap = unknown.clone();
        for &i in &frontier {
            let y = i / mw;
            let x = i % mw;
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || xx < 0 || yy as usize >= mh || xx as usize >= mw {
                        continue;
                    }
                    let j = yy as usize * mw + xx as usize;
                    if unk_snap[j] {
                        continue;
                    }
                    let w = (-((dx * dx + dy * dy) as f32) * inv_2s2).exp();
                    // Soft Telea-ish: boost samples whose gradient points toward p
                    // approximated by preferring nearer known (already in gaussian).
                    let o = j * 3;
                    acc[0] += w * src[o];
                    acc[1] += w * src[o + 1];
                    acc[2] += w * src[o + 2];
                    wsum += w;
                }
            }
            if wsum > 1e-6 {
                let o = i * 3;
                roi[o] = acc[0] / wsum;
                roi[o + 1] = acc[1] / wsum;
                roi[o + 2] = acc[2] / wsum;
                unknown[i] = false;
            }
        }
        if !unknown.iter().any(|&u| u) {
            break;
        }
    }
}

fn fill_from_known(
    roi: &mut [f32],
    unknown: &[bool],
    mw: usize,
    mh: usize,
    x: usize,
    y: usize,
    rad: i32,
    inv_2s2: f32,
) -> bool {
    let mut acc = [0f32; 3];
    let mut wsum = 0f32;
    for dy in -rad..=rad {
        for dx in -rad..=rad {
            let yy = y as i32 + dy;
            let xx = x as i32 + dx;
            if yy < 0 || xx < 0 || yy as usize >= mh || xx as usize >= mw {
                continue;
            }
            let j = yy as usize * mw + xx as usize;
            if unknown[j] {
                continue;
            }
            let w = (-((dx * dx + dy * dy) as f32) * inv_2s2).exp();
            let o = j * 3;
            acc[0] += w * roi[o];
            acc[1] += w * roi[o + 1];
            acc[2] += w * roi[o + 2];
            wsum += w;
        }
    }
    if wsum <= 1e-6 {
        return false;
    }
    let o = (y * mw + x) * 3;
    roi[o] = acc[0] / wsum;
    roi[o + 1] = acc[1] / wsum;
    roi[o + 2] = acc[2] / wsum;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_center_hole_toward_neighbors() {
        let mw = 8usize;
        let mh = 8usize;
        let mut roi = vec![100f32; mw * mh * 3];
        // Known border already 100; put a hole of 0s in the center.
        let mut mask = vec![false; mw * mh];
        for y in 2..6 {
            for x in 2..6 {
                let i = y * mw + x;
                mask[i] = true;
                let o = i * 3;
                roi[o] = 0.0;
                roi[o + 1] = 0.0;
                roi[o + 2] = 0.0;
            }
        }
        inpaint_telea(&mut roi, &mask, mw, mh, 3);
        for y in 2..6 {
            for x in 2..6 {
                let o = (y * mw + x) * 3;
                assert!(
                    (roi[o] - 100.0).abs() < 5.0,
                    "center ({x},{y}) = {} want ~100",
                    roi[o]
                );
            }
        }
    }
}
