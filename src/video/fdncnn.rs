//! In-process FDnCNN (NcnnDenoiser) ROI postpass.
//!
//! sigma=75, strength=1.8, padding=64; gradient-masked footprint blend.
//! Compositing: `result = w*denoised + (1-w)*blend` with Gaussian σ=1.0
//! on the padded weight. No classical post-flatten after denoise.
//!
//! Soft ROI: weight pad + Gauss σ=1.0 (no corner α-gate).
//! Full-strength FDnCNN blend always.
//!
//! Temporal: motion-gated EMA on high-α ROI after parallel denoise.

use std::cell::RefCell;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rayon::prelude::*;

use super::detect::VideoDetection;
use super::maps::VideoMap;
use super::ncnn_ffi::{
    gwr_ncnn_configure_fdncnn, ncnn_extractor_create, ncnn_extractor_destroy,
    ncnn_extractor_extract_index, ncnn_extractor_input_index, ncnn_mat_create_3d, ncnn_mat_destroy,
    ncnn_mat_get_channel_data, ncnn_net_create, ncnn_net_destroy, ncnn_net_load_model,
    ncnn_net_load_param_bin, ncnn_net_t, BLOB_INPUT, BLOB_OUTPUT,
};

const SIGMA: f32 = 75.0;
const STRENGTH: f32 = 1.8;
const PADDING: i32 = 64;
/// Classical post-FDnCNN flatten is disabled.
const POST_GRAIN_STRENGTH: f32 = 0.0;
const POST_GRAIN_BLUR_SIGMA: f32 = 1.15;
const POST_GRAIN_GUIDE_SIGMA: f32 = 2.0;
const POST_LUMA_MATCH: f32 = 0.0;
const RESIDUAL_LF_ATTEN: f32 = 0.0;
const RESIDUAL_LF_BLUR_SIGMA: f32 = 1.2;
const DARK_FLAT_STRENGTH: f32 = 0.0;
const DARK_TIP_OVERSHOOT: f32 = 0.0;
const DARK_BG_LUMA: f32 = 52.0;
const DARK_FLAT_GUIDE_SIGMA: f32 = 5.0;
const DARK_HOLE_PULL: f32 = 0.0;
const BRIGHT_BG_LUMA: f32 = 80.0;
/// Motion-gated temporal EMA on high-α ROI (second pass, sequential).
/// Temporal EMA disabled (0 = off).
const TEMPORAL_EMA: f32 = 0.0;
/// Exterior MAD (0–255) at which temporal EMA fully disengages.
const TEMPORAL_MOTION_SCALE: f32 = 20.0;
/// Weight-mask Gaussian sigma.
const WEIGHT_FEATHER_SIGMA: f32 = 1.0;
/// Soft-gate footprint by map α: only near-zero α AABB *corners* are attenuated
/// (tips sit mid-side at high α and stay fully weighted).
const ALPHA_GATE_LO: f32 = 0.010;
const ALPHA_GATE_HI: f32 = 0.055;
/// Corner falloff radius in map px (both axes near AABB edge).
const CORNER_GATE_PX: f32 = 10.0;

/// OpenCV `MORPH_ELLIPSE` 5×5 structuring element.
const ELLIPSE5: [[u8; 5]; 5] = [
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
];

struct WeightCache {
    key: (u32, u32), // map w,h
    weight: Vec<f32>,
    active: usize,
}

static WEIGHT_CACHE: OnceLock<std::sync::Mutex<Option<WeightCache>>> = OnceLock::new();

/// Loaded FDnCNN net (not Sync — keep per-thread via `thread_local!`).
pub struct FdncnnNet {
    net: ncnn_net_t,
}

impl Drop for FdncnnNet {
    fn drop(&mut self) {
        if !self.net.is_null() {
            unsafe { ncnn_net_destroy(self.net) };
            self.net = std::ptr::null_mut();
        }
    }
}

// Safety: we never share FdncnnNet across threads; thread_local only.
unsafe impl Send for FdncnnNet {}

impl FdncnnNet {
    pub fn load(param: &Path, model: &Path, num_threads: i32) -> Option<Self> {
        let param_c = CString::new(param.to_str()?).ok()?;
        let model_c = CString::new(model.to_str()?).ok()?;
        unsafe {
            let net = ncnn_net_create();
            if net.is_null() {
                return None;
            }
            gwr_ncnn_configure_fdncnn(net, num_threads.max(1));
            if ncnn_net_load_param_bin(net, param_c.as_ptr()) != 0 {
                ncnn_net_destroy(net);
                return None;
            }
            if ncnn_net_load_model(net, model_c.as_ptr()) != 0 {
                ncnn_net_destroy(net);
                return None;
            }
            Some(Self { net })
        }
    }

    /// RGB uint8 H×W×3 interleaved → denoised RGB float in [0, 255] (no u8 round-trip).
    /// Float composite path (no u8 round-trip before blend).
    pub fn run_fdncnn_f32(
        &self,
        rgb_u8: &[u8],
        w: usize,
        h: usize,
        sigma: f32,
    ) -> Option<Vec<f32>> {
        assert_eq!(rgb_u8.len(), w * h * 3);
        unsafe {
            let mat = ncnn_mat_create_3d(w as i32, h as i32, 4, std::ptr::null_mut());
            if mat.is_null() {
                return None;
            }
            let sigma_n = sigma / 255.0;
            for c in 0..3 {
                let ptr = ncnn_mat_get_channel_data(mat, c) as *mut f32;
                if ptr.is_null() {
                    ncnn_mat_destroy(mat);
                    return None;
                }
                let plane = std::slice::from_raw_parts_mut(ptr, w * h);
                for i in 0..w * h {
                    plane[i] = rgb_u8[i * 3 + c as usize] as f32 / 255.0;
                }
            }
            {
                let ptr = ncnn_mat_get_channel_data(mat, 3) as *mut f32;
                if ptr.is_null() {
                    ncnn_mat_destroy(mat);
                    return None;
                }
                let plane = std::slice::from_raw_parts_mut(ptr, w * h);
                plane.fill(sigma_n);
            }

            let ex = ncnn_extractor_create(self.net);
            if ex.is_null() {
                ncnn_mat_destroy(mat);
                return None;
            }
            let ri = ncnn_extractor_input_index(ex, BLOB_INPUT, mat);
            let mut out: *mut std::ffi::c_void = std::ptr::null_mut();
            let re = ncnn_extractor_extract_index(ex, BLOB_OUTPUT, &mut out);
            ncnn_extractor_destroy(ex);
            ncnn_mat_destroy(mat);
            if ri != 0 || re != 0 || out.is_null() {
                if !out.is_null() {
                    ncnn_mat_destroy(out);
                }
                return None;
            }

            let mut den = vec![0f32; w * h * 3];
            for c in 0..3 {
                let ptr = ncnn_mat_get_channel_data(out, c) as *const f32;
                if ptr.is_null() {
                    ncnn_mat_destroy(out);
                    return None;
                }
                let plane = std::slice::from_raw_parts(ptr, w * h);
                for i in 0..w * h {
                    den[i * 3 + c as usize] = plane[i].clamp(0.0, 1.0) * 255.0;
                }
            }
            ncnn_mat_destroy(out);
            Some(den)
        }
    }

    /// RGB uint8 → denoised RGB uint8 (tests / helpers).
    #[allow(dead_code)]
    pub fn run_fdncnn(&self, rgb_u8: &[u8], w: usize, h: usize, sigma: f32) -> Option<Vec<u8>> {
        let den = self.run_fdncnn_f32(rgb_u8, w, h, sigma)?;
        Some(
            den.into_iter()
                .map(|v| (v + 0.5).clamp(0.0, 255.0) as u8)
                .collect(),
        )
    }
}

thread_local! {
    static TLS_NET: RefCell<Option<FdncnnNet>> = const { RefCell::new(None) };
}

fn with_tls_net<R>(param: &Path, model: &Path, f: impl FnOnce(&FdncnnNet) -> R) -> Option<R> {
    TLS_NET.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            // One thread inside the net; outer rayon already parallelizes frames.
            let nthreads = 1;
            *slot = FdncnnNet::load(param, model, nthreads);
        }
        let net = slot.as_ref()?;
        Some(f(net))
    })
}

/// OpenCV Sobel ksize=3 (BORDER_REFLECT_101) magnitude.
fn sobel_mag(alpha: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut mag = vec![0f32; w * h];
    let at = |x: i32, y: i32| -> f32 {
        // BORDER_REFLECT_101: -1 → 1, w → w-2
        let xx = reflect101(x, w as i32) as usize;
        let yy = reflect101(y, h as i32) as usize;
        alpha[yy * w + xx]
    };
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let gx = -at(x - 1, y - 1) + at(x + 1, y - 1) - 2.0 * at(x - 1, y) + 2.0 * at(x + 1, y)
                - at(x - 1, y + 1)
                + at(x + 1, y + 1);
            let gy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1)
                + at(x - 1, y + 1)
                + 2.0 * at(x, y + 1)
                + at(x + 1, y + 1);
            mag[y as usize * w + x as usize] = (gx * gx + gy * gy).sqrt();
        }
    }
    mag
}

fn reflect101(p: i32, len: i32) -> i32 {
    if len <= 1 {
        return 0;
    }
    let mut x = p;
    loop {
        if x < 0 {
            x = -x;
        } else if x >= len {
            x = 2 * len - 2 - x;
        } else {
            return x;
        }
    }
}

fn dilate_ellipse5(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut dst = vec![0f32; w * h];
    let r = 2i32;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut m = 0f32;
            for ky in 0..5 {
                for kx in 0..5 {
                    if ELLIPSE5[ky][kx] == 0 {
                        continue;
                    }
                    let yy = y + ky as i32 - r;
                    let xx = x + kx as i32 - r;
                    if yy < 0 || xx < 0 || yy >= h as i32 || xx >= w as i32 {
                        continue;
                    }
                    m = m.max(src[yy as usize * w + xx as usize]);
                }
            }
            dst[y as usize * w + x as usize] = m;
        }
    }
    dst
}

/// Separable Gaussian blur; ksize = round(sigma*8+1)|1 (OpenCV float formula).
fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let ksize = ((sigma * 4.0 * 2.0).round() as i32 + 1) | 1;
    let ksize = ksize.max(1);
    let rad = ksize / 2;
    let mut ker = vec![0f32; ksize as usize];
    let inv = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0f32;
    for i in 0..ksize {
        let x = (i - rad) as f32;
        let v = (-x * x * inv).exp();
        ker[i as usize] = v;
        sum += v;
    }
    for v in ker.iter_mut() {
        *v /= sum;
    }

    // horizontal
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w as i32 {
            let mut acc = 0f32;
            for i in 0..ksize {
                let xx = reflect101(x + i - rad, w as i32) as usize;
                acc += src[y * w + xx] * ker[i as usize];
            }
            tmp[y * w + x as usize] = acc;
        }
    }
    // vertical
    let mut dst = vec![0f32; w * h];
    for y in 0..h as i32 {
        for x in 0..w {
            let mut acc = 0f32;
            for i in 0..ksize {
                let yy = reflect101(y + i - rad, h as i32) as usize;
                acc += tmp[yy * w + x] * ker[i as usize];
            }
            dst[y as usize * w + x] = acc;
        }
    }
    dst
}

/// Gradient-masked footprint.
pub fn footprint_weight(alpha: &[f32], w: usize, h: usize, strength: f32) -> (Vec<f32>, usize) {
    assert_eq!(alpha.len(), w * h);
    let mag = sobel_mag(alpha, w, h);
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in &mag {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let mut weight = if mx <= mn {
        vec![1f32; w * h]
    } else {
        let inv = 1.0 / (mx - mn);
        let mut gn: Vec<f32> = mag
            .iter()
            .map(|v| ((v - mn) * inv).max(0.0).sqrt())
            .collect();
        gn = dilate_ellipse5(&gn, w, h);
        gaussian_blur(&gn, w, h, 2.0)
    };
    let mut active = 0usize;
    for v in weight.iter_mut() {
        *v = (*v * strength).clamp(0.0, 1.0);
        if *v > 0.01 {
            active += 1;
        }
    }
    (weight, active)
}

fn cached_weight(map: &VideoMap) -> (Vec<f32>, usize) {
    let key = (map.width, map.height);
    let lock = WEIGHT_CACHE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = lock.lock().unwrap();
    if let Some(c) = guard.as_ref() {
        if c.key == key {
            return (c.weight.clone(), c.active);
        }
    }
    let (weight, active) = footprint_weight(
        &map.alpha,
        map.width as usize,
        map.height as usize,
        STRENGTH,
    );
    *guard = Some(WeightCache {
        key,
        weight: weight.clone(),
        active,
    });
    (weight, active)
}

#[allow(dead_code)]
fn alpha_soft_gate(a: f32) -> f32 {
    ((a - ALPHA_GATE_LO) / (ALPHA_GATE_HI - ALPHA_GATE_LO)).clamp(0.0, 1.0)
}

/// Attenuate weight only in AABB corners (both axes near the border). Mid-side
/// diamond tips stay at full weight so sparkle arms are still denoised.
#[allow(dead_code)]
fn corner_alpha_gate(a: f32, col: usize, row: usize, mw: usize, mh: usize) -> f32 {
    let ex = (col.min(mw - 1 - col)) as f32;
    let ey = (row.min(mh - 1 - row)) as f32;
    let cx = (1.0 - (ex / CORNER_GATE_PX).clamp(0.0, 1.0)).max(0.0);
    let cy = (1.0 - (ey / CORNER_GATE_PX).clamp(0.0, 1.0)).max(0.0);
    let corner = cx * cy; // high only near corners
    if corner <= 1e-4 {
        return 1.0;
    }
    let ag = alpha_soft_gate(a);
    1.0 - corner * (1.0 - ag)
}

fn process_frame_rgba(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
    weight: &[f32],
    net: &FdncnnNet,
) -> bool {
    let frame_w = width as i32;
    let frame_h = height as i32;
    let rw = map.width as i32;
    let rh = map.height as i32;
    let x = det.x as i32;
    let y = det.y as i32;
    if x < 0 || y < 0 || x + rw > frame_w || y + rh > frame_h {
        return false;
    }
    let x0 = (x - PADDING).max(0);
    let y0 = (y - PADDING).max(0);
    let x1 = (x + rw + PADDING).min(frame_w);
    let y1 = (y + rh + PADDING).min(frame_h);
    let pw = (x1 - x0) as usize;
    let ph = (y1 - y0) as usize;
    if pw < 4 || ph < 4 {
        return false;
    }

    // Extract RGB ROI from RGBA frame.
    let mut rgb = vec![0u8; pw * ph * 3];
    let sw = width as usize;
    for py in 0..ph {
        for px in 0..pw {
            let src = ((y0 as usize + py) * sw + (x0 as usize + px)) * 4;
            let dst = (py * pw + px) * 3;
            rgb[dst] = rgba[src];
            rgb[dst + 1] = rgba[src + 1];
            rgb[dst + 2] = rgba[src + 2];
        }
    }

    let Some(den) = net.run_fdncnn_f32(&rgb, pw, ph, SIGMA) else {
        return false;
    };

    // Pad weight into ROI (no corner α-gate), then σ=1 feather.
    let mw = map.width as usize;
    let mh = map.height as usize;
    let mut wpad = vec![0f32; pw * ph];
    let ox = (x - x0) as usize;
    let oy = (y - y0) as usize;
    for row in 0..mh {
        for col in 0..mw {
            wpad[(oy + row) * pw + (ox + col)] = weight[row * mw + col];
        }
    }
    wpad = gaussian_blur(&wpad, pw, ph, WEIGHT_FEATHER_SIGMA);

    // Full-strength FDnCNN blend: result = w*denoised + (1-w)*original.
    let mut out = vec![0f32; pw * ph * 3];
    for i in 0..pw * ph {
        let wi = wpad[i];
        let si = i * 3;
        for c in 0..3 {
            let o = rgb[si + c] as f32;
            let d = den[si + c];
            out[si + c] = d * wi + o * (1.0 - wi);
        }
    }

    // Classical post (grain / luma / dark-flat / LF atten) is disabled.
    if POST_GRAIN_STRENGTH > 1e-6
        || POST_LUMA_MATCH > 1e-6
        || RESIDUAL_LF_ATTEN > 1e-6
        || DARK_FLAT_STRENGTH > 1e-6
    {
        let mut apad = vec![0f32; pw * ph];
        for row in 0..mh {
            for col in 0..mw {
                apad[(oy + row) * pw + (ox + col)] = map.alpha[row * mw + col];
            }
        }
        post_denoise_grain_and_luma(&mut out, &rgb, &wpad, &apad, pw, ph);
    }

    // Write back only where the soft weight is meaningful — pure exterior keeps
    // original bytes (avoids float round-trip edge on pad).
    for py in 0..ph {
        for px in 0..pw {
            let i = py * pw + px;
            if wpad[i] <= 1e-6 {
                continue;
            }
            let dst = ((y0 as usize + py) * sw + (x0 as usize + px)) * 4;
            let si = i * 3;
            for c in 0..3 {
                rgba[dst + c] = out[si + c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    true
}

/// Reinject exterior HF grain into the denoise footprint, softly match luma,
/// and α-weighted-attenuate LF residual (fixed strength — no mode flips).
/// Operates on the padded ROI in RGB float.
fn post_denoise_grain_and_luma(
    out: &mut [f32],
    orig: &[u8],
    weight: &[f32],
    alpha_pad: &[f32],
    pw: usize,
    ph: usize,
) {
    let n = pw * ph;
    debug_assert_eq!(out.len(), n * 3);
    debug_assert_eq!(orig.len(), n * 3);
    debug_assert_eq!(weight.len(), n);
    debug_assert_eq!(alpha_pad.len(), n);

    // Soft exterior mask: low denoise weight (pad ring + map exterior).
    let mut ext = vec![0f32; n];
    let mut ext_n = 0f32;
    for i in 0..n {
        if weight[i] < 0.08 {
            ext[i] = 1.0;
            ext_n += 1.0;
        }
    }
    if ext_n < 16.0 {
        return;
    }

    // --- Luma match: pull weighted interior mean toward exterior mean ---
    let luma = |rgb: &[f32], i: usize| -> f32 {
        let o = i * 3;
        0.299 * rgb[o] + 0.587 * rgb[o + 1] + 0.114 * rgb[o + 2]
    };
    let mut sum_ext = 0f32;
    let mut sum_int = 0f32;
    let mut w_int = 0f32;
    for i in 0..n {
        if ext[i] > 0.5 {
            let o = i * 3;
            sum_ext +=
                0.299 * orig[o] as f32 + 0.587 * orig[o + 1] as f32 + 0.114 * orig[o + 2] as f32;
        }
        let wi = weight[i];
        if wi > 0.05 {
            sum_int += luma(out, i) * wi;
            w_int += wi;
        }
    }
    if w_int > 1e-3 {
        let mean_ext = sum_ext / ext_n;
        let mean_int = sum_int / w_int;
        let bias = (mean_ext - mean_int) * POST_LUMA_MATCH;
        if bias.abs() > 0.15 {
            for i in 0..n {
                let wi = weight[i];
                if wi <= 1e-6 {
                    continue;
                }
                let o = i * 3;
                for c in 0..3 {
                    out[o + c] = (out[o + c] + bias * wi).clamp(0.0, 255.0);
                }
            }
        }
    }

    // --- Residual attenuator (LF mid/bright + full-RGB dark flatten) ---
    // Tip sparkles on dark fabric are mostly HF at σ≈1.2, so LF-only leave
    // bright tips + hollow centers. On dark exterior, α-weighted full-RGB
    // lerp toward (ext − overshoot), then grain reinject restores texture.
    // Mid/bright scenes keep mild LF-only atten (avoid f120 over-dark MAD).
    {
        let mut a_max = 1e-6f32;
        for i in 0..n {
            a_max = a_max.max(alpha_pad[i]);
        }
        let mut ext_mean = [0f32; 3];
        for i in 0..n {
            if ext[i] < 0.5 {
                continue;
            }
            let o = i * 3;
            for c in 0..3 {
                ext_mean[c] += out[o + c];
            }
        }
        for c in 0..3 {
            ext_mean[c] /= ext_n;
        }
        let ext_y = 0.299 * ext_mean[0] + 0.587 * ext_mean[1] + 0.114 * ext_mean[2];
        let bright_f = ((ext_y - BRIGHT_BG_LUMA) / 80.0).clamp(0.0, 1.0);

        // Always build local exterior (used for dark-local bright residual kill).
        let mut ext_w = vec![0f32; n];
        for i in 0..n {
            if ext[i] > 0.5 || alpha_pad[i] < 0.02 {
                ext_w[i] = 1.0;
            }
        }
        let wblur = gaussian_blur(&ext_w, pw, ph, DARK_FLAT_GUIDE_SIGMA);
        let mut local = vec![0f32; n * 3];
        {
            let mut planar = vec![0f32; n];
            for c in 0..3 {
                for i in 0..n {
                    planar[i] = out[i * 3 + c] * ext_w[i];
                }
                let num = gaussian_blur(&planar, pw, ph, DARK_FLAT_GUIDE_SIGMA);
                for i in 0..n {
                    local[i * 3 + c] = num[i] / (wblur[i] + 1e-6);
                }
            }
        }

        // Dark-local full-RGB flatten for bright residual (per-pixel local luma gate).
        {
            let over = DARK_TIP_OVERSHOOT;
            for i in 0..n {
                let an = (alpha_pad[i] / a_max).clamp(0.0, 1.0);
                if an <= 1e-6 {
                    continue;
                }
                let o = i * 3;
                let ly = 0.299 * local[o] + 0.587 * local[o + 1] + 0.114 * local[o + 2];
                if ly >= DARK_BG_LUMA {
                    continue; // local surround not dark fabric
                }
                let y = 0.299 * out[o] + 0.587 * out[o + 1] + 0.114 * out[o + 2];
                let excess = y - ly;
                if excess < 0.5 {
                    continue;
                }
                let dark_f = (1.0 - (ly / DARK_BG_LUMA).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                let gate = (excess / (excess + 3.2)).clamp(0.0, 1.0);
                // Soften on near-threshold local luma to avoid a dark geometric blob.
                let t = (DARK_FLAT_STRENGTH * an * gate * (0.40 + 0.60 * dark_f)).clamp(0.0, 0.85);
                for c in 0..3 {
                    let tgt = (local[o + c] - over * dark_f).max(0.0);
                    out[o + c] = (out[o + c] * (1.0 - t) + tgt * t).clamp(0.0, 255.0);
                }
            }
        }

        // Mild LF-only path for remaining mid/bright residual (preserve texture).
        let atten = RESIDUAL_LF_ATTEN.clamp(0.0, 1.0);
        if atten > 1e-4 {
            let mut lf = vec![0f32; n * 3];
            {
                let mut planar = vec![0f32; n];
                for c in 0..3 {
                    for i in 0..n {
                        planar[i] = out[i * 3 + c];
                    }
                    let blurred = gaussian_blur(&planar, pw, ph, RESIDUAL_LF_BLUR_SIGMA);
                    for i in 0..n {
                        lf[i * 3 + c] = blurred[i];
                    }
                }
            }
            for i in 0..n {
                let an = (alpha_pad[i] / a_max).clamp(0.0, 1.0);
                if an <= 1e-6 {
                    continue;
                }
                let o = i * 3;
                let ly = 0.299 * local[o] + 0.587 * local[o + 1] + 0.114 * local[o + 2];
                // Skip pixels already handled by dark-local flatten.
                if ly < DARK_BG_LUMA {
                    continue;
                }
                let lf_y = 0.299 * lf[o] + 0.587 * lf[o + 1] + 0.114 * lf[o + 2];
                let t = if lf_y > ext_y + 0.5 {
                    let excess = lf_y - ext_y;
                    let gate = (excess / (excess + 8.0)).clamp(0.0, 1.0);
                    atten * an * gate
                } else if lf_y < ext_y - 1.0 {
                    let deficit = ext_y - lf_y;
                    let gate = (deficit / (deficit + 8.0)).clamp(0.0, 1.0);
                    DARK_HOLE_PULL * an * bright_f * gate
                } else {
                    atten * 0.35 * an
                };
                if t <= 1e-6 {
                    continue;
                }
                let t = t.clamp(0.0, 0.75);
                for c in 0..3 {
                    let hf = out[o + c] - lf[o + c];
                    let new_lf = lf[o + c] * (1.0 - t) + ext_mean[c] * t;
                    out[o + c] = (new_lf + hf).clamp(0.0, 255.0);
                }
            }
        }
    }

    // --- HF grain from exterior of *original* ROI ---
    let strength = POST_GRAIN_STRENGTH.clamp(0.0, 1.5);
    if strength <= 0.0 {
        return;
    }
    let mut blur_src = vec![0f32; n * 3];
    for i in 0..n * 3 {
        blur_src[i] = orig[i] as f32;
    }
    // Reuse gaussian on each channel via planar temp
    let mut noise = vec![0f32; n * 3];
    {
        let mut planar = vec![0f32; n];
        for c in 0..3 {
            for i in 0..n {
                planar[i] = blur_src[i * 3 + c];
            }
            let blurred = gaussian_blur(&planar, pw, ph, POST_GRAIN_BLUR_SIGMA);
            for i in 0..n {
                noise[i * 3 + c] = planar[i] - blurred[i];
            }
        }
    }

    // Target exterior noise std per channel.
    let mut ext_sum = [0f32; 3];
    let mut ext_sq = [0f32; 3];
    for i in 0..n {
        if ext[i] < 0.5 {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            let v = noise[o + c];
            ext_sum[c] += v;
            ext_sq[c] += v * v;
        }
    }
    let mut target_std = [0f32; 3];
    for c in 0..3 {
        let mean = ext_sum[c] / ext_n;
        target_std[c] = ((ext_sq[c] / ext_n) - mean * mean)
            .max(0.0)
            .sqrt()
            .max(1e-3);
    }

    // Zero noise inside footprint before guiding (exterior-only sources).
    let mut synth = noise.clone();
    for i in 0..n {
        if ext[i] < 0.5 {
            let o = i * 3;
            synth[o] = 0.0;
            synth[o + 1] = 0.0;
            synth[o + 2] = 0.0;
        }
    }

    // Guide exterior noise into the ROI (Gaussian on exterior-weighted field).
    let mut guided = vec![0f32; n * 3];
    {
        let mut planar = vec![0f32; n];
        let wplane = ext.clone();
        // blur weight for normalization
        let wblur = gaussian_blur(&wplane, pw, ph, POST_GRAIN_GUIDE_SIGMA);
        for c in 0..3 {
            for i in 0..n {
                planar[i] = synth[i * 3 + c] * ext[i];
            }
            let num = gaussian_blur(&planar, pw, ph, POST_GRAIN_GUIDE_SIGMA);
            for i in 0..n {
                guided[i * 3 + c] = num[i] / (wblur[i] + 1e-6);
            }
        }
    }

    // Scale guided grain to exterior std, add into weighted interior.
    let mut g_sum = [0f32; 3];
    let mut g_sq = [0f32; 3];
    let mut g_n = 0f32;
    for i in 0..n {
        if weight[i] <= 0.05 {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            let v = guided[o + c];
            g_sum[c] += v;
            g_sq[c] += v * v;
        }
        g_n += 1.0;
    }
    if g_n < 4.0 {
        return;
    }
    let mut scale = [1f32; 3];
    for c in 0..3 {
        let mean = g_sum[c] / g_n;
        let std = ((g_sq[c] / g_n) - mean * mean).max(0.0).sqrt().max(1e-3);
        // Tighter clamp than before — reduces grain-amount flicker across frames.
        scale[c] = (target_std[c] / std).clamp(0.45, 1.55);
    }

    // Measure interior HF deficit vs exterior and boost grain accordingly.
    let mut int_sum = [0f32; 3];
    let mut int_sq = [0f32; 3];
    let mut int_n = 0f32;
    {
        let mut planar = vec![0f32; n];
        let mut out_noise = vec![0f32; n * 3];
        for c in 0..3 {
            for i in 0..n {
                planar[i] = out[i * 3 + c];
            }
            let blurred = gaussian_blur(&planar, pw, ph, POST_GRAIN_BLUR_SIGMA);
            for i in 0..n {
                out_noise[i * 3 + c] = planar[i] - blurred[i];
            }
        }
        for i in 0..n {
            if weight[i] <= 0.05 {
                continue;
            }
            let o = i * 3;
            for c in 0..3 {
                let v = out_noise[o + c];
                int_sum[c] += v;
                int_sq[c] += v * v;
            }
            int_n += 1.0;
        }
    }
    if int_n > 4.0 {
        for c in 0..3 {
            let mean = int_sum[c] / int_n;
            let istd = ((int_sq[c] / int_n) - mean * mean).max(0.0).sqrt();
            // If interior already has enough HF, dial grain down.
            let deficit = (target_std[c] - istd).max(0.0) / target_std[c];
            // Milder deficit curve — less frame-to-frame grain pop.
            scale[c] *= deficit.clamp(0.0, 1.0) * 0.70 + 0.20;
        }
    }

    for i in 0..n {
        let wi = weight[i];
        if wi <= 1e-6 {
            continue;
        }
        let o = i * 3;
        for c in 0..3 {
            let grain = guided[o + c] * scale[c] * strength * wi;
            out[o + c] = (out[o + c] + grain).clamp(0.0, 255.0);
        }
    }
}

/// Motion-gated temporal EMA on the diamond ROI (high-α interior only).
/// Runs sequentially after the parallel FDnCNN pass so residual tip contrast
/// cannot flicker independently of exterior scene motion.
fn temporal_smooth_rois(
    frames: &mut [Vec<u8>],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
) {
    let ema = TEMPORAL_EMA.clamp(0.0, 0.85);
    if ema <= 1e-4 || frames.is_empty() {
        return;
    }
    let mw = map.width as usize;
    let mh = map.height as usize;
    let x = det.x as usize;
    let y = det.y as usize;
    let sw = width as usize;
    let sh = height as usize;
    if x + mw > sw || y + mh > sh {
        return;
    }
    let mut a_max = 1e-6f32;
    for &a in &map.alpha {
        a_max = a_max.max(a);
    }
    let nbytes = sw.saturating_mul(sh).saturating_mul(4);
    let mut prev: Option<Vec<f32>> = None; // RGB float mw*mh*3
    for rgba in frames.iter_mut() {
        if rgba.len() != nbytes {
            prev = None;
            continue;
        }
        let mut curr = vec![0f32; mw * mh * 3];
        for row in 0..mh {
            for col in 0..mw {
                let src = ((y + row) * sw + (x + col)) * 4;
                let dst = (row * mw + col) * 3;
                curr[dst] = rgba[src] as f32;
                curr[dst + 1] = rgba[src + 1] as f32;
                curr[dst + 2] = rgba[src + 2] as f32;
            }
        }
        if let Some(ref prev_roi) = prev {
            // Exterior motion from low-α pixels (AABB corners / outside diamond).
            let mut motion_acc = 0f32;
            let mut motion_n = 0f32;
            for i in 0..mw * mh {
                if map.alpha[i] >= 0.02 {
                    continue;
                }
                let o = i * 3;
                for c in 0..3 {
                    motion_acc += (curr[o + c] - prev_roi[o + c]).abs();
                    motion_n += 1.0;
                }
            }
            let motion = if motion_n > 0.0 {
                motion_acc / motion_n
            } else {
                0.0
            };
            let gate = (1.0 - (motion / TEMPORAL_MOTION_SCALE).clamp(0.0, 0.9)).max(0.0);
            let a = ema * gate;
            if a > 1e-4 {
                for i in 0..mw * mh {
                    let an = (map.alpha[i] / a_max).clamp(0.0, 1.0);
                    let w = a * an;
                    if w <= 1e-6 {
                        continue;
                    }
                    let o = i * 3;
                    for c in 0..3 {
                        curr[o + c] = curr[o + c] * (1.0 - w) + prev_roi[o + c] * w;
                    }
                }
            }
        }
        // Write back.
        for row in 0..mh {
            for col in 0..mw {
                let src = (row * mw + col) * 3;
                let dst = ((y + row) * sw + (x + col)) * 4;
                for c in 0..3 {
                    rgba[dst + c] = curr[src + c].round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        prev = Some(curr);
    }
}

fn default_weight_paths() -> (PathBuf, PathBuf) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    (
        manifest.join("assets/video/fdncnn/fdncnn.param.bin"),
        manifest.join("assets/video/fdncnn/fdncnn.bin"),
    )
}

/// Try in-process FDnCNN postpass on all frames. Returns false if Net cannot load.
pub fn fdncnn_postpass_inprocess(
    frames: &mut [Vec<u8>],
    width: u32,
    height: u32,
    det: &VideoDetection,
    map: &VideoMap,
) -> bool {
    let (param, model) = default_weight_paths();
    if !param.is_file() || !model.is_file() {
        return false;
    }
    // Probe-load on this thread; if it fails, caller falls back to Python.
    if FdncnnNet::load(&param, &model, 1).is_none() {
        return false;
    }
    let (weight, active) = cached_weight(map);
    if active == 0 {
        return true;
    }

    let nbytes = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    frames.par_iter_mut().for_each(|rgba| {
        if rgba.len() != nbytes {
            return;
        }
        let _ = with_tls_net(&param, &model, |net| {
            process_frame_rgba(rgba, width, height, det, map, &weight, net)
        });
    });
    // Sequential temporal EMA — must follow the parallel denoise pass.
    temporal_smooth_rois(frames, width, height, det, map);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::maps::diamond_map_720p_standard;

    #[test]
    fn footprint_weight_720p_active_2304() {
        let map = diamond_map_720p_standard();
        let (_w, active) =
            footprint_weight(&map.alpha, map.width as usize, map.height as usize, 1.8);
        assert_eq!(
            active, 2304,
            "expected full 48×48 footprint at strength 1.8"
        );
    }

    #[test]
    fn net_loads_and_runs_tiny() {
        let (param, model) = default_weight_paths();
        if !param.is_file() {
            eprintln!("skip: missing weights");
            return;
        }
        let net = FdncnnNet::load(&param, &model, 2).expect("load net");
        let w = 16usize;
        let h = 16usize;
        let rgb = vec![128u8; w * h * 3];
        let out = net.run_fdncnn(&rgb, w, h, 75.0).expect("infer");
        assert_eq!(out.len(), w * h * 3);
        // Denoised mid-gray should stay near mid-gray.
        let mean = out.iter().map(|&v| v as f64).sum::<f64>() / out.len() as f64;
        assert!(mean > 50.0 && mean < 200.0, "mean={mean}");
    }

    #[test]
    fn alpha_soft_gate_kills_aabb_corners() {
        assert!(alpha_soft_gate(0.0) <= 1e-6);
        assert!(alpha_soft_gate(0.008) <= 1e-6);
        assert!(alpha_soft_gate(0.22) >= 0.99);
        assert!(alpha_soft_gate(0.04) > 0.2 && alpha_soft_gate(0.04) < 0.8);
    }

    #[test]
    fn corner_gate_spares_tips_kills_corners() {
        let map = diamond_map_720p_standard();
        let (raw, active_raw) =
            footprint_weight(&map.alpha, map.width as usize, map.height as usize, 1.8);
        assert_eq!(active_raw, 2304);
        let mw = map.width as usize;
        let mh = map.height as usize;
        let mut corner = 0f32;
        let mut tip = 0f32;
        for row in 0..mh {
            for col in 0..mw {
                let i = row * mw + col;
                let g = raw[i] * corner_alpha_gate(map.alpha[i], col, row, mw, mh);
                if (row < 2 || row + 2 >= mh) && (col < 2 || col + 2 >= mw) {
                    corner = corner.max(g);
                }
            }
        }
        // Mid-side tips
        for &(col, row) in &[
            (mw / 2, 0usize),
            (0, mh / 2),
            (mw / 2, mh - 1),
            (mw - 1, mh / 2),
        ] {
            let i = row * mw + col;
            tip = tip.max(raw[i] * corner_alpha_gate(map.alpha[i], col, row, mw, mh));
        }
        assert!(corner < 0.15, "AABB corners attenuated, got {corner}");
        assert!(tip > 0.85, "diamond tips must stay strong, got {tip}");
    }

    #[test]
    fn post_off_and_feather_exact() {
        // Classical post off; feather σ=1; temporal EMA off.
        assert_eq!(POST_GRAIN_STRENGTH, 0.0);
        assert_eq!(POST_LUMA_MATCH, 0.0);
        assert_eq!(RESIDUAL_LF_ATTEN, 0.0);
        assert_eq!(DARK_FLAT_STRENGTH, 0.0);
        assert!((WEIGHT_FEATHER_SIGMA - 1.0).abs() < 1e-6);
        assert_eq!(TEMPORAL_EMA, 0.0);
        assert_eq!(SIGMA, 75.0);
        assert!((STRENGTH - 1.8).abs() < 1e-6);
        assert_eq!(PADDING, 64);
    }
}
