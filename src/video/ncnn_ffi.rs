//! Thin FFI over ncnn `c_api.h` (hand-written bindgen subset).
//!
//! Only the symbols needed for FDnCNN binary-param inference:
//! `load_param_bin`, `input_index(0)`, `extract_index(20)`.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

pub type ncnn_allocator_t = *mut c_void;
pub type ncnn_option_t = *mut c_void;
pub type ncnn_mat_t = *mut c_void;
pub type ncnn_net_t = *mut c_void;
pub type ncnn_extractor_t = *mut c_void;

extern "C" {
    pub fn ncnn_option_create() -> ncnn_option_t;
    pub fn ncnn_option_destroy(opt: ncnn_option_t);
    pub fn ncnn_option_set_num_threads(opt: ncnn_option_t, num_threads: c_int);
    pub fn ncnn_option_set_use_vulkan_compute(opt: ncnn_option_t, use_vulkan_compute: c_int);

    pub fn ncnn_mat_create_3d(
        w: c_int,
        h: c_int,
        c: c_int,
        allocator: ncnn_allocator_t,
    ) -> ncnn_mat_t;
    pub fn ncnn_mat_destroy(mat: ncnn_mat_t);
    pub fn ncnn_mat_get_channel_data(mat: ncnn_mat_t, c: c_int) -> *mut c_void;
    pub fn ncnn_mat_get_w(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_h(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_c(mat: ncnn_mat_t) -> c_int;
    pub fn ncnn_mat_get_cstep(mat: ncnn_mat_t) -> usize;

    pub fn ncnn_net_create() -> ncnn_net_t;
    pub fn ncnn_net_destroy(net: ncnn_net_t);
    pub fn ncnn_net_set_option(net: ncnn_net_t, opt: ncnn_option_t);
    pub fn ncnn_net_get_option(net: ncnn_net_t) -> ncnn_option_t;
    pub fn ncnn_net_load_param_bin(net: ncnn_net_t, path: *const c_char) -> c_int;
    pub fn ncnn_net_load_model(net: ncnn_net_t, path: *const c_char) -> c_int;

    pub fn ncnn_extractor_create(net: ncnn_net_t) -> ncnn_extractor_t;
    pub fn ncnn_extractor_destroy(ex: ncnn_extractor_t);
    pub fn ncnn_extractor_input_index(
        ex: ncnn_extractor_t,
        index: c_int,
        mat: ncnn_mat_t,
    ) -> c_int;
    pub fn ncnn_extractor_extract_index(
        ex: ncnn_extractor_t,
        index: c_int,
        mat: *mut ncnn_mat_t,
    ) -> c_int;

    /// C++ shim: set FDnCNN Option flags (fp16 arith off, packing on, …).
    pub fn gwr_ncnn_configure_fdncnn(net: ncnn_net_t, num_threads: c_int);
}

pub const BLOB_INPUT: c_int = 0;
pub const BLOB_OUTPUT: c_int = 20;
