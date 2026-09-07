//! Linker extras for static ncnn must be per-target and must not assume libgomp.
use gemini_sparkle_remove::native_link::{ncnn_link_spec, NcnnExtraLib};

#[test]
fn linux_gnu_links_pthread_libstdcxx_not_gomp() {
    let s = ncnn_link_spec("x86_64-unknown-linux-gnu");
    assert_eq!(s.lib_filename, "libncnn.a");
    let names: Vec<_> = s.extra_libs.iter().map(|l| l.name).collect();
    assert!(names.contains(&"pthread"), "{names:?}");
    assert!(names.contains(&"stdc++"), "{names:?}");
    assert!(!names.iter().any(|n| *n == "gomp"), "{names:?}");
}

#[test]
fn apple_links_libcxx_not_gomp() {
    let s = ncnn_link_spec("aarch64-apple-darwin");
    assert_eq!(s.lib_filename, "libncnn.a");
    assert_eq!(
        s.extra_libs,
        &[NcnnExtraLib {
            kind: "dylib",
            name: "c++"
        }]
    );
}

#[test]
fn windows_msvc_uses_ncnn_lib_no_pthread_or_gomp() {
    let s = ncnn_link_spec("x86_64-pc-windows-msvc");
    assert_eq!(s.lib_filename, "ncnn.lib");
    assert!(s.extra_libs.is_empty(), "{:?}", s.extra_libs);
}

#[test]
fn windows_gnu_uses_archive_and_libstdcxx() {
    let s = ncnn_link_spec("x86_64-pc-windows-gnu");
    assert_eq!(s.lib_filename, "libncnn.a");
    let names: Vec<_> = s.extra_libs.iter().map(|l| l.name).collect();
    assert!(names.contains(&"stdc++"), "{names:?}");
}
