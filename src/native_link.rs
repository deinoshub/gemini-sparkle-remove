// Per-target linker extras for static ncnn (no libgomp).
// Included from `build.rs` as well as the library so the table stays in one place.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcnnLinkSpec {
    /// Filename cmake produces (`libncnn.a` or `ncnn.lib`).
    pub lib_filename: &'static str,
    pub extra_libs: &'static [NcnnExtraLib],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NcnnExtraLib {
    /// `dylib` → `cargo:rustc-link-lib=dylib=NAME`; empty → `cargo:rustc-link-lib=NAME`.
    pub kind: &'static str,
    pub name: &'static str,
}

pub fn ncnn_link_spec(target: &str) -> NcnnLinkSpec {
    if target.contains("windows") {
        if target.contains("gnu") {
            NcnnLinkSpec {
                lib_filename: "libncnn.a",
                extra_libs: &[NcnnExtraLib {
                    kind: "dylib",
                    name: "stdc++",
                }],
            }
        } else {
            NcnnLinkSpec {
                lib_filename: "ncnn.lib",
                extra_libs: &[],
            }
        }
    } else if target.contains("apple") {
        NcnnLinkSpec {
            lib_filename: "libncnn.a",
            extra_libs: &[NcnnExtraLib {
                kind: "dylib",
                name: "c++",
            }],
        }
    } else {
        NcnnLinkSpec {
            lib_filename: "libncnn.a",
            extra_libs: &[
                NcnnExtraLib {
                    kind: "",
                    name: "pthread",
                },
                NcnnExtraLib {
                    kind: "dylib",
                    name: "stdc++",
                },
            ],
        }
    }
}
