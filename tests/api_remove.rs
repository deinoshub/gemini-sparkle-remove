//! Integration tests for the public remove API.

use gemini_unmark::{remove_at, remove_gemini_sparkle, RemoveResult, RgbaImage};

fn load_rgba(path: &str) -> (u32, u32, Vec<u8>) {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let rgba = img.to_rgba8();
    (rgba.width(), rgba.height(), rgba.into_raw())
}

#[test]
fn remove_gemini_sparkle_on_fixture() {
    let (w, h, mut data) = load_rgba("tests/fixtures/sparkle_sample.jpeg");
    let before = data.clone();
    let mut img = RgbaImage {
        width: w,
        height: h,
        data: &mut data,
    };
    let result = remove_gemini_sparkle(&mut img);
    match result {
        RemoveResult::Removed { x, y } => {
            // Expected sparkle near (1255, 647) on 1376×768.
            assert!((x as i32 - 1255).abs() <= 20, "x={x}");
            assert!((y as i32 - 647).abs() <= 20, "y={y}");
            assert_ne!(data, before, "buffer should change when mark is removed");
        }
        RemoveResult::NotFound => panic!("expected Removed on sparkle fixture"),
    }
}

#[test]
fn unmarked_solid_returns_not_found_unchanged() {
    let w = 200u32;
    let h = 200u32;
    let mut data = vec![40u8, 80, 120, 255]
        .into_iter()
        .cycle()
        .take((w * h * 4) as usize)
        .collect::<Vec<_>>();
    let before = data.clone();
    let mut img = RgbaImage {
        width: w,
        height: h,
        data: &mut data,
    };
    let result = remove_gemini_sparkle(&mut img);
    assert!(matches!(result, RemoveResult::NotFound));
    assert_eq!(data, before, "NotFound must leave buffer unchanged");
}

#[test]
fn remove_at_changes_pixels_under_template() {
    // Solid mid-grey; force-blend at (10,10) must alter some pixels in the 48×48 footprint.
    let w = 100u32;
    let h = 100u32;
    let mut data = vec![128u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        data[i * 4 + 3] = 255;
    }
    let before = data.clone();
    let mut img = RgbaImage {
        width: w,
        height: h,
        data: &mut data,
    };
    remove_at(&mut img, 10, 10);
    assert_ne!(
        data, before,
        "remove_at should reverse-blend the sparkle footprint"
    );
}
