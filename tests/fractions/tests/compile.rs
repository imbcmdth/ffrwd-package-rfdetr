//! draw_boxes refuses a producer of fractional boxes when the query is
//! compiled, through the shapes the real modules report. It needs the built
//! modules, ffmpeg, and an ffrwd that compiles nodes (0.29 or later), named by
//! `FFRWD_CLI` when it is not `ffrwd` on the path:
//!
//! ```text
//! cargo build --release --target wasm32-wasip2
//! cargo test --release -p fractions -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn package() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn clip() -> PathBuf {
    let clip = std::env::temp_dir().join("rfdetr-fractions-clip.mp4");
    if !clip.exists() {
        let made = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1"])
            .arg(&clip)
            .status()
            .expect("ffmpeg runs");
        assert!(made.success(), "ffmpeg made no clip");
    }
    clip
}

fn compile(recipe: &str) -> Output {
    let cli = std::env::var("FFRWD_CLI").unwrap_or_else(|_| "ffrwd".to_owned());
    let mut words = cli.split_whitespace();
    let mut command = Command::new(words.next().expect("a command"));
    command
        .args(words)
        .current_dir(package())
        .args(["compile", "-f", recipe, "-v"])
        .arg(format!("source={}", clip().display()))
        .args(["-v", "dest=out.mp4"]);
    command.output().expect("ffrwd runs")
}

#[test]
#[ignore = "needs the built modules, ffmpeg and ffrwd 0.29"]
fn draw_boxes_refuses_boxes_in_fractions_of_a_pixel() {
    let refused = compile("tests/draw-fractions.sql");
    let said = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "compiled: {said}");
    assert!(
        said.contains("integer") && said.contains("number"),
        "the refusal names both types: {said}"
    );

    let compiled = compile("tests/draw-detections.sql");
    assert!(
        compiled.status.success(),
        "a detector's whole pixels compile: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}
