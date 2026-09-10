//! Licence plate detection: every frame passes through untouched, with one
//! row per plate beside it - the class always `plate`, the confidence, and
//! the box in the frame's own pixels. The rows are `detect`'s rows, so
//! `boxes_mask` and `draw_boxes` read them without knowing which detector
//! wrote them.
//!
//! The graph is RF-DETR Medium fine-tuned on one class, run through
//! `wasi:nn`. A DETR head returns a fixed set of queries with no duplicates
//! among them, so there is no NMS: decoding is a sigmoid over each query's
//! plate logit, a threshold, and a coordinate map. The module never opens a
//! file - the host binds the graph to a name with `-nn detect_plates=<path>`
//! and this module asks for that name and nothing else.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:rfdetr-detect-plates/detect-plates",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use ffrwd_frame::{Filter, Rect, Rgba, IMAGENET};
use rfdetr_common::{frame_box, le_f32s, sigmoid};
use serde::{Deserialize, Serialize};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn detect_plates=<path>`.
const MODEL: &str = "detect_plates";

/// What the export calls its input tensor.
const INPUT_NAME: &str = "input";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// The square the graph is run at. The export is static at this size.
const SIDE: usize = 576;

/// Class logits per query. The head was trained one class wide by `rfdetr`,
/// which lays the head out as the classes plus one spare slot, so the export
/// carries two logits and the plate is the first. There is nothing to take
/// an argmax over: the spare slot was never trained toward anything.
const CLASSES: usize = 2;

/// Which of a query's logits is the plate.
const PLATE_LOGIT: usize = 0;

/// What every row's class says. The graph knows one thing.
const PLATE: &str = "plate";

/// A box's channels: centre, width and height, each a fraction of the picture.
const BOX_CHANNELS: usize = 4;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25}},"additionalProperties":false}"#;

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number"},"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}},"required":["class","conf","x","y","w","h"],"additionalProperties":false}"#;

fn default_conf() -> f64 {
    0.25
}

#[derive(Clone, Copy, Debug, Deserialize)]
// The schema says this one and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "default_conf")]
    conf: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            conf: default_conf(),
        }
    }
}

/// One row per plate, the box in the frame's own pixels.
#[derive(Serialize)]
struct Row {
    class: String,
    conf: f64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// One plate out of the graph's queries, already on the frame's own axes. It
/// carries no class: every query this head answers is the one class.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Found {
    conf: f32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    width: usize,
    height: usize,
    params: Params,
    /// What the graph calls its input, settled by the first call that works.
    input_name: Cell<&'static str>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed)
            .map_err(|e| format!("detect_plates cannot read its params: {e}"))?
    };
    if !parsed.conf.is_finite() || !(0.0..=1.0).contains(&parsed.conf) {
        return Err(format!(
            "detect_plates needs conf between 0 and 1, got {}",
            parsed.conf
        ));
    }
    Ok(parsed)
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("detect_plates: {what}: {code} ({})", error.data())
}

/// Which returned tensor is the boxes and which the class logits, by shape:
/// both are rank 3 over the same queries, and the last dimension - four
/// coordinates against the one class - tells them apart. Names are not read,
/// so an export that spells them differently - or hands them back in the
/// other order - still resolves.
fn outputs(shapes: &[Vec<u32>]) -> Result<(usize, usize), String> {
    let by_channels = |want: usize| {
        shapes.iter().position(
            |dimensions| matches!(dimensions.as_slice(), [_, _, channels] if *channels as usize == want),
        )
    };
    match (by_channels(BOX_CHANNELS), by_channels(CLASSES)) {
        (Some(boxes), Some(logits)) => Ok((boxes, logits)),
        _ => Err(format!(
            "detect_plates: the graph returned {shapes:?}, and this module wants \
             RF-DETR's [1, queries, {BOX_CHANNELS}] boxes beside \
             [1, queries, {CLASSES}] class logits"
        )),
    }
}

/// The queries thresholded and brought onto the frame. Each query's box is
/// `cx, cy, w, h` as a fraction of the picture, and its confidence is its
/// plate logit through the logistic curve. A DETR head returns no duplicates,
/// so a query that clears the threshold is a row.
fn decode(
    boxes: &[f32],
    logits: &[f32],
    conf: f32,
    width: usize,
    height: usize,
) -> Result<Vec<Found>, String> {
    if boxes.len() / BOX_CHANNELS != logits.len() / CLASSES {
        return Err(format!(
            "detect_plates: the graph returned {} boxes and {} query logits",
            boxes.len() / BOX_CHANNELS,
            logits.len() / CLASSES
        ));
    }
    let mut found = Vec::new();
    for (query, logit) in logits.iter().skip(PLATE_LOGIT).step_by(CLASSES).enumerate() {
        let score = sigmoid(*logit);
        if score < conf {
            continue;
        }
        let at = query * BOX_CHANNELS;
        let (x0, y0, x1, y1) = frame_box(
            (boxes[at], boxes[at + 1], boxes[at + 2], boxes[at + 3]),
            width,
            height,
        );
        if x0 == x1 || y0 == y1 {
            continue;
        }
        let (w, h) = (x1 - x0, y1 - y0);
        found.push(Found {
            conf: score,
            x: x0 as u32,
            y: y0 as u32,
            w: w as u32,
            h: h as u32,
        });
    }
    Ok(found)
}

/// One plate's row, as the NDJSON line that rides its frame.
fn to_row(found: &Found) -> String {
    serde_json::to_string(&Row {
        class: PLATE.to_string(),
        // To four places: the graph's own precision is nowhere near the
        // sixteen digits an f32 widened to an f64 prints.
        conf: (f64::from(found.conf) * 10_000.0).round() / 10_000.0,
        x: found.x,
        y: found.y,
        w: found.w,
        h: found.h,
    })
    .expect("row serializes")
}

/// One frame through the graph, however the graph names its input.
fn compute(opened: &Opened, input: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    let dimensions = [1, 3, SIDE as u32, SIDE as u32];
    let name = opened.input_name.get();
    let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
    match opened.context.compute(vec![(name.to_string(), tensor)]) {
        Ok(returned) => Ok(returned),
        // An export whose input is not called what this one calls it. The
        // host takes a position where it takes a name, so the retry names none,
        // and the name that worked is kept for every frame after this one.
        Err(_) if name == INPUT_NAME => {
            opened.input_name.set(INPUT_INDEX);
            let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
            opened
                .context
                .compute(vec![(INPUT_INDEX.to_string(), tensor)])
                .map_err(|e| failed("compute", &e))
        }
        Err(e) => Err(failed("compute", &e)),
    }
}

/// One frame in, its rows out.
fn run(opened: &Opened, frame: &[u8]) -> Result<Vec<String>, String> {
    let (width, height) = (opened.width, opened.height);
    let input = ffrwd_frame::tensor(
        &Rgba::new(frame, width, height)?,
        Rect::whole(width, height),
        SIDE,
        SIDE,
        Filter::Bilinear,
        IMAGENET,
    );
    let returned = compute(opened, &input)?;
    let tensors: Vec<Tensor> = returned.into_iter().map(|(_, tensor)| tensor).collect();
    let shapes: Vec<Vec<u32>> = tensors.iter().map(Tensor::dimensions).collect();
    let (boxes, logits) = outputs(&shapes)?;
    let found = decode(
        &le_f32s(&tensors[boxes].data()),
        &le_f32s(&tensors[logits].data()),
        opened.params.conf as f32,
        width,
        height,
    )?;
    Ok(found.iter().map(to_row).collect())
}

struct DetectPlates;

impl Guest for DetectPlates {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "detect_plates".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            pure: true,
            one_to_one: true,
            reads_rows: false,
            // The rows leaving are this module's own detections.
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("detect_plates reads frames, and this stream is audio".to_string());
        };
        if video.pix_fmt != "rgba" {
            return Err(format!(
                "detect_plates does not accept pixel format {}",
                video.pix_fmt
            ));
        }
        let parsed = parse_params(&params)?;

        // The graph is loaded once per instance, and the session built once:
        // the first frame is what a provider picks its kernels on, and every
        // frame after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                params: parsed,
                input_name: Cell::new(INPUT_NAME),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.params = parsed;
            }
        });
        Ok(())
    }

    fn process(window: &InWindow, _trailing: Vec<String>, _last: bool) -> Processed {
        // The final call carries nothing: window and stride are 1, so no frame
        // is ever left over.
        let mut out = Vec::with_capacity(window.len() as usize);
        OPENED.with(|opened| {
            let borrowed = opened.borrow();
            let opened = borrowed
                .as_ref()
                .expect("init loads the graph before any frame arrives");
            for i in 0..window.len() {
                // `process` has no way to say no, so a graph that failed
                // mid-stream stops the run rather than dropping detections
                // on the floor.
                let rows = run(opened, &window.fetch(i)).unwrap_or_else(|m| panic!("{m}"));
                out.push(OutFrame {
                    pts: window.pts(i),
                    // The picture leaves untouched; the rows are the work.
                    frame: FramePayload::Same,
                    rows,
                });
            }
        });
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(DetectPlates);

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 1280;
    const HEIGHT: usize = 720;

    /// Three queries: a sure plate, a doubtful one, and a query whose spare
    /// slot is the loud one, which must not count.
    fn queries() -> (Vec<f32>, Vec<f32>) {
        let boxes = vec![
            0.5, 0.6, 0.10, 0.04, // a plate mid-frame, wide and low
            0.2, 0.2, 0.05, 0.02, // a doubtful one
            0.8, 0.8, 0.20, 0.20, // the spare slot's answer
        ];
        let logits = vec![
            3.0, -6.0, // plate 0.95, spare quiet
            -1.0, -6.0, // plate 0.27
            -6.0, 4.0, // plate 0.002, spare loud
        ];
        (boxes, logits)
    }

    #[test]
    fn reads_the_plate_logit_and_leaves_the_spare_slot_alone() {
        let (boxes, logits) = queries();
        let found = decode(&boxes, &logits, 0.25, WIDTH, HEIGHT).unwrap();
        assert_eq!(found.len(), 2);
        assert!(found[0].conf > 0.94 && found[0].conf < 0.96);
        // 0.58 and 0.62 of 720 are 417.6 and 446.4; the box takes the whole pixels they touch.
        assert_eq!((found[0].x, found[0].y, found[0].w, found[0].h), (576, 417, 128, 30));
        assert!(found[1].conf > 0.26 && found[1].conf < 0.28);
    }

    #[test]
    fn a_higher_threshold_keeps_only_the_sure_one() {
        let (boxes, logits) = queries();
        let found = decode(&boxes, &logits, 0.5, WIDTH, HEIGHT).unwrap();
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn a_wide_plate_is_not_mistaken_for_a_crowd() {
        // The face module drops a box that is both wide and a quarter of the
        // frame. A plate close to the camera is exactly that, and stays.
        let boxes = vec![0.5, 0.5, 0.8, 0.4];
        let logits = vec![4.0, -6.0];
        let found = decode(&boxes, &logits, 0.25, WIDTH, HEIGHT).unwrap();
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn a_row_says_plate() {
        let row = to_row(&Found { conf: 0.87654, x: 1, y: 2, w: 3, h: 4 });
        assert_eq!(row, r#"{"class":"plate","conf":0.8765,"x":1,"y":2,"w":3,"h":4}"#);
    }

    #[test]
    fn outputs_are_told_apart_by_their_last_dimension() {
        let (boxes, logits) = outputs(&[vec![1, 300, 2], vec![1, 300, 4]]).unwrap();
        assert_eq!((boxes, logits), (1, 0));
        assert!(outputs(&[vec![1, 300, 4], vec![1, 300, 1]]).is_err());
    }
}
