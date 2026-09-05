//! Object detection: every frame passes through untouched, with one row per
//! object beside it - the class as COCO label text, the confidence, and the
//! box in the frame's own pixels.
//!
//! The graph is RF-DETR Large's export, run through `wasi:nn`. A DETR head
//! returns a fixed set of queries with no duplicates among them, so there is
//! no NMS: decoding is a sigmoid over each query's class logits, a threshold,
//! and a coordinate map. The module never opens a file - the host binds the
//! graph to a name with `-nn detect=<path>` and this module asks for that
//! name and nothing else.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:rfdetr-detect/detect",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use rfdetr_common::{class_name, frame_box, le_f32s, sigmoid, to_input, PixFmt};
use serde::{Deserialize, Serialize};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn detect=<path>`.
const MODEL: &str = "detect";

/// What the export calls its input tensor.
const INPUT_NAME: &str = "input";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// The square the graph is run at. The export is static at this size.
const SIDE: usize = 704;

/// Class logits per query: COCO's 91-slot category numbering.
const CLASSES: usize = 91;

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

/// One row per detection, the box in the frame's own pixels.
#[derive(Serialize)]
struct Row {
    class: String,
    conf: f64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// One object out of the graph's queries, already on the frame's own axes.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Found {
    class: usize,
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
    pix_fmt: PixFmt,
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
        serde_json::from_str(trimmed).map_err(|e| format!("detect cannot read its params: {e}"))?
    };
    if !parsed.conf.is_finite() || !(0.0..=1.0).contains(&parsed.conf) {
        return Err(format!(
            "detect needs conf between 0 and 1, got {}",
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
    format!("detect: {what}: {code} ({})", error.data())
}

/// Which returned tensor is the boxes and which the class logits, by shape:
/// both are rank 3 over the same queries, and the last dimension tells them
/// apart. Names are not read, so an export that spells them differently - or
/// hands them back in the other order - still resolves.
fn outputs(shapes: &[Vec<u32>]) -> Result<(usize, usize), String> {
    let by_channels = |want: usize| {
        shapes.iter().position(
            |dimensions| matches!(dimensions.as_slice(), [_, _, channels] if *channels as usize == want),
        )
    };
    match (by_channels(BOX_CHANNELS), by_channels(CLASSES)) {
        (Some(boxes), Some(logits)) => Ok((boxes, logits)),
        _ => Err(format!(
            "detect: the graph returned {shapes:?}, and this module wants \
             RF-DETR's [1, queries, {BOX_CHANNELS}] boxes beside \
             [1, queries, {CLASSES}] class logits"
        )),
    }
}

/// The queries thresholded and brought onto the frame. Each query's box is
/// `cx, cy, w, h` as a fraction of the picture, and its class is whichever of
/// the logits is largest once through the logistic curve. A DETR head returns
/// no duplicates, so a query that clears the threshold is a row.
fn decode(
    boxes: &[f32],
    logits: &[f32],
    conf: f32,
    width: usize,
    height: usize,
) -> Result<Vec<Found>, String> {
    if boxes.len() / BOX_CHANNELS != logits.len() / CLASSES {
        return Err(format!(
            "detect: the graph returned {} boxes and {} query logits",
            boxes.len() / BOX_CHANNELS,
            logits.len() / CLASSES
        ));
    }
    let mut found = Vec::new();
    for (query, scores) in logits.as_chunks::<CLASSES>().0.iter().enumerate() {
        let mut class = 0;
        let mut best = f32::NEG_INFINITY;
        for (index, logit) in scores.iter().enumerate() {
            if *logit > best {
                best = *logit;
                class = index;
            }
        }
        let score = sigmoid(best);
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
        found.push(Found {
            class,
            conf: score,
            x: x0 as u32,
            y: y0 as u32,
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        });
    }
    Ok(found)
}

/// One detection's row, as the NDJSON line that rides its frame.
fn to_row(found: &Found) -> String {
    serde_json::to_string(&Row {
        class: class_name(found.class),
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
    let input = to_input(frame, opened.pix_fmt, opened.width, opened.height, SIDE);
    let returned = compute(opened, &input)?;
    let tensors: Vec<Tensor> = returned.into_iter().map(|(_, tensor)| tensor).collect();
    let shapes: Vec<Vec<u32>> = tensors.iter().map(Tensor::dimensions).collect();
    let (boxes, logits) = outputs(&shapes)?;
    let found = decode(
        &le_f32s(&tensors[boxes].data()),
        &le_f32s(&tensors[logits].data()),
        opened.params.conf as f32,
        opened.width,
        opened.height,
    )?;
    Ok(found.iter().map(to_row).collect())
}

struct Detect;

impl Guest for Detect {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "detect".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["yuv420p".to_string(), "rgba".to_string()],
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
            return Err("detect reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = PixFmt::parse(&video.pix_fmt, "detect")?;
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
                pix_fmt,
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

export!(Detect);

#[cfg(test)]
mod tests {
    use super::*;

    /// What the real export returned for one 720x576 frame, captured from a
    /// run of the pinned graph: the five surest queries, each a box as a
    /// fraction of the picture beside the class logit that won it. Three
    /// people, a chair, a screen. The fixture is the decode's contract, not
    /// the model's.
    const CAPTURED: [(f32, f32, f32, f32, usize, f32); 5] = [
        (0.264, 0.548, 0.307, 0.898, 1, 2.976),
        (0.621, 0.857, 0.270, 0.279, 62, 1.966),
        (0.523, 0.824, 0.234, 0.347, 1, 1.629),
        (0.092, 0.901, 0.147, 0.196, 1, 0.364),
        (0.467, 0.414, 0.905, 0.644, 72, -0.253),
    ];

    /// The captured queries as the two tensors the graph hands back, plus the
    /// quiet queries a DETR head always returns beside them.
    fn captured(padding: usize) -> (Vec<f32>, Vec<f32>) {
        let mut boxes = Vec::new();
        let mut logits = Vec::new();
        for (cx, cy, w, h, class, logit) in CAPTURED {
            boxes.extend([cx, cy, w, h]);
            let mut scores = vec![-10.0f32; CLASSES];
            scores[class] = logit;
            logits.extend(scores);
        }
        for _ in 0..padding {
            boxes.extend([0.5, 0.5, 0.1, 0.1]);
            logits.extend(vec![-10.0f32; CLASSES]);
        }
        (boxes, logits)
    }

    fn decoded(conf: f32) -> Vec<Found> {
        let (boxes, logits) = captured(0);
        decode(&boxes, &logits, conf, 720, 576).expect("the tensors agree")
    }

    #[test]
    fn the_captured_tensors_decode_to_one_row_per_object() {
        let found = decoded(0.25);
        assert_eq!(found.len(), 5, "every captured query clears 0.25");
        assert_eq!(found[0].class, 1, "the surest object is a person");
        assert_eq!(found[1].class, 62, "and the second is the chair");
        for object in &found {
            assert!(object.x + object.w <= 720, "boxes stay on the picture");
            assert!(object.y + object.h <= 576);
        }
        // The centre is 0.264 of 720 and the width 0.307, so the left edge is
        // (0.264 - 0.1535) * 720.
        assert_eq!(found[0].x, 79);
        assert_eq!(found[0].y, 57);
    }

    #[test]
    fn the_threshold_drops_what_scores_under_it() {
        let found = decoded(0.5);
        assert_eq!(found.len(), 4, "the screen, at 0.437, is the one that goes");
        assert!(found.iter().all(|object| object.conf >= 0.5));
    }

    #[test]
    fn the_quiet_queries_a_detr_head_returns_never_decode() {
        // The head always returns its full set; the ones that found nothing
        // score far under any threshold.
        let (boxes, logits) = captured(295);
        let found = decode(&boxes, &logits, 0.25, 720, 576).expect("the tensors agree");
        assert_eq!(found.len(), 5);
    }

    #[test]
    fn a_box_naming_no_pixel_of_the_picture_is_dropped() {
        let boxes = [1.6, 0.5, 0.2, 0.2];
        let mut logits = vec![-10.0f32; CLASSES];
        logits[1] = 5.0;
        assert!(decode(&boxes, &logits, 0.25, 720, 576)
            .expect("the tensors agree")
            .is_empty());
    }

    #[test]
    fn tensors_that_disagree_on_how_many_queries_are_refused_by_name() {
        let boxes = [0.5, 0.5, 0.2, 0.2, 0.4, 0.4, 0.2, 0.2];
        let logits = vec![-10.0f32; CLASSES];
        let error = decode(&boxes, &logits, 0.25, 720, 576).expect_err("two boxes, one query");
        assert!(error.starts_with("detect: "), "{error}");
    }

    #[test]
    fn a_row_spells_the_class_as_text_and_rounds_the_confidence() {
        assert_eq!(
            to_row(&Found {
                class: 1,
                conf: 0.95132,
                x: 79,
                y: 57,
                w: 221,
                h: 517,
            }),
            r#"{"class":"person","conf":0.9513,"x":79,"y":57,"w":221,"h":517}"#
        );
    }

    #[test]
    fn the_two_returned_tensors_are_told_apart_by_shape_in_either_order() {
        assert_eq!(
            outputs(&[vec![1, 300, 4], vec![1, 300, 91]]).expect("both found"),
            (0, 1)
        );
        assert_eq!(
            outputs(&[vec![1, 300, 91], vec![1, 300, 4]]).expect("both found"),
            (1, 0)
        );
        let error = outputs(&[vec![1, 84, 8400]]).expect_err("a dense grid");
        assert!(error.starts_with("detect: "), "{error}");
    }

    #[test]
    fn params_default_to_the_threshold_the_schema_publishes() {
        let parsed = parse_params("").expect("empty is the defaults");
        assert_eq!(parsed.conf, 0.25);
        let braces = parse_params("{}").expect("and so is an empty object");
        assert_eq!(braces.conf, 0.25);
    }

    #[test]
    fn params_outside_zero_to_one_are_refused_by_name() {
        for bad in [r#"{"conf":1.5}"#, r#"{"conf":-0.1}"#] {
            let error = parse_params(bad).expect_err(bad);
            assert!(error.starts_with("detect "), "{error}");
        }
        assert!(
            parse_params(r#"{"radius":3}"#).is_err(),
            "and so is a param this module has none of"
        );
    }
}
