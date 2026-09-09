//! Face detection: every frame passes through untouched, with one row per
//! face beside it - the class always `face`, the confidence, and the box in
//! the frame's own pixels. The rows are `detect`'s rows, so `boxes_mask` and
//! `draw_boxes` read them without knowing which detector wrote them.
//!
//! The graph is RF-DETR Medium fine-tuned on one class, run through
//! `wasi:nn`. A DETR head returns a fixed set of queries with no duplicates
//! among them, so there is no NMS: decoding is a sigmoid over each query's
//! one class logit, a threshold, and a coordinate map. The module never opens
//! a file - the host binds the graph to a name with `-nn detect_faces=<path>`
//! and this module asks for that name and nothing else.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:rfdetr-detect-faces/detect-faces",
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

/// The name the host binds the graph to. `-nn detect_faces=<path>`.
const MODEL: &str = "detect_faces";

/// What the export calls its input tensor.
const INPUT_NAME: &str = "input";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// The square the graph is run at. The export is static at this size.
const SIDE: usize = 576;

/// Class logits per query: one, and it is the face. The head was fine-tuned
/// one class wide, so there is no background slot and nothing to take an
/// argmax over.
const CLASSES: usize = 1;

/// What every row's class says. The graph knows one thing.
const FACE: &str = "face";

/// A box's channels: centre, width and height, each a fraction of the picture.
const BOX_CHANNELS: usize = 4;

/// How many times its own height a box has to be to count as wide.
const WIDE_ASPECT: f32 = 1.3;

/// The share of the frame a wide box has to cover to count as large.
const WIDE_AREA: f32 = 0.25;

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

/// One row per face, the box in the frame's own pixels.
#[derive(Serialize)]
struct Row {
    class: String,
    conf: f64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// One face out of the graph's queries, already on the frame's own axes. It
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
            .map_err(|e| format!("detect_faces cannot read its params: {e}"))?
    };
    if !parsed.conf.is_finite() || !(0.0..=1.0).contains(&parsed.conf) {
        return Err(format!(
            "detect_faces needs conf between 0 and 1, got {}",
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
    format!("detect_faces: {what}: {code} ({})", error.data())
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
            "detect_faces: the graph returned {shapes:?}, and this module wants \
             RF-DETR's [1, queries, {BOX_CHANNELS}] boxes beside \
             [1, queries, {CLASSES}] class logits"
        )),
    }
}

/// The queries thresholded and brought onto the frame. Each query's box is
/// `cx, cy, w, h` as a fraction of the picture, and its confidence is its one
/// logit through the logistic curve - the head is one class wide, so there is
/// no argmax to take. A DETR head returns no duplicates, so a query that
/// clears the threshold is a row, bar the whole-frame box it answers a crowd
/// with.
fn decode(
    boxes: &[f32],
    logits: &[f32],
    conf: f32,
    width: usize,
    height: usize,
) -> Result<Vec<Found>, String> {
    if boxes.len() / BOX_CHANNELS != logits.len() / CLASSES {
        return Err(format!(
            "detect_faces: the graph returned {} boxes and {} query logits",
            boxes.len() / BOX_CHANNELS,
            logits.len() / CLASSES
        ));
    }
    let mut found = Vec::new();
    for (query, logit) in logits.iter().enumerate() {
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
        // The head's whole-frame answer, which arrives on a crowd at a middling
        // confidence. A face is taller than it is wide, or near square when the
        // head is tilted, so a box both this wide and this large is not one - a
        // wide box that is small is a face half behind something, and a large
        // box that is tall is a face close to the camera, and both stay.
        let wide = w as f32 > h as f32 * WIDE_ASPECT;
        let large = (w * h) as f32 > (width * height) as f32 * WIDE_AREA;
        if wide && large {
            continue;
        }
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

/// One face's row, as the NDJSON line that rides its frame.
fn to_row(found: &Found) -> String {
    serde_json::to_string(&Row {
        class: FACE.to_string(),
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

struct DetectFaces;

impl Guest for DetectFaces {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "detect_faces".to_string(),
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
            return Err("detect_faces reads frames, and this stream is audio".to_string());
        };
        if video.pix_fmt != "rgba" {
            return Err(format!(
                "detect_faces does not accept pixel format {}",
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

export!(DetectFaces);

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame the capture was taken on: WIDER FACE validation image
    /// 2--Demonstration/2_Demonstration_Protesters_2_738.jpg, ten annotated
    /// faces, five of them over 96 px.
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1428;

    /// What the real export returned for that frame, captured from a run of
    /// the pinned graph: the five surest queries, each a box as a fraction of
    /// the picture beside the one class logit the head carries. The fixture
    /// is the decode's contract, not the model's.
    const CAPTURED: [(f32, f32, f32, f32, f32); 5] = [
        (0.951608, 0.792185, 0.096596, 0.131434, 1.986528),
        (0.055047, 0.925899, 0.110031, 0.147960, 1.979382),
        (0.768078, 0.763318, 0.119112, 0.115945, 1.794836),
        (0.297883, 0.663443, 0.120973, 0.103971, 1.708497),
        (0.327975, 0.066133, 0.058301, 0.058259, 1.286335),
    ];

    /// The captured queries as the two tensors the graph hands back, plus the
    /// quiet queries a DETR head always returns beside them.
    fn captured(padding: usize) -> (Vec<f32>, Vec<f32>) {
        let mut boxes = Vec::new();
        let mut logits = Vec::new();
        for (cx, cy, w, h, logit) in CAPTURED {
            boxes.extend([cx, cy, w, h]);
            logits.push(logit);
        }
        for _ in 0..padding {
            boxes.extend([0.5, 0.5, 0.1, 0.1]);
            logits.push(-10.0);
        }
        (boxes, logits)
    }

    fn decoded(conf: f32) -> Vec<Found> {
        let (boxes, logits) = captured(0);
        decode(&boxes, &logits, conf, WIDTH, HEIGHT).expect("the tensors agree")
    }

    #[test]
    fn the_captured_tensors_decode_to_one_row_per_face() {
        let found = decoded(0.25);
        assert_eq!(found.len(), 5, "every captured query clears 0.25");
        for face in &found {
            assert!(face.x + face.w <= WIDTH as u32, "boxes stay on the picture");
            assert!(face.y + face.h <= HEIGHT as u32);
            assert!(face.conf > 0.78, "the five surest are all well clear");
        }
        // The centre is 0.951608 of 1024 and the width 0.096596, so the left
        // edge is (0.951608 - 0.048298) * 1024 and the right runs off the
        // frame and clips to it.
        assert_eq!((found[0].x, found[0].w), (924, 100));
        // And 0.792185 of 1428 with a height of 0.131434.
        assert_eq!((found[0].y, found[0].h), (1037, 189));
    }

    #[test]
    fn the_threshold_drops_what_scores_under_it() {
        let found = decoded(0.8);
        assert_eq!(found.len(), 4, "the one at 0.7835 is what goes");
        assert!(found.iter().all(|face| face.conf >= 0.8));
        assert!(decoded(0.9).is_empty(), "and past the surest, nothing left");
    }

    #[test]
    fn the_quiet_queries_a_detr_head_returns_never_decode() {
        // The head always returns its full set - 300 queries - and the ones
        // that found nothing score far under any threshold.
        let (boxes, logits) = captured(295);
        assert_eq!(logits.len(), 300);
        let found = decode(&boxes, &logits, 0.25, WIDTH, HEIGHT).expect("the tensors agree");
        assert_eq!(found.len(), 5);
    }

    #[test]
    fn the_wide_whole_frame_box_goes_and_the_faces_beside_it_stay() {
        // Wide and most of the picture: the head's whole-frame answer. Beside
        // it a close-up, as large but taller than wide, and a face half behind
        // something, as wide but small - a row each.
        let boxes = [
            0.5, 0.5, 0.95, 0.4, //
            0.5, 0.5, 0.5, 0.6, //
            0.5, 0.5, 0.3, 0.1,
        ];
        let logits = [5.0, 5.0, 5.0];
        let found = decode(&boxes, &logits, 0.25, WIDTH, HEIGHT).expect("the tensors agree");
        assert_eq!(found.len(), 2, "only the wide whole-frame box goes");
        assert_eq!((found[0].w, found[0].h), (512, 858), "the close-up stays");
        assert_eq!(
            (found[1].w, found[1].h),
            (308, 144),
            "and so does the small"
        );
    }

    #[test]
    fn a_box_naming_no_pixel_of_the_picture_is_dropped() {
        let boxes = [1.6, 0.5, 0.2, 0.2];
        let logits = [5.0];
        assert!(decode(&boxes, &logits, 0.25, WIDTH, HEIGHT)
            .expect("the tensors agree")
            .is_empty());
    }

    #[test]
    fn tensors_that_disagree_on_how_many_queries_are_refused_by_name() {
        let boxes = [0.5, 0.5, 0.2, 0.2, 0.4, 0.4, 0.2, 0.2];
        let logits = [1.0];
        let error =
            decode(&boxes, &logits, 0.25, WIDTH, HEIGHT).expect_err("two boxes, one query");
        assert!(error.starts_with("detect_faces: "), "{error}");
    }

    #[test]
    fn a_row_spells_the_class_face_and_rounds_the_confidence() {
        assert_eq!(
            to_row(&Found {
                conf: 0.87941,
                x: 924,
                y: 1037,
                w: 100,
                h: 189,
            }),
            r#"{"class":"face","conf":0.8794,"x":924,"y":1037,"w":100,"h":189}"#
        );
    }

    #[test]
    fn the_two_returned_tensors_are_told_apart_by_shape_in_either_order() {
        assert_eq!(
            outputs(&[vec![1, 300, 4], vec![1, 300, 1]]).expect("both found"),
            (0, 1)
        );
        assert_eq!(
            outputs(&[vec![1, 300, 1], vec![1, 300, 4]]).expect("both found"),
            (1, 0)
        );
        let error = outputs(&[vec![1, 300, 4], vec![1, 300, 91]]).expect_err("a COCO head");
        assert!(error.starts_with("detect_faces: "), "{error}");
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
            assert!(error.starts_with("detect_faces "), "{error}");
        }
        assert!(
            parse_params(r#"{"radius":3}"#).is_err(),
            "and so is a param this module has none of"
        );
    }
}
