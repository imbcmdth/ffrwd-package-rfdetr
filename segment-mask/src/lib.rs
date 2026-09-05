//! Instance segmentation as a matte: every frame leaves as a grayscale mask
//! of the instances the model found - 255 where an instance owns the pixel,
//! 0 everywhere else - optionally narrowed to one class name.
//!
//! The graph is RF-DETR Seg Large's export, run through `wasi:nn`. A DETR
//! head returns a fixed set of queries with no duplicates among them, so
//! there is no NMS: what comes back is one box, one set of class logits and
//! one mask per query, the mask a quarter-resolution plane of logits over the
//! whole square. A pixel is inside an instance where its mask logit crosses
//! zero - sigmoid rises with its argument and a half is where it crosses zero,
//! so no sigmoid is computed for the mask at all. The module never opens a
//! file - the host binds the graph to a name with `-nn segment_mask=<path>`
//! and this module asks for that name and nothing else.
//!
//! The matte keeps the instance's own geometry and pixel format, so it feeds
//! straight into whatever reads a mask beside the picture: in yuv420p the
//! mask is the luma with neutral chroma, in rgba the same value in red, green
//! and blue, opaque.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:rfdetr-segment/segment-mask",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use rfdetr_common::{class_index, frame_box, le_f32s, sigmoid, to_input, PixFmt, Taps};
use serde::Deserialize;
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn segment_mask=<path>`.
const MODEL: &str = "segment_mask";

/// What the export calls its input tensor.
const INPUT_NAME: &str = "input";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// The square the graph is run at. The export is static at this size.
const SIDE: usize = 504;

/// Class logits per query: COCO's 91-slot category numbering.
const CLASSES: usize = 91;

/// A box's channels: centre, width and height, each a fraction of the picture.
const BOX_CHANNELS: usize = 4;

/// What a matte pixel inside an instance carries.
const KEEP: u8 = 255;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25}},"additionalProperties":false}"#;

fn default_conf() -> f64 {
    0.25
}

#[derive(Clone, Deserialize)]
// The schema says these two and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Params {
    /// One COCO class name to keep, or None for every class.
    #[serde(default)]
    class: Option<String>,
    #[serde(default = "default_conf")]
    conf: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            class: None,
            conf: default_conf(),
        }
    }
}

/// What the params settled once checked: the class as the graph's own label
/// index.
#[derive(Clone, Copy, Debug)]
struct Settled {
    class: Option<usize>,
    conf: f32,
}

/// One instance out of the graph's queries: its box as a fraction of the
/// picture, and which mask plane belongs to it.
#[derive(Clone, Copy, Debug)]
struct Instance {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    query: usize,
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    settled: Settled,
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

fn parse_params(params: &str) -> Result<Settled, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed)
            .map_err(|e| format!("segment_mask cannot read its params: {e}"))?
    };
    if !parsed.conf.is_finite() || !(0.0..=1.0).contains(&parsed.conf) {
        return Err(format!(
            "segment_mask needs conf between 0 and 1, got {}",
            parsed.conf
        ));
    }
    let class = match parsed.class.as_deref() {
        None | Some("") => None,
        Some(name) => Some(class_index(name).ok_or_else(|| {
            format!(
                "segment_mask does not know the class '{name}'; the model was \
                 trained on the 80 COCO classes, 'person' through 'toothbrush'"
            )
        })?),
    };
    Ok(Settled {
        class,
        conf: parsed.conf as f32,
    })
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
    format!("segment_mask: {what}: {code} ({})", error.data())
}

/// Which returned tensor is which, by shape: the masks are the rank-4 one,
/// and the boxes and class logits the rank-3 ones their last dimension tells
/// apart. Names are not read - one published export spells them in the wrong
/// order - so a graph that names them differently still resolves.
fn outputs(shapes: &[Vec<u32>]) -> Result<(usize, usize, usize), String> {
    let by_channels = |want: usize| {
        shapes.iter().position(
            |dimensions| matches!(dimensions.as_slice(), [_, _, channels] if *channels as usize == want),
        )
    };
    let masks = shapes.iter().position(|dimensions| dimensions.len() == 4);
    match (by_channels(BOX_CHANNELS), by_channels(CLASSES), masks) {
        (Some(boxes), Some(logits), Some(masks)) => Ok((boxes, logits, masks)),
        _ => Err(format!(
            "segment_mask: the graph returned {shapes:?}, and this module wants \
             RF-DETR's [1, queries, {BOX_CHANNELS}] boxes beside \
             [1, queries, {CLASSES}] class logits and \
             [1, queries, height, width] masks"
        )),
    }
}

/// The queries thresholded and narrowed to the class the params keep. Each
/// query carries a box as a fraction of the picture and a class taken from
/// whichever of its logits is largest.
fn decode(boxes: &[f32], logits: &[f32], settled: Settled) -> Result<Vec<Instance>, String> {
    if boxes.len() / BOX_CHANNELS != logits.len() / CLASSES {
        return Err(format!(
            "segment_mask: the graph returned {} boxes and {} query logits",
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
        if sigmoid(best) < settled.conf {
            continue;
        }
        if let Some(wanted) = settled.class {
            if class != wanted {
                continue;
            }
        }
        let at = query * BOX_CHANNELS;
        found.push(Instance {
            cx: boxes[at],
            cy: boxes[at + 1],
            w: boxes[at + 2],
            h: boxes[at + 3],
            query,
        });
    }
    Ok(found)
}

/// One mask row resized to a run of frame columns.
fn resize_row(source: &[f32], columns: &Taps, out: &mut [f32]) {
    for (((sample, low), high), fraction) in out
        .iter_mut()
        .zip(&columns.low)
        .zip(&columns.high)
        .zip(&columns.fraction)
    {
        let (a, b) = (source[*low], source[*high]);
        *sample = a + (b - a) * fraction;
    }
}

/// The combined matte: `KEEP` where any surviving instance owns the pixel.
///
/// A mask is `sigmoid(logit)` thresholded at a half, clipped to its own box.
/// Sigmoid rises with its argument and a half is where it crosses zero, so
/// the logit is compared against zero directly. The mask plane covers the
/// whole square, and the square is the whole frame stretched, so a mask
/// column is a frame column by the same ratio. The resize is separable: a
/// mask row is stretched to the box's columns once, and the two frame rows
/// reading it mix the same numbers, which leaves the per-pixel work one mix,
/// one compare and one store along a contiguous run.
fn matte(
    instances: &[Instance],
    masks: &[f32],
    mask: (usize, usize),
    frame: (usize, usize),
) -> Vec<u8> {
    let (mask_w, mask_h) = mask;
    let (width, height) = frame;
    let plane = mask_w * mask_h;

    let mut map = vec![0u8; width * height];
    let per_frame_x = mask_w as f32 / width as f32;
    let per_frame_y = mask_h as f32 / height as f32;

    for instance in instances {
        let (x0, y0, x1, y1) = frame_box(
            (instance.cx, instance.cy, instance.w, instance.h),
            width,
            height,
        );
        if x0 == x1 || y0 == y1 {
            continue;
        }
        let start = instance.query * plane;
        let Some(planes) = masks.get(start..start + plane) else {
            continue;
        };
        let run = x1 - x0;

        let columns = Taps::build(run, mask_w, |step| {
            ((x0 + step) as f32 + 0.5) * per_frame_x - 0.5
        });
        let rows = Taps::build(y1 - y0, mask_h, |step| {
            ((y0 + step) as f32 + 0.5) * per_frame_y - 0.5
        });

        let mut resized = [vec![0f32; run], vec![0f32; run]];
        let mut held: [Option<usize>; 2] = [None, None];

        for step in 0..rows.low.len() {
            for row in [rows.low[step], rows.high[step]] {
                let slot = row % 2;
                if held[slot] != Some(row) {
                    resize_row(
                        &planes[row * mask_w..(row + 1) * mask_w],
                        &columns,
                        &mut resized[slot],
                    );
                    held[slot] = Some(row);
                }
            }
            let (top, bottom) = (&resized[rows.low[step] % 2], &resized[rows.high[step] % 2]);
            let ty = rows.fraction[step];

            let at = (y0 + step) * width + x0;
            let target = &mut map[at..at + run];
            for ((slot, a), b) in target.iter_mut().zip(top).zip(bottom) {
                let logit = a + (b - a) * ty;
                if logit > 0.0 {
                    *slot = KEEP;
                }
            }
        }
    }
    map
}

/// A matte written as a frame of the instance's own format: the luma plane
/// with neutral chroma, or the same value in red, green and blue.
fn to_frame(map: &[u8], pix_fmt: PixFmt, width: usize, height: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    match pix_fmt {
        PixFmt::Yuv420p => {
            out[..width * height].copy_from_slice(map);
            // 128 in both chroma planes is no colour at all.
            out[width * height..].fill(128);
        }
        PixFmt::Rgba => {
            for (pixel, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(map) {
                *pixel = [*value, *value, *value, 255];
            }
        }
    }
    out
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

/// One frame in, its matte out.
fn run(opened: &Opened, frame: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let input = to_input(frame, opened.pix_fmt, opened.width, opened.height, SIDE);
    let returned = compute(opened, &input)?;
    let tensors: Vec<Tensor> = returned.into_iter().map(|(_, tensor)| tensor).collect();
    let shapes: Vec<Vec<u32>> = tensors.iter().map(Tensor::dimensions).collect();
    let (boxes, logits, masks) = outputs(&shapes)?;

    let mask_h = shapes[masks][2] as usize;
    let mask_w = shapes[masks][3] as usize;

    let found = decode(
        &le_f32s(&tensors[boxes].data()),
        &le_f32s(&tensors[logits].data()),
        opened.settled,
    )?;
    let map = matte(
        &found,
        &le_f32s(&tensors[masks].data()),
        (mask_w, mask_h),
        (opened.width, opened.height),
    );
    Ok(to_frame(
        &map,
        opened.pix_fmt,
        opened.width,
        opened.height,
        len,
    ))
}

struct SegmentMask;

impl Guest for SegmentMask {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "segment_mask".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
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
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("segment_mask reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = PixFmt::parse(&video.pix_fmt, "segment_mask")?;
        let settled = parse_params(&params)?;

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
                settled,
                input_name: Cell::new(INPUT_NAME),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let settled = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.settled = settled;
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
                let frame = window.fetch(i);
                // `process` has no way to say no, so a graph that failed
                // mid-stream stops the run rather than passing a frame off
                // as a matte.
                let map = run(opened, &frame, frame.len()).unwrap_or_else(|m| panic!("{m}"));
                out.push(OutFrame {
                    pts: window.pts(i),
                    frame: FramePayload::New(map),
                    rows: vec![],
                });
            }
        });
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(SegmentMask);

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask plane's side, a quarter of the square the graph is run at.
    const MASK: usize = SIDE / 4;

    /// A frame the size of the square, so a fraction of the picture and a
    /// fraction of the mask plane are the same fraction.
    const W: usize = SIDE;
    const H: usize = SIDE;

    fn settled(class: Option<usize>, conf: f32) -> Settled {
        Settled { class, conf }
    }

    /// The two rank-3 tensors for a list of `(cx, cy, w, h, class, logit)`
    /// queries, in the graph's own layout.
    fn queries(rows: &[(f32, f32, f32, f32, usize, f32)]) -> (Vec<f32>, Vec<f32>) {
        let mut boxes = Vec::new();
        let mut logits = Vec::new();
        for (cx, cy, w, h, class, logit) in rows {
            boxes.extend([*cx, *cy, *w, *h]);
            let mut scores = vec![-10.0f32; CLASSES];
            scores[*class] = *logit;
            logits.extend(scores);
        }
        (boxes, logits)
    }

    /// Mask planes for `count` queries, each filled with one logit.
    fn planes(logits: &[f32]) -> Vec<f32> {
        let mut all = Vec::new();
        for logit in logits {
            all.extend(std::iter::repeat_n(*logit, MASK * MASK));
        }
        all
    }

    #[test]
    fn the_threshold_and_the_class_both_narrow_the_queries() {
        // A person, a bus, and a query that found nothing.
        let (boxes, logits) = queries(&[
            (0.2, 0.2, 0.2, 0.2, 1, 2.2),
            (0.6, 0.6, 0.2, 0.2, 6, 0.4),
            (0.5, 0.5, 0.1, 0.1, 0, -10.0),
        ]);
        let count = |s: Settled| decode(&boxes, &logits, s).expect("the tensors agree").len();
        assert_eq!(count(settled(None, 0.25)), 2);
        assert_eq!(count(settled(Some(1), 0.25)), 1, "narrowed to person");
        assert_eq!(count(settled(Some(6), 0.25)), 1, "narrowed to bus");
        assert_eq!(
            count(settled(None, 0.7)),
            1,
            "the threshold drops the bus on its own"
        );
    }

    #[test]
    fn an_instance_keeps_the_query_its_mask_plane_belongs_to() {
        let (boxes, logits) =
            queries(&[(0.5, 0.5, 0.1, 0.1, 0, -10.0), (0.2, 0.2, 0.2, 0.2, 1, 2.2)]);
        let found = decode(&boxes, &logits, settled(None, 0.25)).expect("the tensors agree");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].query, 1, "the second query, not the first");
    }

    #[test]
    fn tensors_that_disagree_on_how_many_queries_are_refused_by_name() {
        let boxes = [0.5, 0.5, 0.2, 0.2, 0.4, 0.4, 0.2, 0.2];
        let logits = vec![-10.0f32; CLASSES];
        let error = decode(&boxes, &logits, settled(None, 0.25)).expect_err("two boxes, one query");
        assert!(error.starts_with("segment_mask: "), "{error}");
    }

    #[test]
    fn the_matte_paints_an_instance_over_its_own_box() {
        let (boxes, logits) = queries(&[(0.5, 0.5, 0.25, 0.25, 1, 2.2)]);
        let found = decode(&boxes, &logits, settled(None, 0.25)).expect("the tensors agree");
        let map = matte(&found, &planes(&[1.0]), (MASK, MASK), (W, H));
        let at = |x: usize, y: usize| map[y * W + x];
        assert_eq!(at(W / 2, H / 2), KEEP, "inside the box is the instance");
        assert_eq!(at(W / 8, H / 8), 0, "outside it is background");
        assert_eq!(at(7 * W / 8, 7 * H / 8), 0);
    }

    #[test]
    fn a_mask_that_never_crosses_zero_paints_nothing_inside_its_box() {
        let (boxes, logits) = queries(&[(0.5, 0.5, 0.25, 0.25, 1, 2.2)]);
        let found = decode(&boxes, &logits, settled(None, 0.25)).expect("the tensors agree");
        let map = matte(&found, &planes(&[-1.0]), (MASK, MASK), (W, H));
        assert!(map.iter().all(|value| *value == 0));
    }

    #[test]
    fn two_instances_paint_one_combined_matte() {
        let (boxes, logits) = queries(&[
            (0.25, 0.25, 0.2, 0.2, 1, 2.2),
            (0.75, 0.75, 0.2, 0.2, 1, 2.0),
        ]);
        let found = decode(&boxes, &logits, settled(None, 0.25)).expect("the tensors agree");
        let map = matte(&found, &planes(&[1.0, 1.0]), (MASK, MASK), (W, H));
        let at = |x: usize, y: usize| map[y * W + x];
        assert_eq!(at(W / 4, H / 4), KEEP);
        assert_eq!(at(3 * W / 4, 3 * H / 4), KEEP);
        assert_eq!(at(W / 2, H / 2), 0, "between them is background");
    }

    #[test]
    fn a_mask_is_clipped_to_its_own_box_however_far_it_spreads() {
        // The plane is positive everywhere; only the box is painted.
        let (boxes, logits) = queries(&[(0.25, 0.25, 0.1, 0.1, 1, 2.2)]);
        let found = decode(&boxes, &logits, settled(None, 0.25)).expect("the tensors agree");
        let map = matte(&found, &planes(&[5.0]), (MASK, MASK), (W, H));
        assert_eq!(map[(H / 4) * W + W / 4], KEEP, "inside the box");
        assert_eq!(map[(3 * H / 4) * W + 3 * W / 4], 0, "and nowhere else");
    }

    #[test]
    fn a_matte_writes_neutral_chroma_and_opaque_alpha() {
        let map = vec![KEEP; 4 * 4];
        let yuv = to_frame(&map, PixFmt::Yuv420p, 4, 4, 4 * 4 + 2 * 2 * 2);
        assert!(
            yuv[..16].iter().all(|v| *v == KEEP),
            "the luma is the matte"
        );
        assert!(
            yuv[16..].iter().all(|v| *v == 128),
            "and the chroma is neutral"
        );

        let rgba = to_frame(&map, PixFmt::Rgba, 4, 4, 4 * 4 * 4);
        let (pixels, _) = rgba.as_chunks::<4>();
        for pixel in pixels {
            assert_eq!(
                *pixel,
                [KEEP, KEEP, KEEP, 255],
                "equal in every channel, and opaque"
            );
        }
    }

    #[test]
    fn the_three_returned_tensors_are_told_apart_by_shape_in_any_order() {
        assert_eq!(
            outputs(&[vec![1, 200, 4], vec![1, 200, 91], vec![1, 200, 126, 126]])
                .expect("all three found"),
            (0, 1, 2)
        );
        // The order one published export hands them back in, its names in the
        // wrong places.
        assert_eq!(
            outputs(&[vec![1, 200, 126, 126], vec![1, 200, 91], vec![1, 200, 4]])
                .expect("all three found"),
            (2, 1, 0)
        );
        let error = outputs(&[vec![1, 116, 8400], vec![1, 32, 160, 160]]).expect_err("no boxes");
        assert!(error.starts_with("segment_mask: "), "{error}");
    }

    #[test]
    fn params_default_to_every_class_at_the_published_threshold() {
        let parsed = parse_params("").expect("empty is the defaults");
        assert_eq!(parsed.class, None);
        assert_eq!(parsed.conf, 0.25);
        let named = parse_params(r#"{"class":"person"}"#).expect("a known class");
        assert_eq!(named.class, Some(1));
    }

    #[test]
    fn a_class_the_model_was_not_trained_on_is_refused_by_name() {
        let error = parse_params(r#"{"class":"warp core"}"#).expect_err("not a COCO class");
        assert!(error.contains("warp core"), "{error}");
        assert!(
            parse_params(r#"{"conf":1.5}"#).is_err(),
            "and so is a threshold outside 0..1"
        );
    }
}
