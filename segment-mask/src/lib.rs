//! Instance segmentation as a matte: every frame leaves as a grayscale mask
//! of the instances the model found, 255 where an instance owns the pixel and
//! 0 everywhere else, optionally narrowed to one class name.
//!
//! The graph is RF-DETR Seg Large's export, run through `wasi:nn` and bound
//! to the name `segment_mask`. A DETR head returns a fixed set of queries
//! with no duplicates among them, so there is no NMS: what comes back is one
//! box, one set of class logits and one mask per query, the mask a
//! quarter-resolution plane of logits over the whole square. A pixel is
//! inside an instance where its mask logit crosses zero: sigmoid rises with
//! its argument and a half is where it crosses zero, so no sigmoid is
//! computed for the mask at all.
//!
//! The matte keeps the instance's own geometry, one byte a pixel, so it
//! feeds straight into whatever reads a mask beside the picture.

use ffrwd_frame::{Filter, Rect, Rgba, IMAGENET};
use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use rfdetr_common::detector::INPUT_NAME;
use rfdetr_common::nn::Model;
use rfdetr_common::{class_index, frame_box, sigmoid, Taps};
use serde::Deserialize;

/// The square the graph is run at. The export is static at this size.
const SIDE: usize = 504;

/// Class logits per query: COCO's 91-slot category numbering.
const CLASSES: usize = 91;

/// A box's channels: centre, width and height, each a fraction of the picture.
const BOX_CHANNELS: usize = 4;

/// What a matte pixel inside an instance carries.
const KEEP: u8 = 255;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25}},"additionalProperties":false}"#;

#[derive(Deserialize)]
struct Params {
    /// One COCO class name to keep, or none for every class.
    class: Option<String>,
    conf: f64,
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

fn settle(params: &Params) -> Result<Settled, String> {
    let class = match params.class.as_deref() {
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
        conf: params.conf as f32,
    })
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

struct SegmentMask {
    v: u32,
    width: usize,
    height: usize,
    settled: Settled,
    model: Model,
}

impl SegmentMask {
    /// One frame in, its matte out.
    fn run(&mut self, pixels: &[u8]) -> Result<Vec<u8>, String> {
        let (width, height) = (self.width, self.height);
        let input = ffrwd_frame::tensor(
            &Rgba::new(pixels, width, height)?,
            Rect::whole(width, height),
            SIDE,
            SIDE,
            Filter::Bilinear,
            IMAGENET,
        );
        let returned = self.model.run(&[1, 3, SIDE as u32, SIDE as u32], &input)?;
        let shapes: Vec<Vec<u32>> = returned.iter().map(|r| r.dimensions.clone()).collect();
        let (boxes, logits, masks) = outputs(&shapes)?;
        let (mask_h, mask_w) = (shapes[masks][2] as usize, shapes[masks][3] as usize);
        let found = decode(
            &returned[boxes].values,
            &returned[logits].values,
            self.settled,
        )?;
        Ok(matte(
            &found,
            &returned[masks].values,
            (mask_w, mask_h),
            (width, height),
        ))
    }
}

impl Node for SegmentMask {
    const NAME: &'static str = "segment_mask";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .output(Output::like("v").pixel_format("gray"))
            .pure()
            .one_to_one())
    }

    fn init(params: Params, init: &Init) -> Result<SegmentMask> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("segment_mask reads frames, and `v` is not video")?;
        Ok(SegmentMask {
            v: v.id,
            width: video.width as usize,
            height: video.height as usize,
            settled: settle(&params)?,
            model: Model::load(Self::NAME, Self::NAME, INPUT_NAME)?,
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        self.settled = settle(&params)?;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.v) {
            let map = self.run(&tick.fetch(self.v, frame.index))?;
            out.frame("v", frame.pts, frame.duration, map)?;
        }
        Ok(())
    }
}

ffrwd_node::export!(SegmentMask);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::{read_params, Runner};

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
    fn the_matte_is_the_picture_in_gray() {
        let shape = Runner::<SegmentMask>::shape(r#"{"class":"person"}"#, &["v".to_owned()])
            .expect("a shape");
        assert_eq!(shape.clock_input(), Some("v"));
        assert_eq!(shape.outputs.len(), 1);
        let like = shape.outputs[0].like.as_ref().expect("follows its input");
        assert_eq!(
            (like.port.as_deref(), like.pixel_format.as_deref()),
            (Some("v"), Some("gray"))
        );
        assert!(shape.pure && shape.one_to_one);
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

    fn parse_params(params: &str) -> Result<Settled, String> {
        let (params, _) = read_params::<Params>(PARAMS_SCHEMA, params)?;
        settle(&params)
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
