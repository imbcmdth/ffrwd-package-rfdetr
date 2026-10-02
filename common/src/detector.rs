//! The node every detector in the package is: a picture in, a row per box
//! out, and nothing else. The heads differ in their graph, their classes and
//! which boxes they keep; the rest is here once.
//!
//! A DETR head returns a fixed set of queries with no duplicates among them,
//! so there is no NMS: decoding is a sigmoid over each query's class logits, a
//! threshold, and a coordinate map.

use std::marker::PhantomData;

use ffrwd_frame::{Filter, Rect, Rgba, IMAGENET};
use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

use crate::nn::Model;
use crate::{frame_box, sigmoid};

pub const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25}},"additionalProperties":false}"#;

/// Every detector's rows: the class as text, the confidence, and the box in
/// the frame's own pixels, whole numbers each.
pub const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number"},"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}},"required":["class","conf","x","y","w","h"],"additionalProperties":false}"#;

/// What the exports call their input tensor.
pub const INPUT_NAME: &str = "input";

/// A box's channels: centre, width and height, each a fraction of the picture.
const BOX_CHANNELS: usize = 4;

#[derive(Deserialize)]
pub struct Params {
    pub conf: f64,
}

/// One row per box, the box in the frame's own pixels.
#[derive(Serialize)]
pub struct Row {
    pub class: String,
    pub conf: f64,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// One query that cleared the threshold, already on the frame's own axes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Found {
    pub class: usize,
    pub conf: f32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// What sets one detector apart from the others.
pub trait Head: 'static {
    /// The module's name, which is also the name the host binds its graph to.
    const NAME: &'static str;
    const VERSION: &'static str;
    /// The square the graph is run at. The export is static at this size.
    const SIDE: usize;
    /// Class logits per query.
    const CLASSES: usize;

    /// Which class a query's logits name, and the logit that names it.
    fn classify(logits: &[f32]) -> (usize, f32);

    /// The class as a row spells it.
    fn class_name(class: usize) -> String;

    /// Whether a box `w` by `h` on a `width` by `height` frame is one this
    /// head means. Every box it returns, unless the head says otherwise.
    fn keeps(w: usize, h: usize, width: usize, height: usize) -> bool {
        let _ = (w, h, width, height);
        true
    }
}

/// Which returned tensor is the boxes and which the class logits, by shape:
/// both are rank 3 over the same queries, and the last dimension tells them
/// apart. Names are not read, so an export that spells them differently, or
/// hands them back in the other order, still resolves.
pub fn outputs<H: Head>(shapes: &[Vec<u32>]) -> Result<(usize, usize), String> {
    let by_channels = |want: usize| {
        shapes.iter().position(
            |dimensions| matches!(dimensions.as_slice(), [_, _, channels] if *channels as usize == want),
        )
    };
    match (by_channels(BOX_CHANNELS), by_channels(H::CLASSES)) {
        (Some(boxes), Some(logits)) => Ok((boxes, logits)),
        _ => Err(format!(
            "{}: the graph returned {shapes:?}, and this module wants \
             RF-DETR's [1, queries, {BOX_CHANNELS}] boxes beside \
             [1, queries, {}] class logits",
            H::NAME,
            H::CLASSES
        )),
    }
}

/// The queries thresholded and brought onto the frame. Each query's box is
/// `cx, cy, w, h` as a fraction of the picture, and its confidence the logit
/// its head reads through the logistic curve.
pub fn decode<H: Head>(
    boxes: &[f32],
    logits: &[f32],
    conf: f32,
    width: usize,
    height: usize,
) -> Result<Vec<Found>, String> {
    if boxes.len() / BOX_CHANNELS != logits.len() / H::CLASSES {
        return Err(format!(
            "{}: the graph returned {} boxes and {} query logits",
            H::NAME,
            boxes.len() / BOX_CHANNELS,
            logits.len() / H::CLASSES
        ));
    }
    let mut found = Vec::new();
    for (query, scores) in logits.chunks_exact(H::CLASSES).enumerate() {
        let (class, logit) = H::classify(scores);
        let score = sigmoid(logit);
        if score < conf {
            continue;
        }
        let at = query * BOX_CHANNELS;
        let (x0, y0, x1, y1) = frame_box(
            (boxes[at], boxes[at + 1], boxes[at + 2], boxes[at + 3]),
            width,
            height,
        );
        if x0 == x1 || y0 == y1 || !H::keeps(x1 - x0, y1 - y0, width, height) {
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

/// One box's row.
pub fn row<H: Head>(found: &Found) -> Row {
    Row {
        class: H::class_name(found.class),
        // To four places: the graph's own precision is nowhere near the
        // sixteen digits an f32 widened to an f64 prints.
        conf: (f64::from(found.conf) * 10_000.0).round() / 10_000.0,
        x: found.x,
        y: found.y,
        w: found.w,
        h: found.h,
    }
}

pub struct Detector<H> {
    v: u32,
    width: usize,
    height: usize,
    conf: f32,
    model: Model,
    head: PhantomData<H>,
}

impl<H: Head> Detector<H> {
    fn detect(&mut self, pixels: &[u8]) -> Result<Vec<Found>, String> {
        let (width, height) = (self.width, self.height);
        let input = ffrwd_frame::tensor(
            &Rgba::new(pixels, width, height)?,
            Rect::whole(width, height),
            H::SIDE,
            H::SIDE,
            Filter::Bilinear,
            IMAGENET,
        );
        let side = H::SIDE as u32;
        let returned = self.model.run(&[1, 3, side, side], &input)?;
        let shapes: Vec<Vec<u32>> = returned.iter().map(|r| r.dimensions.clone()).collect();
        let (boxes, logits) = outputs::<H>(&shapes)?;
        decode::<H>(
            &returned[boxes].values,
            &returned[logits].values,
            self.conf,
            width,
            height,
        )
    }
}

impl<H: Head> Node for Detector<H> {
    const NAME: &'static str = H::NAME;
    const VERSION: &'static str = H::VERSION;
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .output(Output::rows("boxes").schema_json(ROWS_SCHEMA))
            .pure())
    }

    fn init(params: Params, init: &Init) -> Result<Detector<H>> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or_else(|| format!("{} reads frames, and `v` is not video", H::NAME))?;
        Ok(Detector {
            v: v.id,
            width: video.width as usize,
            height: video.height as usize,
            conf: params.conf as f32,
            model: Model::load(H::NAME, H::NAME, INPUT_NAME)?,
            head: PhantomData,
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        self.conf = params.conf as f32;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.v) {
            let found = self.detect(&tick.fetch(self.v, frame.index))?;
            for one in &found {
                out.row("boxes", frame.pts, &row::<H>(one))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::{read_params, Kind, Runner};

    struct Two;

    impl Head for Two {
        const NAME: &'static str = "two";
        const VERSION: &'static str = "0.0.0";
        const SIDE: usize = 8;
        const CLASSES: usize = 2;

        fn classify(logits: &[f32]) -> (usize, f32) {
            if logits[1] > logits[0] {
                (1, logits[1])
            } else {
                (0, logits[0])
            }
        }

        fn class_name(class: usize) -> String {
            ["cat", "dog"][class].to_string()
        }
    }

    #[test]
    fn a_detector_writes_rows_alone_and_the_picture_stays_at_the_source() {
        let shape = Runner::<Detector<Two>>::shape("", &["v".to_owned()]).expect("a shape");
        assert_eq!(shape.clock_input(), Some("v"));
        assert_eq!(shape.inputs[0].accepts.pixel_formats, ["rgba"]);
        assert_eq!(shape.outputs.len(), 1, "no picture leaves");
        assert_eq!(shape.outputs[0].name, "boxes");
        assert_eq!(shape.outputs[0].kind, Kind::Data);
        assert_eq!(shape.outputs[0].schema.as_deref(), Some(ROWS_SCHEMA));
        assert!(shape.pure);
    }

    #[test]
    fn a_row_spells_the_class_as_text_and_rounds_the_confidence() {
        let found = Found {
            class: 1,
            conf: 0.95132,
            x: 79,
            y: 57,
            w: 221,
            h: 517,
        };
        assert_eq!(
            serde_json::to_string(&row::<Two>(&found)).unwrap(),
            r#"{"class":"dog","conf":0.9513,"x":79,"y":57,"w":221,"h":517}"#
        );
    }

    #[test]
    fn the_head_picks_the_class_and_the_threshold_reads_its_logit() {
        let boxes = [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1];
        let logits = [-3.0, 2.0, 1.0, -9.0];
        let found = decode::<Two>(&boxes, &logits, 0.5, 100, 100).expect("the tensors agree");
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].class, found[1].class), (1, 0));
        assert_eq!(
            (found[0].x, found[0].y, found[0].w, found[0].h),
            (25, 25, 50, 50)
        );
        let sure = decode::<Two>(&boxes, &logits, 0.8, 100, 100).expect("the tensors agree");
        assert_eq!(sure.len(), 1, "sigmoid(1) is 0.73, under 0.8");
    }

    #[test]
    fn tensors_that_disagree_on_how_many_queries_are_refused_by_name() {
        let boxes = [0.5, 0.5, 0.2, 0.2, 0.4, 0.4, 0.2, 0.2];
        let error =
            decode::<Two>(&boxes, &[0.0, 0.0], 0.25, 720, 576).expect_err("two boxes, one query");
        assert!(error.starts_with("two: "), "{error}");
    }

    #[test]
    fn the_two_returned_tensors_are_told_apart_by_shape_in_either_order() {
        assert_eq!(
            outputs::<Two>(&[vec![1, 300, 4], vec![1, 300, 2]]).unwrap(),
            (0, 1)
        );
        assert_eq!(
            outputs::<Two>(&[vec![1, 300, 2], vec![1, 300, 4]]).unwrap(),
            (1, 0)
        );
        let error = outputs::<Two>(&[vec![1, 84, 8400]]).expect_err("a dense grid");
        assert!(error.starts_with("two: "), "{error}");
    }

    #[test]
    fn params_default_to_the_threshold_the_schema_publishes() {
        let (empty, _) = read_params::<Params>(PARAMS_SCHEMA, "").expect("empty is the defaults");
        assert_eq!(empty.conf, 0.25);
        let (braces, _) = read_params::<Params>(PARAMS_SCHEMA, "{}").expect("and so is {}");
        assert_eq!(braces.conf, 0.25);
    }

    #[test]
    fn params_outside_zero_to_one_are_refused() {
        for bad in [r#"{"conf":1.5}"#, r#"{"conf":-0.1}"#, r#"{"radius":3}"#] {
            assert!(read_params::<Params>(PARAMS_SCHEMA, bad).is_err(), "{bad}");
        }
    }
}
