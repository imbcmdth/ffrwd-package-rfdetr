//! Boxes to matte: the rows paired with each frame are rasterized into a
//! grayscale mask, 255 inside each box and 0 everywhere else. `grow` pads
//! every box outward in pixels; `feather` softens the edge over that many
//! pixels, falling linearly from the box's edge to nothing.
//!
//! The picture is read for its times and size alone, so the host hands it in
//! whatever format its source has and carries no pixels for it; the matte
//! leaves one byte a pixel. Any row carrying `x`, `y`, `w` and `h` is a box, so the rows
//! of every detector, and of a tracker that adds fields of its own, read the
//! same.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"grow":{"type":"number","minimum":0,"maximum":4096,"default":0},"feather":{"type":"number","minimum":0,"maximum":4096,"default":0}},"additionalProperties":false}"#;

#[derive(Clone, Copy, Deserialize)]
struct Params {
    grow: f64,
    feather: f64,
}

/// One box to rasterize: the four fields this module reads, and any others
/// the row carries pass by.
#[derive(Default, Serialize, Deserialize)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// The alpha of one axis at a pixel centre `p`: full inside `[edge0, edge1]`,
/// falling linearly to nothing `feather` pixels outside either edge.
fn axis_alpha(p: f64, edge0: f64, edge1: f64, feather: f64) -> f64 {
    let outside = (edge0 - p).max(p - edge1).max(0.0);
    if outside <= 0.0 {
        return 1.0;
    }
    if feather <= 0.0 {
        return 0.0;
    }
    (1.0 - outside / feather).max(0.0)
}

/// One box painted into the matte, combined by max so overlapping boxes keep
/// each other's coverage. The horizontal ramp is computed once per box; each
/// row then scales it by its own vertical alpha, one multiply and one store
/// along a contiguous run.
fn paint(map: &mut [u8], width: usize, height: usize, rect: &Rect, params: Params) {
    if rect.w <= 0.0 || rect.h <= 0.0 {
        return;
    }
    let (grow, feather) = (params.grow, params.feather);
    let x0 = rect.x - grow;
    let x1 = rect.x + rect.w + grow;
    let y0 = rect.y - grow;
    let y1 = rect.y + rect.h + grow;

    let first_x = (x0 - feather).floor().max(0.0) as usize;
    let last_x = ((x1 + feather).ceil().min(width as f64) as usize).max(first_x);
    let first_y = (y0 - feather).floor().max(0.0) as usize;
    let last_y = ((y1 + feather).ceil().min(height as f64) as usize).max(first_y);
    if first_x == last_x || first_y == last_y {
        return;
    }

    // The horizontal ramp, in eight bits so the row loop stays integer.
    let ramp: Vec<u16> = (first_x..last_x)
        .map(|x| (axis_alpha(x as f64 + 0.5, x0, x1, feather) * 255.0).round() as u16)
        .collect();

    for y in first_y..last_y {
        let ay = (axis_alpha(y as f64 + 0.5, y0, y1, feather) * 255.0).round() as u16;
        if ay == 0 {
            continue;
        }
        let row = &mut map[y * width + first_x..y * width + last_x];
        for (slot, ax) in row.iter_mut().zip(&ramp) {
            let value = ((ax * ay + 127) / 255) as u8;
            if value > *slot {
                *slot = value;
            }
        }
    }
}

/// The matte the boxes rasterize to, one byte a pixel.
fn rasterize(rects: &[Rect], width: usize, height: usize, params: Params) -> Vec<u8> {
    let mut map = vec![0u8; width * height];
    for rect in rects {
        paint(&mut map, width, height, rect, params);
    }
    map
}

struct BoxesMask {
    v: u32,
    boxes: u32,
    width: usize,
    height: usize,
    params: Params,
}

impl Node for BoxesMask {
    const NAME: &'static str = "boxes_mask";
    const VERSION: &'static str = "0.2.1";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().timing())
            .input(Input::rows("boxes").schema::<Rect>())
            .output(Output::like("v").pixel_format("gray"))
            .pure()
            .one_to_one())
    }

    fn init(params: Params, init: &Init) -> Result<BoxesMask> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("boxes_mask reads frames, and `v` is not video")?;
        Ok(BoxesMask {
            v: v.id,
            boxes: init.stream("boxes")?.id,
            width: video.width as usize,
            height: video.height as usize,
            params,
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        self.params = params;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let rects: Vec<Rect> = tick.rows(self.boxes)?;
        let map = rasterize(&rects, self.width, self.height, self.params);
        Ok(out.frame("v", frame.pts, frame.duration, map)?)
    }
}

ffrwd_node::export!(BoxesMask);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{read_params, BoundStream, Payload, Rational, Wants};

    /// What a matte pixel fully inside a box carries.
    const KEEP: u8 = 255;

    const W: usize = 64;
    const H: usize = 48;

    fn params(grow: f64, feather: f64) -> Params {
        Params { grow, feather }
    }

    fn rects(rows: &[&str]) -> Vec<Rect> {
        rows.iter()
            .map(|row| ffrwd_node::parse(row).unwrap())
            .collect()
    }

    fn harness(pix_fmt: &str) -> Harness<BoxesMask> {
        let tb = Rational::new(1, 25);
        let bound = vec![
            BoundStream::video("v", 0, W as u32, H as u32, pix_fmt, tb),
            BoundStream::rows("boxes", 1, tb),
        ];
        Harness::new(r#"{"grow":2}"#, bound).unwrap()
    }

    #[test]
    fn a_box_paints_hard_edges_when_nothing_feathers() {
        let map = rasterize(
            &rects(&[r#"{"class":"person","conf":0.9,"x":10,"y":10,"w":20,"h":10}"#]),
            W,
            H,
            params(0.0, 0.0),
        );
        assert_eq!(map[15 * W + 15], KEEP, "inside the box");
        assert_eq!(map[15 * W + 10], KEEP, "the left edge is inside");
        assert_eq!(map[15 * W + 29], KEEP, "and so is the last column");
        assert_eq!(map[15 * W + 30], 0, "one past it is not");
        assert_eq!(map[9 * W + 15], 0, "above the box is background");
    }

    #[test]
    fn grow_pads_the_box_outward() {
        let map = rasterize(
            &rects(&[r#"{"x":10,"y":10,"w":20,"h":10}"#]),
            W,
            H,
            params(4.0, 0.0),
        );
        assert_eq!(map[15 * W + 7], KEEP, "four pixels left of the box");
        assert_eq!(map[7 * W + 15], KEEP, "and four above it");
        assert_eq!(map[15 * W + 5], 0, "five is past the growth");
    }

    #[test]
    fn feather_ramps_from_full_at_the_edge_to_nothing() {
        let map = rasterize(
            &rects(&[r#"{"x":16,"y":16,"w":16,"h":16}"#]),
            W,
            H,
            params(0.0, 8.0),
        );
        assert_eq!(map[20 * W + 20], KEEP, "inside stays full");
        let mid = map[20 * W + 12]; // four pixels outside a left edge at 16
        assert!(
            (100..=160).contains(&mid),
            "half way out is about half, got {mid}"
        );
        assert_eq!(map[20 * W + 4], 0, "past the feather is background");
        let corner = map[12 * W + 12]; // four outside on both axes
        assert!(corner < mid, "a corner takes both ramps: {corner} < {mid}");
    }

    #[test]
    fn overlapping_boxes_keep_the_stronger_coverage() {
        let map = rasterize(
            &rects(&[
                r#"{"x":10,"y":10,"w":10,"h":10}"#,
                r#"{"x":15,"y":10,"w":10,"h":10}"#,
            ]),
            W,
            H,
            params(0.0, 0.0),
        );
        assert_eq!(map[15 * W + 17], KEEP, "the overlap is full, not doubled");
        assert_eq!(map[15 * W + 12], KEEP);
        assert_eq!(map[15 * W + 23], KEEP);
    }

    #[test]
    fn a_box_running_off_the_frame_is_clipped_not_refused() {
        let map = rasterize(
            &rects(&[r#"{"x":-10,"y":-10,"w":30,"h":30}"#]),
            W,
            H,
            params(0.0, 4.0),
        );
        assert_eq!(map[0], KEEP, "the corner the box covers is painted");
        assert_eq!(map[25 * W + 25], 0, "past its clipped extent is not");
    }

    #[test]
    fn no_rows_at_all_is_an_entirely_black_matte() {
        let map = rasterize(&[], W, H, params(8.0, 8.0));
        assert!(map.iter().all(|v| *v == 0));
    }

    #[test]
    fn a_degenerate_box_paints_nothing() {
        let map = rasterize(
            &rects(&[r#"{"x":10,"y":10,"w":0,"h":10}"#]),
            W,
            H,
            params(0.0, 0.0),
        );
        assert!(map.iter().all(|v| *v == 0));
    }

    #[test]
    fn params_outside_the_schema_are_refused_by_name() {
        for bad in [r#"{"grow":-1}"#, r#"{"feather":5000}"#, r#"{"radius":3}"#] {
            assert!(read_params::<Params>(PARAMS_SCHEMA, bad).is_err(), "{bad}");
        }
        let (parsed, _) = read_params::<Params>(PARAMS_SCHEMA, "").expect("empty is the defaults");
        assert_eq!((parsed.grow, parsed.feather), (0.0, 0.0));
    }

    #[test]
    fn the_matte_is_the_picture_in_gray_and_reads_any_row_with_a_box() {
        let shape = harness("rgba").shape().clone();
        assert_eq!(shape.clock_input(), Some("v"));
        let v = shape.find_input("v").expect("the picture");
        assert_eq!(v.accepts.wants, Wants::Timing, "its times and size, never its pixels");
        assert!(v.accepts.pixel_formats.is_empty(), "in whatever format it comes");
        let boxes = shape.find_input("boxes").expect("a rows input");
        let schema: serde_json::Value =
            serde_json::from_str(boxes.schema.as_deref().expect("a schema")).unwrap();
        assert_eq!(
            schema["required"],
            serde_json::json!(["h", "w", "x", "y"]),
            "the four fields it reads, and no others asked for"
        );
        assert_eq!(schema["properties"]["x"]["type"], "number");
        assert!(
            schema.get("additionalProperties").is_none(),
            "the rest pass"
        );
        let like = shape.outputs[0].like.as_ref().expect("follows its input");
        assert_eq!(
            (like.port.as_deref(), like.pixel_format.as_deref()),
            (Some("v"), Some("gray"))
        );
        assert!(shape.pure && shape.one_to_one);
    }

    #[test]
    fn a_frame_leaves_as_its_matte_without_its_picture_being_read() {
        for pix_fmt in ["rgba", "yuv420p", "yuv444p10le"] {
            let mut node = harness(pix_fmt);
            let row = r#"{"class":"face","conf":0.9,"x":10,"y":10,"w":4,"h":4,"age":8.4}"#;
            let tick = node
                .tick(3)
                .frame_with(0, 3, Some(1), &[], Vec::new())
                .message(1, 3, row.as_bytes());
            let emitted = node.process(&tick).unwrap();
            let [Payload::Frame { pts: 3, data, .. }] = emitted.on("v")[..] else {
                panic!("no matte: {emitted:?}")
            };
            assert_eq!(data.len(), W * H, "one byte a pixel");
            assert_eq!(data[12 * W + 12], KEEP, "inside the box");
            assert_eq!(data[8 * W + 8], KEEP, "and grown by two");
            assert_eq!(data[7 * W + 7], 0, "and no further");
        }
    }
}
