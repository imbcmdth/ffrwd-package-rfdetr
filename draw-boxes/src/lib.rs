//! Boxes on the picture: the rows paired with each frame are drawn as green
//! rectangle outlines, `thickness` pixels wide, growing inward from each
//! box's edge so a drawn box never spills past what the detector reported. A
//! frame with no box on it leaves as it arrived, never fetched.
//!
//! The coordinates are whole pixels, as every detector writes them: a row
//! whose box is in fractions is refused when the query is compiled.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"thickness":{"type":"integer","minimum":1,"maximum":64,"default":2}},"additionalProperties":false}"#;

/// The outline's colour: green, as each format spells it. The yuv triple is
/// pure RGB green through the full-range BT.601 forward transform.
const GREEN_RGB: [u8; 3] = [0, 255, 0];
const GREEN_Y: u8 = 150;
const GREEN_U: u8 = 43;
const GREEN_V: u8 = 21;

#[derive(Clone, Copy, Deserialize)]
struct Params {
    thickness: u32,
}

/// One box to draw: the four fields this module reads, and any others the
/// row carries pass by.
#[derive(Default, Serialize, Deserialize)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

/// A rectangle cut down to the frame: `x0..x1` by `y0..y1`, exclusive.
#[derive(Clone, Copy)]
struct Region {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// A rectangle inside the frame, or None when it falls outside it entirely.
fn clamp(rect: &Rect, width: usize, height: usize) -> Option<Region> {
    let w = width as i64;
    let h = height as i64;
    let x0 = rect.x.clamp(0, w);
    let y0 = rect.y.clamp(0, h);
    let x1 = rect.x.saturating_add(rect.w).clamp(0, w);
    let y1 = rect.y.saturating_add(rect.h).clamp(0, h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Region {
        x0: x0 as usize,
        y0: y0 as usize,
        x1: x1 as usize,
        y1: y1 as usize,
    })
}

/// The pixel format the host chose at `init`, fixed for the instance's life.
#[derive(Clone, Copy, PartialEq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

/// What `init` settled.
#[derive(Clone, Copy)]
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    params: Params,
}

/// One filled band of the picture painted green, in whichever format the
/// instance was opened for. The band is already inside the frame.
fn fill(frame: &mut [u8], opened: &Opened, band: Region) {
    let (width, height) = (opened.width, opened.height);
    match opened.pix_fmt {
        PixFmt::Rgba => {
            for y in band.y0..band.y1 {
                let row = &mut frame[(y * width + band.x0) * 4..(y * width + band.x1) * 4];
                for pixel in row.as_chunks_mut::<4>().0 {
                    pixel[0] = GREEN_RGB[0];
                    pixel[1] = GREEN_RGB[1];
                    pixel[2] = GREEN_RGB[2];
                }
            }
        }
        PixFmt::Yuv420p => {
            for y in band.y0..band.y1 {
                frame[y * width + band.x0..y * width + band.x1].fill(GREEN_Y);
            }
            // Chroma is half resolution both ways; the band rounds outward so
            // a drawn edge never keeps the picture's own colour.
            let (cw, ch) = (width / 2, height / 2);
            let pixels = width * height;
            let cx0 = band.x0 / 2;
            let cx1 = (band.x1.div_ceil(2)).min(cw);
            let cy0 = band.y0 / 2;
            let cy1 = (band.y1.div_ceil(2)).min(ch);
            let (u, v) = frame[pixels..].split_at_mut(cw * ch);
            for cy in cy0..cy1 {
                u[cy * cw + cx0..cy * cw + cx1].fill(GREEN_U);
                v[cy * cw + cx0..cy * cw + cx1].fill(GREEN_V);
            }
        }
    }
}

/// One box's outline: four bands growing inward from its edges, so a
/// thickness wider than the box fills it and never spills.
fn outline(frame: &mut [u8], opened: &Opened, region: Region, thickness: usize) {
    let top = (region.y0 + thickness).min(region.y1);
    let bottom = region.y1.saturating_sub(thickness).max(top);
    for band in [
        // Top and bottom, full width.
        Region { y1: top, ..region },
        Region {
            y0: bottom,
            ..region
        },
        // Left and right, between them.
        Region {
            y0: top,
            y1: bottom,
            x1: (region.x0 + thickness).min(region.x1),
            ..region
        },
        Region {
            y0: top,
            y1: bottom,
            x0: region.x1.saturating_sub(thickness).max(region.x0),
            ..region
        },
    ] {
        if band.x0 < band.x1 && band.y0 < band.y1 {
            fill(frame, opened, band);
        }
    }
}

/// The regions the boxes name, each cut down to the frame. A box that falls
/// outside the picture is skipped.
fn boxes(rects: &[Rect], opened: &Opened) -> Vec<Region> {
    rects
        .iter()
        .filter_map(|rect| clamp(rect, opened.width, opened.height))
        .collect()
}

/// Every region outlined on one frame.
fn draw(frame: &mut [u8], opened: &Opened, regions: &[Region]) {
    for region in regions {
        outline(frame, opened, *region, opened.params.thickness as usize);
    }
}

struct DrawBoxes {
    v: u32,
    boxes: u32,
    opened: Opened,
}

impl Node for DrawBoxes {
    const NAME: &'static str = "draw_boxes";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(
                Input::video("v")
                    .clock()
                    .pixel_formats(&["rgba", "yuv420p"]),
            )
            .input(Input::rows("boxes").schema::<Rect>())
            .output(Output::like("v"))
            .pure()
            .one_to_one())
    }

    fn init(params: Params, init: &Init) -> Result<DrawBoxes> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("draw_boxes reads frames, and `v` is not video")?;
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("draw_boxes does not accept pixel format {other}").into()),
        };
        Ok(DrawBoxes {
            v: v.id,
            boxes: init.stream("boxes")?.id,
            opened: Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
                params,
            },
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        self.opened.params = params;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let rects: Vec<Rect> = tick.rows(self.boxes)?;
        let regions = boxes(&rects, &self.opened);
        if regions.is_empty() {
            return Ok(out.pass("v", self.v, &frame)?);
        }
        let mut picture = tick.fetch(self.v, frame.index);
        draw(&mut picture, &self.opened, &regions);
        Ok(out.frame("v", frame.pts, frame.duration, picture)?)
    }
}

ffrwd_node::export!(DrawBoxes);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{read_params, BoundStream, Payload, Rational};

    const W: usize = 32;
    const H: usize = 24;

    fn opened(pix_fmt: PixFmt, thickness: u32) -> Opened {
        Opened {
            width: W,
            height: H,
            pix_fmt,
            params: Params { thickness },
        }
    }

    fn grey_yuv() -> Vec<u8> {
        vec![100u8; W * H + 2 * (W / 2) * (H / 2)]
    }

    /// The regions rows name, as the node reads them.
    fn rows(items: &[&str], opened: &Opened) -> Vec<Region> {
        let rects: Vec<Rect> = items
            .iter()
            .map(|row| ffrwd_node::parse(row).unwrap())
            .collect();
        boxes(&rects, opened)
    }

    fn harness() -> Harness<DrawBoxes> {
        let tb = Rational::new(1, 25);
        let bound = vec![
            BoundStream::video("v", 0, W as u32, H as u32, "rgba", tb),
            BoundStream::rows("boxes", 1, tb),
        ];
        Harness::new("", bound).unwrap()
    }

    #[test]
    fn an_outline_paints_the_edges_and_leaves_the_interior() {
        let mut frame = grey_yuv();
        let settled = opened(PixFmt::Yuv420p, 2);
        let named = rows(
            &[r#"{"class":"person","conf":0.9,"x":4,"y":4,"w":16,"h":12}"#],
            &settled,
        );
        assert_eq!(named.len(), 1);
        draw(&mut frame, &settled, &named);
        assert_eq!(frame[4 * W + 10], GREEN_Y, "the top edge is drawn");
        assert_eq!(frame[5 * W + 10], GREEN_Y, "two rows thick");
        assert_eq!(frame[10 * W + 4], GREEN_Y, "the left edge is drawn");
        assert_eq!(frame[10 * W + 19], GREEN_Y, "and the right");
        assert_eq!(frame[10 * W + 10], 100, "the interior is untouched");
        assert_eq!(frame[2 * W + 10], 100, "and so is outside the box");
    }

    #[test]
    fn the_outline_colours_the_chroma_under_its_edges() {
        let mut frame = grey_yuv();
        let settled = opened(PixFmt::Yuv420p, 2);
        draw(
            &mut frame,
            &settled,
            &rows(&[r#"{"x":4,"y":4,"w":16,"h":12}"#], &settled),
        );
        let pixels = W * H;
        let (cw, ch) = (W / 2, H / 2);
        assert_eq!(frame[pixels + 2 * cw + 5], GREEN_U, "U under the top edge");
        assert_eq!(
            frame[pixels + cw * ch + 2 * cw + 5],
            GREEN_V,
            "V under the top edge"
        );
        assert_eq!(
            frame[pixels + 4 * cw + 5],
            100,
            "chroma inside the box is untouched"
        );
    }

    #[test]
    fn a_thickness_wider_than_the_box_fills_it_without_spilling() {
        let mut frame = grey_yuv();
        let settled = opened(PixFmt::Yuv420p, 8);
        draw(
            &mut frame,
            &settled,
            &rows(&[r#"{"x":10,"y":10,"w":6,"h":6}"#], &settled),
        );
        for y in 10..16 {
            for x in 10..16 {
                assert_eq!(frame[y * W + x], GREEN_Y, "({x},{y}) is filled");
            }
        }
        assert_eq!(frame[9 * W + 12], 100, "nothing above the box");
        assert_eq!(frame[12 * W + 16], 100, "nothing right of it");
    }

    #[test]
    fn a_box_running_off_the_frame_is_clipped_and_one_outside_is_skipped() {
        let mut frame = grey_yuv();
        let settled = opened(PixFmt::Yuv420p, 2);
        let named = rows(
            &[
                r#"{"x":-4,"y":-4,"w":10,"h":10}"#,
                r#"{"x":100,"y":100,"w":10,"h":10}"#,
            ],
            &settled,
        );
        assert_eq!(
            named.len(),
            1,
            "the clipped one still draws, the one outside does not"
        );
        draw(&mut frame, &settled, &named);
        assert_eq!(frame[0], GREEN_Y, "its visible corner is painted");
    }

    #[test]
    fn rgba_paints_green_and_keeps_alpha() {
        let mut frame = vec![100u8; W * H * 4];
        let settled = opened(PixFmt::Rgba, 1);
        draw(
            &mut frame,
            &settled,
            &rows(&[r#"{"x":4,"y":4,"w":8,"h":8}"#], &settled),
        );
        let at = (4 * W + 6) * 4;
        assert_eq!(&frame[at..at + 4], &[0, 255, 0, 100], "green, alpha kept");
        let inside = (8 * W + 8) * 4;
        assert_eq!(frame[inside], 100, "the interior is untouched");
    }

    #[test]
    fn a_frame_with_no_box_on_it_leaves_as_it_arrived() {
        let mut node = harness();
        let outside = r#"{"class":"face","conf":0.9,"x":100,"y":100,"w":4,"h":4}"#;
        for rows in [vec![], vec![outside]] {
            let mut tick = node
                .tick(2)
                .frame_with(0, 2, Some(1), &[], vec![7; W * H * 4]);
            for row in &rows {
                tick = tick.message(1, 2, row.as_bytes());
            }
            let emitted = node.process(&tick).unwrap();
            assert_eq!(
                emitted.on("v"),
                [&Payload::Same {
                    pts: 2,
                    duration: Some(1),
                    id: 0,
                    index: 0
                }],
                "{rows:?}"
            );
        }
    }

    #[test]
    fn boxes_are_whole_pixels_and_a_box_in_fractions_does_not_read() {
        let shape = harness().shape().clone();
        let boxes = shape.find_input("boxes").expect("a rows input");
        let schema: serde_json::Value =
            serde_json::from_str(boxes.schema.as_deref().expect("a schema")).unwrap();
        for field in ["x", "y", "w", "h"] {
            assert_eq!(schema["properties"][field]["type"], "integer", "{field}");
        }
        assert!(ffrwd_node::parse::<Rect>(r#"{"x":4.5,"y":4,"w":8,"h":8}"#).is_err());
        let output = &shape.outputs[0];
        assert_eq!(
            output.like.as_ref().and_then(|like| like.port.as_deref()),
            Some("v")
        );
        assert!(shape.pure && shape.one_to_one);
    }

    #[test]
    fn params_outside_the_schema_are_refused_by_name() {
        for bad in [
            r#"{"thickness":0}"#,
            r#"{"thickness":65}"#,
            r#"{"radius":3}"#,
        ] {
            assert!(read_params::<Params>(PARAMS_SCHEMA, bad).is_err(), "{bad}");
        }
        let (parsed, _) = read_params::<Params>(PARAMS_SCHEMA, "").expect("defaults");
        assert_eq!(parsed.thickness, 2);
    }
}
