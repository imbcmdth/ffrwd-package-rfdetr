//! The GPU build: the host build's node beside `ffrwd:gpu`, with a `memory`
//! param.
//!
//! On `memory = "gpu"` the picture is held on the device, as rgba, and one
//! pass copies it into a new texture with the outlines drawn as the host
//! build draws them on rgba: green, `thickness` pixels inward from each
//! box's edge, alpha kept. That texture leaves through
//! `ffrwd:gpu/frames.emit`, and a frame with no box on it leaves as it
//! arrived. On `host` the node is the host build's own.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use rfdetr_gpu::device::{self, words, Pass, Slot, Upload};
use rfdetr_gpu::frames::{emit, planes};
use rfdetr_gpu::webgpu::{GpuDevice, GpuTexture};
use rfdetr_gpu::{same_memory, Held, Memory};
use serde::Deserialize;

use super::{clamp, Params, Rect, Region};

/// The host build's params, and `memory`.
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"thickness":{"type":"integer","minimum":1,"maximum":64,"default":2},"memory":{"type":"string","enum":["host","gpu"],"default":"host"}},"additionalProperties":false}"#;

/// The picture's output on the device. It is not named `v`, as the host
/// build's is: `ports.memory` names a port, and `v` is the input's name too.
const OUT: &str = "out";

/// The pass: every pixel copied, and the ones in a box's outline made green.
/// A pixel is in the outline when it is inside the box and within
/// `thickness` of one of its edges, which is the union of the four bands
/// [`outline`](super::outline) fills.
const PASS: &str = r#"
struct Frame {
    width: u32,
    height: u32,
    count: u32,
    thickness: u32,
}

@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> regions: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> frame: Frame;
@group(0) @binding(3) var drawn: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= frame.width || id.y >= frame.height) {
        return;
    }
    let at = vec2<i32>(i32(id.x), i32(id.y));
    var colour = textureLoad(picture, at, 0);
    let t = frame.thickness;
    for (var i = 0u; i < frame.count; i = i + 1u) {
        // x0, y0, x1, y1, exclusive on the right and bottom.
        let r = regions[i];
        if (id.x < r.x || id.x >= r.z || id.y < r.y || id.y >= r.w) {
            continue;
        }
        if (id.y < r.y + t || id.y + t >= r.w || id.x < r.x + t || id.x + t >= r.z) {
            colour = vec4<f32>(0.0, 1.0, 0.0, colour.a);
            break;
        }
    }
    textureStore(drawn, at, colour);
}
"#;

#[derive(Clone, Copy, Deserialize)]
pub struct GpuParams {
    thickness: u32,
    memory: Memory,
}

impl GpuParams {
    fn host(self) -> Params {
        Params {
            thickness: self.thickness,
        }
    }
}

/// The regions the boxes name on a `width` by `height` frame, as the words
/// the pass reads.
fn upload(rects: &[Rect], width: usize, height: usize) -> Vec<u32> {
    rects
        .iter()
        .filter_map(|rect| clamp(rect, width, height))
        .flat_map(|region: Region| {
            [region.x0, region.y0, region.x1, region.y1].map(|edge| edge as u32)
        })
        .collect()
}

/// The device's half: the pass and the buffers it reads.
pub struct OnDevice {
    v: u32,
    boxes: u32,
    width: usize,
    height: usize,
    thickness: u32,
    device: GpuDevice,
    pass: Pass,
    regions: Upload,
    frame: Upload,
}

impl OnDevice {
    fn open(thickness: u32, init: &Init) -> Result<OnDevice> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("draw_boxes reads frames, and `v` is not video")?;
        if video.pix_fmt != "rgba" {
            return Err(format!(
                "draw_boxes on the GPU draws on rgba, and `v` is {}",
                video.pix_fmt
            )
            .into());
        }
        let device = device::open("draw_boxes")?;
        Ok(OnDevice {
            v: v.id,
            boxes: init.stream("boxes")?.id,
            width: video.width as usize,
            height: video.height as usize,
            thickness,
            pass: Pass::new(
                &device,
                PASS,
                "draw",
                &[Slot::Texture, Slot::Storage, Slot::Uniform, Slot::Target],
            ),
            regions: Upload::storage(&device),
            frame: Upload::uniform(&device),
            device,
        })
    }

    /// Frame `index` with `regions` outlined: the texture that leaves.
    fn draw(&mut self, index: u32, regions: &[u32]) -> Result<GpuTexture, String> {
        let handed = planes(self.v, index)?;
        let [picture] = &handed[..] else {
            return Err(format!(
                "draw_boxes: an rgba frame is one plane, and {} came",
                handed.len()
            ));
        };
        let (width, height) = (self.width as u32, self.height as u32);
        let count = (regions.len() / 4) as u32;
        let device = &self.device;
        let regions = self.regions.write(device, &words(regions))?;
        let frame = self
            .frame
            .write(device, &words(&[width, height, count, self.thickness]))?;
        let drawn = device::target(device, width, height);
        let (from, to) = (picture.create_view(None), drawn.create_view(None));
        self.pass.run(
            device,
            &[
                device::Bound::View(&from),
                device::Bound::Buffer(regions),
                device::Bound::Buffer(frame),
                device::Bound::View(&to),
            ],
            width,
            height,
        )?;
        Ok(drawn)
    }
}

/// The GPU build's node: on `host`, the host build's node itself.
pub enum DrawBoxes {
    Host(super::DrawBoxes),
    Gpu(OnDevice),
}

impl Node for DrawBoxes {
    const NAME: &'static str = "draw_boxes";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = GpuParams;

    fn shape(params: &GpuParams, bound: &Bound) -> Result<Shape> {
        match params.memory {
            Memory::Host => super::DrawBoxes::shape(&params.host(), bound),
            Memory::Gpu => Ok(Shape::new()
                .input(Input::video("v").clock().pixel_formats(&["rgba"]))
                .input(Input::rows("boxes").schema::<Rect>())
                .output(Output::video(OUT).following("v").pixel_format("rgba"))
                .pure()
                .one_to_one()),
        }
    }

    fn init(params: GpuParams, init: &Init) -> Result<DrawBoxes> {
        Ok(match params.memory {
            Memory::Host => DrawBoxes::Host(super::DrawBoxes::init(params.host(), init)?),
            Memory::Gpu => DrawBoxes::Gpu(OnDevice::open(params.thickness, init)?),
        })
    }

    fn set_params(&mut self, params: GpuParams) -> Result<()> {
        match self {
            DrawBoxes::Host(node) => {
                same_memory(Self::NAME, Memory::Host, params.memory)?;
                node.set_params(params.host())
            }
            DrawBoxes::Gpu(node) => {
                same_memory(Self::NAME, Memory::Gpu, params.memory)?;
                node.thickness = params.thickness;
                Ok(())
            }
        }
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let node = match self {
            DrawBoxes::Host(node) => return node.process(tick, out),
            DrawBoxes::Gpu(node) => node,
        };
        let Some(frame) = tick.frame(node.v) else {
            return Ok(());
        };
        let rects: Vec<Rect> = tick.rows(node.boxes)?;
        let regions = upload(&rects, node.width, node.height);
        if regions.is_empty() {
            return Ok(out.pass(OUT, node.v, &frame)?);
        }
        let drawn = node.draw(frame.index, &regions)?;
        Ok(emit(OUT, frame.pts, frame.duration, drawn)?)
    }
}

impl Held for DrawBoxes {
    const HELD: &'static [&'static str] = &["v", OUT];

    fn memory(params: &GpuParams) -> Memory {
        params.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{draw, Opened, PixFmt};
    use ffrwd_node::{read_params, Runner};
    use rfdetr_gpu::held;

    const W: usize = 32;
    const H: usize = 24;

    fn rects(rows: &[&str]) -> Vec<Rect> {
        rows.iter()
            .map(|row| ffrwd_node::parse(row).unwrap())
            .collect()
    }

    /// What the pass computes over the upload, on the CPU: its arithmetic,
    /// pixel by pixel, on an rgba frame.
    fn drawn(picture: &[u8], regions: &[u32], thickness: u32) -> Vec<u8> {
        let mut out = picture.to_vec();
        for y in 0..H as u32 {
            for x in 0..W as u32 {
                for r in regions.as_chunks::<4>().0 {
                    if x < r[0] || x >= r[2] || y < r[1] || y >= r[3] {
                        continue;
                    }
                    if y < r[1] + thickness
                        || y + thickness >= r[3]
                        || x < r[0] + thickness
                        || x + thickness >= r[2]
                    {
                        let at = (y as usize * W + x as usize) * 4;
                        out[at..at + 3].copy_from_slice(&[0, 255, 0]);
                        break;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn the_pass_draws_what_the_host_build_draws_on_rgba() {
        let boxes = rects(&[
            r#"{"x":4,"y":4,"w":16,"h":12}"#,
            r#"{"x":10,"y":10,"w":6,"h":6}"#,
            r#"{"x":-4,"y":-4,"w":10,"h":10}"#,
            r#"{"x":20,"y":2,"w":3,"h":30}"#,
            r#"{"x":100,"y":100,"w":10,"h":10}"#,
        ]);
        let picture: Vec<u8> = (0..W * H * 4).map(|at| (at * 7 % 251) as u8).collect();
        for thickness in [1, 2, 3, 8, 64] {
            let opened = Opened {
                width: W,
                height: H,
                pix_fmt: PixFmt::Rgba,
                params: Params { thickness },
            };
            let mut want = picture.clone();
            draw(&mut want, &opened, &crate::boxes(&boxes, &opened));
            assert_eq!(
                drawn(&picture, &upload(&boxes, W, H), thickness),
                want,
                "thickness {thickness}"
            );
        }
    }

    #[test]
    fn on_the_gpu_the_picture_and_the_drawing_are_rgba_held_there() {
        let shape =
            Runner::<DrawBoxes>::shape(r#"{"memory":"gpu"}"#, &["v".to_owned()]).expect("a shape");
        assert_eq!(
            shape.find_input("v").unwrap().accepts.pixel_formats,
            ["rgba"]
        );
        let like = shape.find_output(OUT).unwrap().like.as_ref().unwrap();
        assert_eq!(
            (like.port.as_deref(), like.pixel_format.as_deref()),
            (Some("v"), Some("rgba"))
        );
        assert!(shape.pure && shape.one_to_one);
        assert_eq!(held::<DrawBoxes>(r#"{"memory":"gpu"}"#), ["v", OUT]);
    }

    #[test]
    fn on_the_host_it_is_the_host_builds_node() {
        for params in ["", r#"{"memory":"host","thickness":4}"#] {
            let shape = Runner::<DrawBoxes>::shape(params, &["v".to_owned()]).expect("a shape");
            let host = Runner::<crate::DrawBoxes>::shape("", &["v".to_owned()]).unwrap();
            assert_eq!(shape, host, "{params}");
            assert!(held::<DrawBoxes>(params).is_empty(), "{params}");
        }
    }

    #[test]
    fn params_default_to_host_memory_and_refuse_what_the_host_build_refuses() {
        let (parsed, _) = read_params::<GpuParams>(PARAMS_SCHEMA, "").expect("the defaults");
        assert_eq!((parsed.thickness, parsed.memory), (2, Memory::Host));
        for bad in [
            r#"{"thickness":0}"#,
            r#"{"thickness":65}"#,
            r#"{"memory":"disk"}"#,
        ] {
            assert!(
                read_params::<GpuParams>(PARAMS_SCHEMA, bad).is_err(),
                "{bad}"
            );
        }
    }
}
