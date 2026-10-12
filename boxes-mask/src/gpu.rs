//! The GPU build: the host build's node beside `ffrwd:gpu`, with a `memory`
//! param. The picture is still read for its times and size alone.
//!
//! On `memory = "gpu"` the matte is painted on the device and leaves through
//! `ffrwd:gpu/frames.emit` as an rgba texture, the matte in red, green and
//! blue and alpha full. Each frame's boxes go up as their footprints, worked
//! out on the CPU exactly as the host build works them out ([`footprint`]),
//! and one pass combines them per pixel with the host build's integer
//! arithmetic, so the device's matte is the host build's byte for byte. On
//! `host` the node is the host build's own, and its matte leaves in gray.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use rfdetr_gpu::device::{self, words, Pass, Slot, Upload};
use rfdetr_gpu::frames::emit;
use rfdetr_gpu::webgpu::{GpuDevice, GpuTexture};
use rfdetr_gpu::{same_memory, Held, Memory};
use serde::Deserialize;

use super::{footprint, Params, Rect};

/// The host build's params, and `memory`.
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"grow":{"type":"number","minimum":0,"maximum":4096,"default":0},"feather":{"type":"number","minimum":0,"maximum":4096,"default":0},"memory":{"type":"string","enum":["host","gpu"],"default":"host"}},"additionalProperties":false}"#;

/// The matte's output on the device. It is not named `v`, as the host
/// build's is: `ports.memory` names a port, and `v` is the input's name too.
const OUT: &str = "out";

/// Words a box takes in the buffer the pass reads, ahead of the ramps: its
/// footprint's columns `x0..x1` and rows `y0..y1`, then where in the buffer
/// its column ramp and its row ramp start.
const HEADER: usize = 8;

/// The pass: each pixel takes the largest coverage any box gives it, a box's
/// coverage being its column's alpha times its row's, in eight bits, as
/// [`paint`](super::paint) combines them.
const PASS: &str = r#"
struct Frame {
    width: u32,
    height: u32,
    count: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> boxes: array<u32>;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var matte: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= frame.width || id.y >= frame.height) {
        return;
    }
    var value = 0u;
    for (var i = 0u; i < frame.count; i = i + 1u) {
        let at = i * 8u;
        if (id.x < boxes[at] || id.x >= boxes[at + 1u] || id.y < boxes[at + 2u] || id.y >= boxes[at + 3u]) {
            continue;
        }
        let ax = boxes[boxes[at + 4u] + id.x - boxes[at]];
        let ay = boxes[boxes[at + 5u] + id.y - boxes[at + 2u]];
        value = max(value, (ax * ay + 127u) / 255u);
    }
    let c = f32(value) / 255.0;
    textureStore(matte, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(c, c, c, 1.0));
}
"#;

#[derive(Clone, Copy, Deserialize)]
pub struct GpuParams {
    grow: f64,
    feather: f64,
    memory: Memory,
}

impl GpuParams {
    fn host(self) -> Params {
        Params {
            grow: self.grow,
            feather: self.feather,
        }
    }
}

/// The boxes as the words the pass reads: a header per box, then the ramps.
fn upload(rects: &[Rect], width: usize, height: usize, params: Params) -> (u32, Vec<u32>) {
    let footprints: Vec<_> = rects
        .iter()
        .filter_map(|rect| footprint(width, height, rect, params))
        .collect();
    let mut header = Vec::with_capacity(footprints.len() * HEADER);
    let mut ramps = Vec::new();
    let mut at = footprints.len() * HEADER;
    for found in &footprints {
        let (columns, rows) = (found.columns.len(), found.rows.len());
        header.extend([
            found.first_x as u32,
            (found.first_x + columns) as u32,
            found.first_y as u32,
            (found.first_y + rows) as u32,
            at as u32,
            (at + columns) as u32,
            0,
            0,
        ]);
        ramps.extend(found.columns.iter().map(|&alpha| u32::from(alpha)));
        ramps.extend(found.rows.iter().map(|&alpha| u32::from(alpha)));
        at += columns + rows;
    }
    header.extend(ramps);
    (footprints.len() as u32, header)
}

/// The device's half: the pass and the buffers it reads.
pub struct OnDevice {
    v: u32,
    boxes: u32,
    width: usize,
    height: usize,
    params: Params,
    device: GpuDevice,
    pass: Pass,
    rects: Upload,
    frame: Upload,
}

impl OnDevice {
    fn open(params: Params, init: &Init) -> Result<OnDevice> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("boxes_mask reads frames, and `v` is not video")?;
        let device = device::open("boxes_mask")?;
        Ok(OnDevice {
            v: v.id,
            boxes: init.stream("boxes")?.id,
            width: video.width as usize,
            height: video.height as usize,
            params,
            pass: Pass::new(
                &device,
                PASS,
                "paint",
                &[Slot::Storage, Slot::Uniform, Slot::Target],
            ),
            rects: Upload::storage(&device),
            frame: Upload::uniform(&device),
            device,
        })
    }

    /// One frame's boxes painted on the device: the texture that leaves.
    fn paint(&mut self, rects: &[Rect]) -> Result<GpuTexture, String> {
        let (count, boxes) = upload(rects, self.width, self.height, self.params);
        let (width, height) = (self.width as u32, self.height as u32);
        let device = &self.device;
        let boxes = self.rects.write(device, &words(&boxes))?;
        let frame = self
            .frame
            .write(device, &words(&[width, height, count, 0]))?;
        let matte = device::target(device, width, height);
        let view = matte.create_view(None);
        self.pass.run(
            device,
            &[
                device::Bound::Buffer(boxes),
                device::Bound::Buffer(frame),
                device::Bound::View(&view),
            ],
            width,
            height,
        )?;
        Ok(matte)
    }
}

/// The GPU build's node: on `host`, the host build's node itself.
pub enum BoxesMask {
    Host(super::BoxesMask),
    Gpu(OnDevice),
}

impl Node for BoxesMask {
    const NAME: &'static str = "boxes_mask";
    const VERSION: &'static str = "0.2.1";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = GpuParams;

    fn shape(params: &GpuParams, bound: &Bound) -> Result<Shape> {
        match params.memory {
            Memory::Host => super::BoxesMask::shape(&params.host(), bound),
            Memory::Gpu => Ok(Shape::new()
                .input(Input::video("v").clock().timing())
                .input(Input::rows("boxes").schema::<Rect>())
                .output(Output::video(OUT).following("v").pixel_format("rgba"))
                .pure()
                .one_to_one()),
        }
    }

    fn init(params: GpuParams, init: &Init) -> Result<BoxesMask> {
        Ok(match params.memory {
            Memory::Host => BoxesMask::Host(super::BoxesMask::init(params.host(), init)?),
            Memory::Gpu => BoxesMask::Gpu(OnDevice::open(params.host(), init)?),
        })
    }

    fn set_params(&mut self, params: GpuParams) -> Result<()> {
        match self {
            BoxesMask::Host(node) => {
                same_memory(Self::NAME, Memory::Host, params.memory)?;
                node.set_params(params.host())
            }
            BoxesMask::Gpu(node) => {
                same_memory(Self::NAME, Memory::Gpu, params.memory)?;
                node.params = params.host();
                Ok(())
            }
        }
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let node = match self {
            BoxesMask::Host(node) => return node.process(tick, out),
            BoxesMask::Gpu(node) => node,
        };
        let Some(frame) = tick.frame(node.v) else {
            return Ok(());
        };
        let rects: Vec<Rect> = tick.rows(node.boxes)?;
        let matte = node.paint(&rects)?;
        Ok(emit(OUT, frame.pts, frame.duration, matte)?)
    }
}

impl Held for BoxesMask {
    /// The matte; the picture is read for its timing alone, so it is never
    /// held anywhere.
    const HELD: &'static [&'static str] = &[OUT];

    fn memory(params: &GpuParams) -> Memory {
        params.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{read_params, BoundStream, Payload, Rational, Runner, Wants};
    use rfdetr_gpu::held;

    use crate::rasterize;

    const W: usize = 64;
    const H: usize = 48;

    fn rects(rows: &[&str]) -> Vec<Rect> {
        rows.iter()
            .map(|row| ffrwd_node::parse(row).unwrap())
            .collect()
    }

    /// What the pass computes, word for word over the upload: the pass's
    /// arithmetic, on the CPU.
    fn combined(count: u32, boxes: &[u32], width: usize, height: usize) -> Vec<u8> {
        let mut map = vec![0u8; width * height];
        for y in 0..height as u32 {
            for x in 0..width as u32 {
                let mut value = 0u32;
                for i in 0..count as usize {
                    let at = i * HEADER;
                    let (x0, x1, y0, y1) = (boxes[at], boxes[at + 1], boxes[at + 2], boxes[at + 3]);
                    if x < x0 || x >= x1 || y < y0 || y >= y1 {
                        continue;
                    }
                    let ax = boxes[(boxes[at + 4] + x - x0) as usize];
                    let ay = boxes[(boxes[at + 5] + y - y0) as usize];
                    value = value.max((ax * ay + 127) / 255);
                }
                map[y as usize * width + x as usize] = value as u8;
            }
        }
        map
    }

    #[test]
    fn the_upload_combined_as_the_pass_combines_it_is_the_host_builds_matte() {
        let boxes = rects(&[
            r#"{"x":10,"y":10,"w":20,"h":10}"#,
            r#"{"x":15,"y":12,"w":10,"h":20}"#,
            r#"{"x":-10,"y":-10,"w":30,"h":30}"#,
            r#"{"x":60,"y":40,"w":30,"h":30}"#,
            r#"{"x":30,"y":30,"w":0,"h":5}"#,
            r#"{"x":100,"y":100,"w":5,"h":5}"#,
        ]);
        // Feathers whose ramps land on a rounding tie in f64 and not in f32:
        // a fifth of 255 out from an edge is 229.5.
        for (grow, feather) in [(0.0, 0.0), (8.0, 4.0), (2.5, 5.0), (0.0, 7.3), (3.0, 255.0)] {
            let params = Params { grow, feather };
            let (count, words) = upload(&boxes, W, H, params);
            assert_eq!(
                combined(count, &words, W, H),
                rasterize(&boxes, W, H, params),
                "grow {grow}, feather {feather}"
            );
        }
    }

    #[test]
    fn no_boxes_is_an_empty_upload() {
        let (count, words) = upload(
            &[],
            W,
            H,
            Params {
                grow: 8.0,
                feather: 4.0,
            },
        );
        assert_eq!((count, words.len()), (0, 0));
    }

    #[test]
    fn on_the_gpu_the_matte_is_rgba_and_held_there() {
        let shape =
            Runner::<BoxesMask>::shape(r#"{"memory":"gpu"}"#, &["v".to_owned()]).expect("a shape");
        let v = shape.find_input("v").expect("the picture");
        assert_eq!(
            v.accepts.wants,
            Wants::Timing,
            "its times and size, never its pixels"
        );
        let out = shape.find_output(OUT).expect("the matte");
        let like = out.like.as_ref().expect("follows the picture");
        assert_eq!(
            (like.port.as_deref(), like.pixel_format.as_deref()),
            (Some("v"), Some("rgba"))
        );
        assert!(shape.pure && shape.one_to_one);
        assert_eq!(held::<BoxesMask>(r#"{"memory":"gpu","grow":8}"#), [OUT]);
    }

    #[test]
    fn on_the_host_the_matte_is_gray_and_nothing_is_held_on_the_gpu() {
        for params in ["", r#"{"memory":"host","feather":4}"#] {
            let shape = Runner::<BoxesMask>::shape(params, &["v".to_owned()]).expect("a shape");
            let host = Runner::<crate::BoxesMask>::shape("", &["v".to_owned()]).unwrap();
            assert_eq!(shape, host, "the host build's own shape: {params}");
            assert!(held::<BoxesMask>(params).is_empty(), "{params}");
        }
    }

    #[test]
    fn params_default_to_host_memory_and_refuse_what_the_host_build_refuses() {
        let (parsed, _) = read_params::<GpuParams>(PARAMS_SCHEMA, "").expect("the defaults");
        assert_eq!(
            (parsed.grow, parsed.feather, parsed.memory),
            (0.0, 0.0, Memory::Host)
        );
        for bad in [
            r#"{"grow":-1}"#,
            r#"{"feather":5000}"#,
            r#"{"memory":"disk"}"#,
        ] {
            assert!(
                read_params::<GpuParams>(PARAMS_SCHEMA, bad).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn on_the_host_a_frame_leaves_as_the_host_builds_matte() {
        let tb = Rational::new(1, 25);
        let bound = vec![
            BoundStream::video("v", 0, W as u32, H as u32, "yuv420p", tb),
            BoundStream::rows("boxes", 1, tb),
        ];
        let mut node = Harness::<BoxesMask>::new(r#"{"grow":2,"memory":"host"}"#, bound).unwrap();
        let row = r#"{"class":"face","conf":0.9,"x":10,"y":10,"w":4,"h":4}"#;
        let tick = node
            .tick(3)
            .frame_with(0, 3, Some(1), &[], Vec::new())
            .message(1, 3, row.as_bytes());
        let emitted = node.process(&tick).unwrap();
        let [Payload::Frame { pts: 3, data, .. }] = emitted.on("v")[..] else {
            panic!("no matte: {emitted:?}")
        };
        let want = rasterize(
            &rects(&[row]),
            W,
            H,
            Params {
                grow: 2.0,
                feather: 0.0,
            },
        );
        assert_eq!(data, &want);
    }
}
