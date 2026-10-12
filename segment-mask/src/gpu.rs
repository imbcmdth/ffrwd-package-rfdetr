//! The GPU build: the host build's node beside `ffrwd:gpu`, with a `memory`
//! param. The host makes each frame into the graph's input
//! (`ffrwd:gpu/nn.frame-tensor`), stretched to the square and resampled as
//! the host build's `ffrwd_frame::tensor` does it, and the boxes and class
//! logits come back and are decoded here as the host build decodes them.
//!
//! On `host` the input is `rgba` in host memory, the host makes the tensor
//! on the CPU, the masks come back too, and the matte is painted on the CPU
//! and leaves in gray, as the host build's does. On `gpu` the input is held
//! on the device, the tensor is made there, and the masks are bound to a
//! texture (`ffrwd:gpu/nn.output-texture`), so where the model's session
//! writes the device's memory they never come back: a pass samples each
//! surviving instance's mask over its box, as [`matte`](super::matte) does,
//! and the matte leaves as an rgba texture.
//!
//! The masks are found by shape, as the host build finds them, since one
//! published export names its outputs in the wrong places; `output-texture`
//! binds by name, so the name is learned at `init` from one run of the graph
//! over a blank square.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use rfdetr_common::detector::INPUT_NAME;
use rfdetr_common::nn::wasi::nn::tensor::{Tensor, TensorType};
use rfdetr_common::nn::{Model, Returned};
use rfdetr_common::{frame_box, Taps};
use rfdetr_gpu::device::{self, words, Pass, Slot, Upload};
use rfdetr_gpu::frames::emit;
use rfdetr_gpu::nn::{frame_tensor, output_texture, FrameTensorOptions, OutputTextureOptions};
use rfdetr_gpu::webgpu::{GpuBuffer, GpuDevice, GpuTexture, GpuTextureView};
use rfdetr_gpu::{same_memory, whole_frame, Held, Memory, GPU_PIXEL_FORMATS};
use serde::Deserialize;

use super::{decode, matte, outputs, settle, Instance, Params, Settled, SIDE};

/// The host build's params, and `memory`.
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"class":{"type":"string"},"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25},"memory":{"type":"string","enum":["host","gpu"],"default":"host"}},"additionalProperties":false}"#;

/// The matte's output on the device. It is not named `v`, as the host
/// build's is: `ports.memory` names a port, and `v` is the input's name too.
const OUT: &str = "out";

/// The pass. Each frame pixel inside an instance's box samples that
/// instance's mask where [`matte`](super::matte) samples it - the same taps,
/// worked out on the CPU, and the same two mixes in the same order - and is
/// kept where the logit is above zero.
const PASS: &str = r#"
struct Frame {
    width: u32,
    height: u32,
    count: u32,
    mask_w: u32,
    mask_h: u32,
    texture_w: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var masks: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> taps: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> frame: Frame;
@group(0) @binding(4) var matte: texture_storage_2d<rgba8unorm, write>;

// Mask row `y` of the planes stacked one query after another, column `x`:
// the masks' elements in order, `texture_w` to a texture row.
fn at(x: u32, y: u32) -> f32 {
    let element = y * frame.mask_w + x;
    let texel = vec2<i32>(i32(element % frame.texture_w), i32(element / frame.texture_w));
    return textureLoad(masks, texel, 0).r;
}

@compute @workgroup_size(8, 8)
fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= frame.width || id.y >= frame.height) {
        return;
    }
    // A column's taps, then a row's: the two samples and how far along.
    let column = id.x * 3u;
    let row = (frame.width + id.y) * 3u;
    let x0 = taps[column];
    let x1 = taps[column + 1u];
    let tx = bitcast<f32>(taps[column + 2u]);
    let y0 = taps[row];
    let y1 = taps[row + 1u];
    let ty = bitcast<f32>(taps[row + 2u]);
    var keep = 0.0;
    for (var i = 0u; i < frame.count; i = i + 1u) {
        let box = instances[i * 2u];
        if (id.x < box.x || id.x >= box.z || id.y < box.y || id.y >= box.w) {
            continue;
        }
        let plane = instances[i * 2u + 1u].x * frame.mask_h;
        let a = at(x0, plane + y0);
        let b = at(x1, plane + y0);
        let c = at(x0, plane + y1);
        let d = at(x1, plane + y1);
        let top = a + (b - a) * tx;
        let bottom = c + (d - c) * tx;
        if (top + (bottom - top) * ty > 0.0) {
            keep = 1.0;
            break;
        }
    }
    textureStore(matte, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(keep, keep, keep, 1.0));
}
"#;

#[derive(Deserialize)]
pub struct GpuParams {
    class: Option<String>,
    conf: f64,
    memory: Memory,
}

impl GpuParams {
    fn settle(&self) -> Result<Settled, String> {
        settle(&Params {
            class: self.class.clone(),
            conf: self.conf,
        })
    }
}

/// Every frame column's taps and then every frame row's, as the words the
/// pass reads: the low and high sample and the fraction's bits. They are
/// the taps [`matte`](super::matte) builds for each box, which depend on
/// the frame pixel alone.
fn taps(mask: (usize, usize), frame: (usize, usize)) -> Vec<u32> {
    let (mask_w, mask_h) = mask;
    let (width, height) = frame;
    let per_frame_x = mask_w as f32 / width as f32;
    let per_frame_y = mask_h as f32 / height as f32;
    let columns = Taps::build(width, mask_w, |x| (x as f32 + 0.5) * per_frame_x - 0.5);
    let rows = Taps::build(height, mask_h, |y| (y as f32 + 0.5) * per_frame_y - 0.5);
    let mut words = Vec::with_capacity(3 * (width + height));
    for map in [columns, rows] {
        for ((low, high), fraction) in map.low.iter().zip(&map.high).zip(&map.fraction) {
            words.extend([*low as u32, *high as u32, fraction.to_bits()]);
        }
    }
    words
}

/// The instances as the words the pass reads: each one's box on the frame,
/// then its query, the plane its mask is. An instance whose box names no
/// pixel, or whose query has no plane, paints nothing, as on the CPU.
fn upload(instances: &[Instance], queries: usize, frame: (usize, usize)) -> Vec<u32> {
    let mut words = Vec::with_capacity(instances.len() * 8);
    for instance in instances {
        let (x0, y0, x1, y1) = frame_box(
            (instance.cx, instance.cy, instance.w, instance.h),
            frame.0,
            frame.1,
        );
        if x0 == x1 || y0 == y1 || instance.query >= queries {
            continue;
        }
        words.extend([x0, y0, x1, y1].map(|edge| edge as u32));
        words.extend([instance.query as u32, 0, 0, 0]);
    }
    words
}

/// The widest a texture is made, which every WebGPU device allows.
const TEXTURE_LIMIT: usize = 8192;

/// The size of a texture that holds `elements` floats one a texel, row by
/// row, within the limit both ways: the widest that divides them evenly. A
/// mask plane per row is too wide for most exports, and a mask column per
/// plane too tall, so the planes run on from one texture row to the next.
fn texture_size(elements: usize) -> Option<(usize, usize)> {
    (1..=elements.min(TEXTURE_LIMIT))
        .rev()
        .find(|width| elements.is_multiple_of(*width))
        .map(|width| (width, elements / width))
        .filter(|(_, height)| *height <= TEXTURE_LIMIT)
}

/// The device's half: the masks' texture, the pass and what it reads.
struct Painter {
    device: GpuDevice,
    pass: Pass,
    /// Bound to the graph's masks output for the life of the context.
    _masks: GpuTexture,
    masks_view: GpuTextureView,
    /// The masks' `[1, queries, height, width]`.
    queries: usize,
    mask: (usize, usize),
    /// The masks' texture's width.
    texture_w: usize,
    taps: GpuBuffer,
    instances: Upload,
    frame: Upload,
}

pub struct SegmentMask {
    v: u32,
    width: usize,
    height: usize,
    settled: Settled,
    memory: Memory,
    options: FrameTensorOptions,
    model: Model,
    /// On `memory = "gpu"`.
    painter: Option<Painter>,
}

/// The masks output's name and shape, learned from one run of the graph over
/// a blank square: which output is the masks is known from what comes back.
fn learn_masks(model: &mut Model) -> Result<(String, Vec<u32>), String> {
    let side = SIDE as u32;
    let dimensions = [1, 3, side, side];
    let blank = vec![0u8; 3 * SIDE * SIDE * 4];
    let returned = model.compute(|| Ok(Tensor::new(&dimensions, TensorType::Fp32, &blank)))?;
    let shapes: Vec<Vec<u32>> = returned.iter().map(|(_, t)| t.dimensions()).collect();
    let (_, _, masks) = outputs(&shapes)?;
    Ok((returned[masks].0.clone(), shapes[masks].clone()))
}

impl Painter {
    fn open(model: &mut Model, frame: (usize, usize)) -> Result<Painter, String> {
        let (name, shape) = learn_masks(model)?;
        let [_, queries, mask_h, mask_w] = shape[..] else {
            return Err(format!("segment_mask: the masks are {shape:?}"));
        };
        let elements = (queries * mask_h * mask_w) as usize;
        let (texture_w, texture_h) = texture_size(elements).ok_or_else(|| {
            format!("segment_mask: the masks are {shape:?}, too many for one texture")
        })?;
        let device = device::open("segment_mask")?;
        let masks = output_texture(
            model.context(),
            &OutputTextureOptions {
                output: name,
                width: texture_w as u32,
                height: texture_h as u32,
            },
        )
        .map_err(|e| format!("segment_mask: {e}"))?;
        let mask = (mask_w as usize, mask_h as usize);
        let mut taps_upload = Upload::storage(&device);
        taps_upload.write(&device, &words(&taps(mask, frame)))?;
        Ok(Painter {
            pass: Pass::new(
                &device,
                PASS,
                "paint",
                &[
                    Slot::Texture,
                    Slot::Storage,
                    Slot::Storage,
                    Slot::Uniform,
                    Slot::Target,
                ],
            ),
            masks_view: masks.create_view(None),
            _masks: masks,
            queries: queries as usize,
            mask,
            texture_w,
            taps: taps_upload.into_buffer(),
            instances: Upload::storage(&device),
            frame: Upload::uniform(&device),
            device,
        })
    }

    /// The found instances painted on the device: the texture that leaves.
    fn paint(&mut self, found: &[Instance], frame: (usize, usize)) -> Result<GpuTexture, String> {
        let instances = upload(found, self.queries, frame);
        let count = (instances.len() / 8) as u32;
        let (width, height) = (frame.0 as u32, frame.1 as u32);
        let device = &self.device;
        let instances = self.instances.write(device, &words(&instances))?;
        let (mask_w, mask_h) = (self.mask.0 as u32, self.mask.1 as u32);
        let texture_w = self.texture_w as u32;
        let uniform = self.frame.write(
            device,
            &words(&[width, height, count, mask_w, mask_h, texture_w, 0, 0]),
        )?;
        let target = device::target(device, width, height);
        let view = target.create_view(None);
        self.pass.run(
            device,
            &[
                device::Bound::View(&self.masks_view),
                device::Bound::Buffer(&self.taps),
                device::Bound::Buffer(instances),
                device::Bound::Buffer(uniform),
                device::Bound::View(&view),
            ],
            width,
            height,
        )?;
        Ok(target)
    }
}

/// What one frame's run left: the matte's bytes on the host, or its texture.
enum Matte {
    Bytes(Vec<u8>),
    Texture(GpuTexture),
}

impl SegmentMask {
    fn run(&mut self, index: u32) -> Result<Matte, String> {
        let (v, options) = (self.v, &self.options);
        let returned = self.model.compute(|| {
            frame_tensor(v, index, options).map_err(|e| format!("segment_mask: {e}"))
        })?;
        let shapes: Vec<Vec<u32>> = returned.iter().map(|(_, t)| t.dimensions()).collect();
        let (boxes, logits, masks) = outputs(&shapes)?;
        let found = decode(
            &Returned::of(&returned[boxes].1).values,
            &Returned::of(&returned[logits].1).values,
            self.settled,
        )?;
        let frame = (self.width, self.height);
        let (mask_h, mask_w) = (shapes[masks][2] as usize, shapes[masks][3] as usize);
        match &mut self.painter {
            Some(painter) => {
                if (mask_w, mask_h) != painter.mask || shapes[masks][1] as usize != painter.queries
                {
                    return Err(format!(
                        "segment_mask: the masks came back {:?}, and were bound as {} \
                         of {}x{}",
                        shapes[masks], painter.queries, painter.mask.0, painter.mask.1
                    ));
                }
                Ok(Matte::Texture(painter.paint(&found, frame)?))
            }
            None => Ok(Matte::Bytes(matte(
                &found,
                &Returned::of(&returned[masks].1).values,
                (mask_w, mask_h),
                frame,
            ))),
        }
    }
}

impl Node for SegmentMask {
    const NAME: &'static str = "segment_mask";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = GpuParams;

    fn shape(params: &GpuParams, bound: &Bound) -> Result<Shape> {
        match params.memory {
            Memory::Host => super::SegmentMask::shape(
                &Params {
                    class: params.class.clone(),
                    conf: params.conf,
                },
                bound,
            ),
            Memory::Gpu => Ok(Shape::new()
                .input(Input::video("v").clock().pixel_formats(GPU_PIXEL_FORMATS))
                .output(Output::video(OUT).following("v").pixel_format("rgba"))
                .pure()
                .one_to_one()),
        }
    }

    fn init(params: GpuParams, init: &Init) -> Result<SegmentMask> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("segment_mask reads frames, and `v` is not video")?;
        let frame = (video.width as usize, video.height as usize);
        let mut model = Model::load(Self::NAME, Self::NAME, INPUT_NAME)?;
        let painter = match params.memory {
            Memory::Host => None,
            Memory::Gpu => Some(Painter::open(&mut model, frame)?),
        };
        Ok(SegmentMask {
            v: v.id,
            width: frame.0,
            height: frame.1,
            settled: params.settle()?,
            memory: params.memory,
            options: whole_frame(Self::NAME, SIDE),
            model,
            painter,
        })
    }

    fn set_params(&mut self, params: GpuParams) -> Result<()> {
        same_memory(Self::NAME, self.memory, params.memory)?;
        self.settled = params.settle()?;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.v) {
            match self.run(frame.index)? {
                Matte::Texture(texture) => emit(OUT, frame.pts, frame.duration, texture)?,
                Matte::Bytes(map) => out.frame("v", frame.pts, frame.duration, map)?,
            }
        }
        Ok(())
    }
}

impl Held for SegmentMask {
    const HELD: &'static [&'static str] = &["v", OUT];

    fn memory(params: &GpuParams) -> Memory {
        params.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::{read_params, Runner};
    use rfdetr_gpu::held;

    /// What the pass computes over the uploads, on the CPU: its arithmetic,
    /// pixel by pixel.
    fn painted(
        instances: &[u32],
        taps: &[u32],
        masks: &[f32],
        mask: (usize, usize),
        frame: (usize, usize),
    ) -> Vec<u8> {
        let (mask_w, mask_h) = mask;
        let (width, height) = frame;
        let elements = masks.len();
        let (texture_w, _) = texture_size(elements).expect("fits a texture");
        // The texture's texels row by row, as the pass reads them.
        let at = |x: u32, y: u32| {
            let element = y as usize * mask_w + x as usize;
            masks[(element / texture_w) * texture_w + element % texture_w]
        };
        let mut map = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let (column, row) = (x * 3, (width + y) * 3);
                let (x0, x1, tx) = (
                    taps[column],
                    taps[column + 1],
                    f32::from_bits(taps[column + 2]),
                );
                let (y0, y1, ty) = (taps[row], taps[row + 1], f32::from_bits(taps[row + 2]));
                for one in instances.as_chunks::<8>().0 {
                    let (px, py) = (x as u32, y as u32);
                    if px < one[0] || px >= one[2] || py < one[1] || py >= one[3] {
                        continue;
                    }
                    let plane = one[4] * mask_h as u32;
                    let (a, b) = (at(x0, plane + y0), at(x1, plane + y0));
                    let (c, d) = (at(x0, plane + y1), at(x1, plane + y1));
                    let top = a + (b - a) * tx;
                    let bottom = c + (d - c) * tx;
                    if top + (bottom - top) * ty > 0.0 {
                        map[y * width + x] = 255;
                        break;
                    }
                }
            }
        }
        map
    }

    /// Masks whose logits change sign across each plane, a different slope
    /// per query, so the sampling decides every edge.
    fn ramps(queries: usize, mask: (usize, usize)) -> Vec<f32> {
        let (w, h) = mask;
        let mut all = Vec::with_capacity(queries * w * h);
        for q in 0..queries {
            for y in 0..h {
                for x in 0..w {
                    let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                    all.push((fx - 0.4) * (q as f32 + 1.0) + (fy - 0.6) * 0.7 - 0.05 * q as f32);
                }
            }
        }
        all
    }

    #[test]
    fn the_pass_paints_what_the_host_builds_matte_paints() {
        let instances = [
            Instance {
                cx: 0.5,
                cy: 0.5,
                w: 0.6,
                h: 0.7,
                query: 0,
            },
            Instance {
                cx: 0.2,
                cy: 0.8,
                w: 0.3,
                h: 0.3,
                query: 2,
            },
            Instance {
                cx: 0.9,
                cy: 0.1,
                w: 0.5,
                h: 0.5,
                query: 1,
            },
            Instance {
                cx: 1.6,
                cy: 0.5,
                w: 0.2,
                h: 0.2,
                query: 1,
            },
            Instance {
                cx: 0.5,
                cy: 0.5,
                w: 0.2,
                h: 0.2,
                query: 9,
            },
        ];
        let mask = (31, 23);
        let masks = ramps(3, mask);
        for frame in [(64, 48), (101, 37), (20, 90)] {
            let want = matte(&instances, &masks, mask, frame);
            let got = painted(
                &upload(&instances, 3, frame),
                &taps(mask, frame),
                &masks,
                mask,
                frame,
            );
            assert_eq!(got, want, "on a {frame:?} frame");
            assert!(
                want.contains(&255) && want.contains(&0),
                "the test decides edges"
            );
        }
    }

    #[test]
    fn the_masks_fit_a_texture_within_the_limit_both_ways() {
        // RF-DETR Seg Large's: 200 queries of 126 by 126, which is 25,200
        // rows a plane down and 15,876 columns a plane across.
        let (width, height) = texture_size(200 * 126 * 126).expect("fits");
        assert_eq!(width * height, 200 * 126 * 126);
        assert!(width <= TEXTURE_LIMIT && height <= TEXTURE_LIMIT);
        assert_eq!(texture_size(12), Some((12, 1)));
        assert_eq!(
            texture_size(8209 * 8209),
            None,
            "a prime square has no shape that fits"
        );
    }

    #[test]
    fn on_the_gpu_the_picture_comes_in_any_format_held_there_and_the_matte_is_rgba() {
        let shape = Runner::<SegmentMask>::shape(r#"{"memory":"gpu"}"#, &["v".to_owned()])
            .expect("a shape");
        assert_eq!(
            shape.find_input("v").unwrap().accepts.pixel_formats,
            ["rgba", "yuv420p", "yuv422p", "yuv444p"]
        );
        let like = shape.find_output(OUT).unwrap().like.as_ref().unwrap();
        assert_eq!(
            (like.port.as_deref(), like.pixel_format.as_deref()),
            (Some("v"), Some("rgba"))
        );
        assert!(shape.pure && shape.one_to_one);
        assert_eq!(
            held::<SegmentMask>(r#"{"memory":"gpu","class":"person"}"#),
            ["v", OUT]
        );
    }

    #[test]
    fn on_the_host_it_has_the_host_builds_shape() {
        for params in ["", r#"{"memory":"host","class":"person"}"#] {
            let shape = Runner::<SegmentMask>::shape(params, &["v".to_owned()]).expect("a shape");
            let host = Runner::<crate::SegmentMask>::shape("", &["v".to_owned()]).unwrap();
            assert_eq!(shape, host, "{params}");
            assert!(held::<SegmentMask>(params).is_empty(), "{params}");
        }
    }

    #[test]
    fn params_default_to_host_memory_and_refuse_what_the_host_build_refuses() {
        let (parsed, _) = read_params::<GpuParams>(PARAMS_SCHEMA, "").expect("the defaults");
        assert_eq!((parsed.conf, parsed.memory), (0.25, Memory::Host));
        let named = read_params::<GpuParams>(PARAMS_SCHEMA, r#"{"class":"bus","memory":"gpu"}"#)
            .unwrap()
            .0;
        assert_eq!(named.settle().unwrap().class, Some(6));
        let unknown = read_params::<GpuParams>(PARAMS_SCHEMA, r#"{"class":"warp core"}"#)
            .unwrap()
            .0;
        assert!(unknown.settle().is_err());
        for bad in [r#"{"conf":1.5}"#, r#"{"memory":"disk"}"#] {
            assert!(
                read_params::<GpuParams>(PARAMS_SCHEMA, bad).is_err(),
                "{bad}"
            );
        }
    }
}
