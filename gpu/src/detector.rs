//! The node every detector's GPU build is: `rfdetr_common::detector`'s node
//! with the graph's input made by the host (`ffrwd:gpu/nn.frame-tensor`)
//! rather than fetched and made here. The rows are decoded on the host and
//! leave as rows, as the host build's do.
//!
//! On `memory = "gpu"` the input is held on the GPU, in any format the
//! sidecar holds there, and the tensor is made on the device; on `host` it
//! is `rgba` in host memory and the host makes the tensor on the CPU.

use std::marker::PhantomData;

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use rfdetr_common::detector::{decode, outputs, row, Found, Head, INPUT_NAME, ROWS_SCHEMA};
use rfdetr_common::nn::{Model, Returned};
use serde::Deserialize;

use crate::nn::{frame_tensor, FrameTensorOptions};
use crate::{same_memory, whole_frame, Held, Memory, GPU_PIXEL_FORMATS};

/// The host build's params, and `memory`.
pub const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"conf":{"type":"number","minimum":0,"maximum":1,"default":0.25},"memory":{"type":"string","enum":["host","gpu"],"default":"host"}},"additionalProperties":false}"#;

#[derive(Deserialize)]
pub struct Params {
    pub conf: f64,
    pub memory: Memory,
}

pub struct Detector<H> {
    v: u32,
    width: usize,
    height: usize,
    conf: f32,
    memory: Memory,
    options: FrameTensorOptions,
    model: Model,
    head: PhantomData<H>,
}

impl<H: Head> Detector<H> {
    fn detect(&mut self, index: u32) -> Result<Vec<Found>, String> {
        let (v, options) = (self.v, &self.options);
        let returned = self
            .model
            .compute(|| frame_tensor(v, index, options).map_err(|e| format!("{}: {e}", H::NAME)))?;
        let shapes: Vec<Vec<u32>> = returned.iter().map(|(_, t)| t.dimensions()).collect();
        let (boxes, logits) = outputs::<H>(&shapes)?;
        decode::<H>(
            &Returned::of(&returned[boxes].1).values,
            &Returned::of(&returned[logits].1).values,
            self.conf,
            self.width,
            self.height,
        )
    }
}

impl<H: Head> Node for Detector<H> {
    const NAME: &'static str = H::NAME;
    const VERSION: &'static str = H::VERSION;
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        let formats = match params.memory {
            Memory::Gpu => GPU_PIXEL_FORMATS,
            Memory::Host => &["rgba"],
        };
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(formats))
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
            memory: params.memory,
            options: whole_frame(H::NAME, H::SIDE),
            model: Model::load(H::NAME, H::NAME, INPUT_NAME)?,
            head: PhantomData,
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        same_memory(H::NAME, self.memory, params.memory)?;
        self.conf = params.conf as f32;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.v) {
            let found = self.detect(frame.index)?;
            for one in &found {
                out.row("boxes", frame.pts, &row::<H>(one))?;
            }
        }
        Ok(())
    }
}

impl<H: Head> Held for Detector<H> {
    const HELD: &'static [&'static str] = &["v"];

    fn memory(params: &Params) -> Memory {
        params.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::held;
    use ffrwd_node::{read_params, Kind, Runner};

    struct One;

    impl Head for One {
        const NAME: &'static str = "one";
        const VERSION: &'static str = "0.0.0";
        const SIDE: usize = 8;
        const CLASSES: usize = 1;

        fn classify(logits: &[f32]) -> (usize, f32) {
            (0, logits[0])
        }

        fn class_name(_: usize) -> String {
            "thing".to_owned()
        }
    }

    #[test]
    fn on_the_gpu_the_picture_comes_in_any_format_the_sidecar_holds_there() {
        let shape = Runner::<Detector<One>>::shape(r#"{"memory":"gpu"}"#, &["v".to_owned()])
            .expect("a shape");
        assert_eq!(shape.clock_input(), Some("v"));
        assert_eq!(
            shape.inputs[0].accepts.pixel_formats,
            ["rgba", "yuv420p", "yuv422p", "yuv444p"]
        );
        assert_eq!(held::<Detector<One>>(r#"{"memory":"gpu"}"#), ["v"]);
    }

    #[test]
    fn on_the_host_it_is_rgba_and_nothing_is_held_on_the_gpu() {
        for params in ["", r#"{"memory":"host"}"#, r#"{"conf":0.5}"#] {
            let shape = Runner::<Detector<One>>::shape(params, &["v".to_owned()]).expect("a shape");
            assert_eq!(shape.inputs[0].accepts.pixel_formats, ["rgba"], "{params}");
            assert!(held::<Detector<One>>(params).is_empty(), "{params}");
        }
    }

    #[test]
    fn the_rows_are_the_host_builds_rows_and_no_picture_leaves() {
        let shape = Runner::<Detector<One>>::shape(r#"{"memory":"gpu"}"#, &["v".to_owned()])
            .expect("a shape");
        assert_eq!(shape.outputs.len(), 1);
        assert_eq!(shape.outputs[0].name, "boxes");
        assert_eq!(shape.outputs[0].kind, Kind::Data);
        assert_eq!(shape.outputs[0].schema.as_deref(), Some(ROWS_SCHEMA));
        assert!(shape.pure);
    }

    #[test]
    fn params_default_to_host_memory_and_the_published_threshold() {
        let (params, _) = read_params::<Params>(PARAMS_SCHEMA, "").expect("the defaults");
        assert_eq!((params.conf, params.memory), (0.25, Memory::Host));
        let (params, _) =
            read_params::<Params>(PARAMS_SCHEMA, r#"{"conf":0.4,"memory":"gpu"}"#).unwrap();
        assert_eq!((params.conf, params.memory), (0.4, Memory::Gpu));
        for bad in [r#"{"memory":"disk"}"#, r#"{"conf":1.5}"#, r#"{"radius":3}"#] {
            assert!(read_params::<Params>(PARAMS_SCHEMA, bad).is_err(), "{bad}");
            assert!(held::<Detector<One>>(bad).is_empty(), "{bad}");
        }
    }

    #[test]
    fn the_host_builds_params_read_the_same_here() {
        // Every call the host build takes, the GPU build takes too.
        let host = rfdetr_common::detector::PARAMS_SCHEMA;
        let (_, from_host) =
            read_params::<rfdetr_common::detector::Params>(host, r#"{"conf":0.3}"#).unwrap();
        let (params, _) = read_params::<Params>(PARAMS_SCHEMA, &from_host.to_string()).unwrap();
        assert_eq!(params.conf, 0.3);
    }
}
