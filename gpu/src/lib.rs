//! What the GPU builds share: the `ffrwd:gpu@0.1.0` bindings, the `memory`
//! param every GPU build takes, the export that says which of a node's ports
//! that param holds on the GPU, the device and its compute passes, and the
//! node every detector's GPU build is.
//!
//! A GPU build is the same source as the module's host build with its `gpu`
//! feature on, built by the module's `-gpu` crate. The wit is the sidecar's
//! `ffrwd:gpu`, copied into `wit-gpu/` at the workspace's root. Its
//! `wasi:nn` is the one `rfdetr_common::nn` generates, so a tensor the host
//! makes from a frame passes straight to the graph a
//! [`Model`](rfdetr_common::nn::Model) holds.
//!
//! This is a crate of its own, not a feature of `common`, because cargo
//! builds a dependency once for the whole workspace with every feature any
//! member asks for: a `gpu` feature on `common` would reach the host builds.

use std::marker::PhantomData;

use serde::Deserialize;

pub mod detector;
pub mod device;

pub mod bindings {
    // `generate_all`: the world's interfaces come from other packages -
    // ffrwd:gpu, wasi:webgpu and wasi:io - and without it bindgen expects
    // them to have been generated somewhere else. `wasi:nn` has been: it is
    // `rfdetr_common::nn`'s, so the tensor types are one type in both.
    wit_bindgen::generate!({
        path: "../wit-gpu",
        world: "ffrwd:gpu/gpu-nn-module@0.1.0",
        with: {
            "wasi:nn/tensor@0.2.0-rc-2024-10-28": rfdetr_common::nn::wasi::nn::tensor,
            "wasi:nn/errors@0.2.0-rc-2024-10-28": rfdetr_common::nn::wasi::nn::errors,
            "wasi:nn/inference@0.2.0-rc-2024-10-28": rfdetr_common::nn::wasi::nn::inference,
        },
        generate_all,
        pub_export_macro: true,
        default_bindings_module: "rfdetr_gpu::bindings",
    });
}

pub use bindings::ffrwd::gpu::{frames, nn};
pub use bindings::wasi::webgpu::webgpu;

use bindings::exports::ffrwd::gpu::ports::Guest;
use bindings::ffrwd::gpu::types::{FrameMemory, PortMemory};

/// The pixel formats an input held on the GPU takes: every format the
/// sidecar holds there, one texture per plane.
pub const GPU_PIXEL_FORMATS: &[&str] = &["rgba", "yuv420p", "yuv422p", "yuv444p"];

/// Where a GPU build holds its frames: the `memory` param.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Memory {
    Host,
    Gpu,
}

impl Memory {
    /// The param's own spelling.
    pub fn name(self) -> &'static str {
        match self {
            Memory::Host => "host",
            Memory::Gpu => "gpu",
        }
    }
}

/// A GPU build's node: which of its video ports `memory = "gpu"` holds on
/// the device.
pub trait Held: ffrwd_node::Node {
    /// The ports held on the GPU when `memory` is `gpu`.
    const HELD: &'static [&'static str];

    /// The `memory` the params say.
    fn memory(params: &Self::Params) -> Memory;
}

/// What a module exports as `ffrwd:gpu/ports` for node `N`.
pub struct Ports<N>(PhantomData<N>);

/// The ports `params` hold on the GPU for node `N`: none on `host`, and
/// none for params that do not read, which `shape` refuses by name.
pub fn held<N: Held>(params: &str) -> Vec<&'static str> {
    match ffrwd_node::read_params::<N::Params>(N::PARAMS_SCHEMA, params) {
        Ok((params, _)) if N::memory(&params) == Memory::Gpu => N::HELD.to_vec(),
        _ => Vec::new(),
    }
}

impl<N: Held> Guest for Ports<N> {
    fn memory(params: String) -> Vec<PortMemory> {
        held::<N>(&params)
            .into_iter()
            .map(|port| PortMemory {
                port: port.to_owned(),
                memory: FrameMemory::Gpu,
            })
            .collect()
    }
}

/// Exports `ffrwd:gpu/ports` for `$node`, a type implementing [`Held`],
/// beside `ffrwd_node::export!`. Once per module, at its crate root.
#[macro_export]
#[rustfmt::skip]
macro_rules! export {
    ($node:ty) => {
        #[cfg(target_arch = "wasm32")]
        const _: () = {
            type GpuPorts = $crate::Ports<$node>;
            $crate::bindings::export!(GpuPorts with_types_in $crate::bindings);
        };
    };
}

/// The options that make a whole frame into an RF-DETR graph's input: the
/// frame stretched to the `side` square the graph is run at, resampled with
/// Pillow's antialiased bilinear filter, scaled to 0 to 1 and normalized
/// with ImageNet's mean and deviation, planar RGB. This is what the host
/// build's `ffrwd_frame::tensor(.., Filter::Bilinear, IMAGENET)` makes.
pub fn whole_frame(model: &str, side: usize) -> nn::FrameTensorOptions {
    let norm = ffrwd_frame::IMAGENET;
    nn::FrameTensorOptions {
        model: model.to_owned(),
        width: side as u32,
        height: side as u32,
        layout: nn::TensorLayout::Nchw,
        order: nn::ChannelOrder::Rgb,
        fit: nn::Fit::Stretch,
        pad: 0.0,
        scale: 1.0,
        mean: (norm.mean[0], norm.mean[1], norm.mean[2]),
        std: (norm.std[0], norm.std[1], norm.std[2]),
        filter: nn::ResizeFilter::Bilinear,
        antialias: true,
    }
}

/// Refuses a change of `memory` while the instance runs: the ports were
/// placed for the one it was opened with.
pub fn same_memory(module: &str, opened: Memory, now: Memory) -> Result<(), String> {
    if opened == now {
        Ok(())
    } else {
        Err(format!(
            "{module} cannot change memory while it runs: its ports were placed for {}",
            opened.name()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_whole_frame_is_stretched_bilinear_antialiased_and_imagenet_normalized() {
        let options = whole_frame("detect_faces", 576);
        assert_eq!(options.model, "detect_faces");
        assert_eq!((options.width, options.height), (576, 576));
        assert!(matches!(options.layout, nn::TensorLayout::Nchw));
        assert!(matches!(options.order, nn::ChannelOrder::Rgb));
        assert!(matches!(options.fit, nn::Fit::Stretch));
        assert!(matches!(options.filter, nn::ResizeFilter::Bilinear));
        assert!(options.antialias);
        assert_eq!((options.pad, options.scale), (0.0, 1.0));
        assert_eq!(options.mean, (0.485, 0.456, 0.406));
        assert_eq!(options.std, (0.229, 0.224, 0.225));
    }

    #[test]
    fn memory_reads_as_its_two_words() {
        let gpu: Memory = serde_json::from_str(r#""gpu""#).unwrap();
        let host: Memory = serde_json::from_str(r#""host""#).unwrap();
        assert_eq!((gpu, host), (Memory::Gpu, Memory::Host));
        assert!(serde_json::from_str::<Memory>(r#""disk""#).is_err());
    }

    #[test]
    fn memory_cannot_change_while_it_runs() {
        assert!(same_memory("m", Memory::Gpu, Memory::Gpu).is_ok());
        let error = same_memory("m", Memory::Gpu, Memory::Host).expect_err("moved");
        assert!(error.starts_with("m cannot change memory"), "{error}");
    }
}
