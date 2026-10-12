//! The device a GPU build paints on, and the compute passes it runs there:
//! the sidecar's one device, as the module's `wasi:webgpu` answers it.

use crate::webgpu::{
    get_gpu, GpuBindGroupDescriptor, GpuBindGroupEntry, GpuBindGroupLayout,
    GpuBindGroupLayoutDescriptor, GpuBindGroupLayoutEntry, GpuBindingResource, GpuBuffer,
    GpuBufferBinding, GpuBufferBindingLayout, GpuBufferBindingType, GpuBufferDescriptor,
    GpuBufferUsage, GpuComputePipeline, GpuComputePipelineDescriptor, GpuDevice, GpuExtent3D,
    GpuLayoutMode, GpuPipelineLayoutDescriptor, GpuProgrammableStage, GpuShaderModuleDescriptor,
    GpuShaderStage, GpuStorageTextureAccess, GpuStorageTextureBindingLayout, GpuTexture,
    GpuTextureBindingLayout, GpuTextureDescriptor, GpuTextureFormat, GpuTextureSampleType,
    GpuTextureUsage, GpuTextureView, GpuTextureViewDimension,
};

/// The side of a pass's square workgroup, which every pass's WGSL declares
/// as `@workgroup_size(8, 8)`.
pub const WORKGROUP: u32 = 8;

/// The sidecar's device.
pub fn open(module: &str) -> Result<GpuDevice, String> {
    get_gpu()
        .request_adapter(None)
        .ok_or_else(|| format!("{module}: no GPU adapter; memory=host runs without one"))?
        .request_device(None)
        .map_err(|_| format!("{module}: the adapter gave no device"))
}

/// A frame a pass writes and the host is handed: `rgba8unorm`, the size of
/// the port it leaves on.
pub fn target(device: &GpuDevice, width: u32, height: u32) -> GpuTexture {
    device.create_texture(&GpuTextureDescriptor {
        size: GpuExtent3D {
            width,
            height: Some(height),
            depth_or_array_layers: None,
        },
        mip_level_count: None,
        sample_count: None,
        dimension: None,
        format: GpuTextureFormat::Rgba8unorm,
        usage: GpuTextureUsage::storage_binding()
            | GpuTextureUsage::texture_binding()
            | GpuTextureUsage::copy_src(),
        view_formats: None,
        label: None,
    })
}

/// Words as the little-endian bytes a buffer holds.
pub fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// What one binding of a pass is, in binding order from 0.
#[derive(Clone, Copy, Debug)]
pub enum Slot {
    /// A texture read with `textureLoad`: `texture_2d<f32>`.
    Texture,
    /// `var<storage, read>`.
    Storage,
    /// `var<uniform>`.
    Uniform,
    /// The frame written: `texture_storage_2d<rgba8unorm, write>`.
    Target,
}

/// What a binding is bound to, in the pass's slot order.
pub enum Bound<'a> {
    View(&'a GpuTextureView),
    Buffer(&'a GpuBuffer),
}

/// One compute pipeline and the layout of what it binds.
pub struct Pass {
    layout: GpuBindGroupLayout,
    pipeline: GpuComputePipeline,
}

impl Pass {
    /// Entry point `entry` of `code`, binding `slots` in group 0.
    pub fn new(device: &GpuDevice, code: &str, entry: &str, slots: &[Slot]) -> Pass {
        let visibility = GpuShaderStage::compute();
        let entries = slots
            .iter()
            .enumerate()
            .map(|(binding, slot)| {
                let mut entry = GpuBindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility,
                    buffer: None,
                    sampler: None,
                    texture: None,
                    storage_texture: None,
                };
                let buffer = |type_| {
                    Some(GpuBufferBindingLayout {
                        type_: Some(type_),
                        has_dynamic_offset: None,
                        min_binding_size: None,
                    })
                };
                match slot {
                    Slot::Texture => {
                        entry.texture = Some(GpuTextureBindingLayout {
                            sample_type: Some(GpuTextureSampleType::UnfilterableFloat),
                            view_dimension: Some(GpuTextureViewDimension::D2),
                            multisampled: Some(false),
                        })
                    }
                    Slot::Storage => entry.buffer = buffer(GpuBufferBindingType::ReadOnlyStorage),
                    Slot::Uniform => entry.buffer = buffer(GpuBufferBindingType::Uniform),
                    Slot::Target => {
                        entry.storage_texture = Some(GpuStorageTextureBindingLayout {
                            access: Some(GpuStorageTextureAccess::WriteOnly),
                            format: GpuTextureFormat::Rgba8unorm,
                            view_dimension: Some(GpuTextureViewDimension::D2),
                        })
                    }
                }
                entry
            })
            .collect();
        let layout = device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor {
            entries,
            label: None,
        });
        let pipeline_layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor {
            bind_group_layouts: vec![Some(&layout)],
            label: None,
        });
        let module = device.create_shader_module(&GpuShaderModuleDescriptor {
            code: code.to_owned(),
            compilation_hints: None,
            label: Some(entry.to_owned()),
        });
        let pipeline = device.create_compute_pipeline(GpuComputePipelineDescriptor {
            compute: GpuProgrammableStage {
                module: &module,
                entry_point: Some(entry.to_owned()),
                constants: None,
            },
            layout: GpuLayoutMode::Specific(&pipeline_layout),
            label: None,
        });
        Pass { layout, pipeline }
    }

    /// One dispatch over a `width` by `height` frame, a thread a pixel,
    /// with `bound` in slot order, submitted to the device's queue.
    pub fn run(
        &self,
        device: &GpuDevice,
        bound: &[Bound],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let entries = bound
            .iter()
            .enumerate()
            .map(|(binding, bound)| GpuBindGroupEntry {
                binding: binding as u32,
                resource: match bound {
                    Bound::View(view) => GpuBindingResource::GpuTextureView(view),
                    Bound::Buffer(buffer) => {
                        GpuBindingResource::GpuBufferBinding(GpuBufferBinding {
                            buffer,
                            offset: None,
                            size: None,
                        })
                    }
                },
            })
            .collect();
        let group = device.create_bind_group(&GpuBindGroupDescriptor {
            layout: &self.layout,
            entries,
            label: None,
        });
        let encoder = device.create_command_encoder(None);
        let pass = encoder.begin_compute_pass(None);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, Some(&group), None, None, None)
            .map_err(|_| "binding a pass's resources".to_owned())?;
        pass.dispatch_workgroups(
            width.div_ceil(WORKGROUP),
            Some(height.div_ceil(WORKGROUP)),
            None,
        );
        pass.end();
        device.queue().submit(&[&encoder.finish(None)]);
        Ok(())
    }
}

/// A buffer rewritten every frame, grown when a frame needs more of it.
pub struct Upload {
    buffer: GpuBuffer,
    capacity: u64,
    uniform: bool,
}

/// The least a buffer is made: a binding is never empty, and a uniform's
/// struct is a multiple of 16 bytes.
const LEAST: u64 = 16;

impl Upload {
    /// A storage buffer.
    pub fn storage(device: &GpuDevice) -> Upload {
        Upload::new(device, false)
    }

    /// A uniform buffer.
    pub fn uniform(device: &GpuDevice) -> Upload {
        Upload::new(device, true)
    }

    fn new(device: &GpuDevice, uniform: bool) -> Upload {
        Upload {
            buffer: Upload::make(device, LEAST, uniform),
            capacity: LEAST,
            uniform,
        }
    }

    fn make(device: &GpuDevice, size: u64, uniform: bool) -> GpuBuffer {
        let usage = if uniform {
            GpuBufferUsage::uniform()
        } else {
            GpuBufferUsage::storage()
        };
        device.create_buffer(&GpuBufferDescriptor {
            size,
            usage: usage | GpuBufferUsage::copy_dst(),
            mapped_at_creation: None,
            label: None,
        })
    }

    /// `bytes` written from the start, the buffer grown first when they do
    /// not fit, and the buffer they are in.
    pub fn write(&mut self, device: &GpuDevice, bytes: &[u8]) -> Result<&GpuBuffer, String> {
        let wanted = (bytes.len() as u64).next_multiple_of(LEAST).max(LEAST);
        if wanted > self.capacity {
            let grown = wanted.max(self.capacity * 2);
            self.buffer = Upload::make(device, grown, self.uniform);
            self.capacity = grown;
        }
        if !bytes.is_empty() {
            // A write is a multiple of four bytes long.
            let mut padded = bytes.to_vec();
            padded.resize(bytes.len().next_multiple_of(4), 0);
            device
                .queue()
                .write_buffer_with_copy(&self.buffer, 0, &padded, None, None)
                .map_err(|e| format!("writing a buffer: {}", e.message))?;
        }
        Ok(&self.buffer)
    }

    /// The buffer, for one written once and never again.
    pub fn into_buffer(self) -> GpuBuffer {
        self.buffer
    }
}
