//! The graph a model node runs, through `wasi:nn`. The module never opens a
//! file: the host binds the graph to a name with `-nn <name>=<path>`, and the
//! node asks for that name and nothing else.

// `generate_all`: the world's interfaces are wasi:nn's, a package of its own,
// and without it bindgen expects them to have been generated somewhere else.
wit_bindgen::generate!({
    path: "wit-world",
    world: "ffrwd:rfdetr/nn",
    generate_all,
});

use wasi::nn::errors::{Error, ErrorCode};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

use crate::le_f32s;

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// One loaded graph, for the life of the instance.
pub struct Model {
    /// The module, which every message names.
    module: &'static str,
    /// What the graph calls its input, settled by the first call that works.
    input: &'static str,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

/// One tensor the graph returned.
pub struct Returned {
    pub dimensions: Vec<u32>,
    pub values: Vec<f32>,
}

impl Model {
    /// The graph the host bound to `name`, with its execution context built
    /// once: the first frame is what a provider picks its kernels on, and
    /// every frame after it reuses them.
    pub fn load(module: &'static str, name: &str, input: &'static str) -> Result<Model, String> {
        let graph = load_by_name(name)
            .map_err(|e| failed(module, &format!("load-by-name({name:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed(module, "init-execution-context", &e))?;
        Ok(Model {
            module,
            input,
            context,
            _graph: graph,
        })
    }

    /// One fp32 tensor of `dimensions` through the graph, and every tensor
    /// it returned, in its own order.
    pub fn run(&mut self, dimensions: &[u32], data: &[u8]) -> Result<Vec<Returned>, String> {
        let returned = match self.compute(self.input, dimensions, data) {
            Ok(returned) => returned,
            // An export whose input is not called what this one calls it. The
            // host takes a position where it takes a name, so the retry names
            // none, and the name that worked is kept for every call after it.
            Err(_) if self.input != INPUT_INDEX => {
                self.input = INPUT_INDEX;
                self.compute(INPUT_INDEX, dimensions, data)
                    .map_err(|e| failed(self.module, "compute", &e))?
            }
            Err(e) => return Err(failed(self.module, "compute", &e)),
        };
        Ok(returned
            .into_iter()
            .map(|(_, tensor)| Returned {
                dimensions: tensor.dimensions(),
                values: le_f32s(&tensor.data()),
            })
            .collect())
    }

    fn compute(
        &self,
        name: &str,
        dimensions: &[u32],
        data: &[u8],
    ) -> Result<Vec<(String, Tensor)>, Error> {
        let tensor = Tensor::new(dimensions, TensorType::Fp32, data);
        self.context.compute(vec![(name.to_string(), tensor)])
    }
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(module: &str, what: &str, error: &Error) -> String {
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("{module}: {what}: {code} ({})", error.data())
}
