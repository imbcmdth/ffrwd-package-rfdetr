//! A test fixture, not a module the package ships: a box a frame, its
//! coordinates in fractions of a pixel. `draw_boxes` reads whole pixels, so
//! the query that hands it these rows is refused when it is compiled.

use ffrwd_node::{Bound, Init, Input, NoParams, Node, Out, Output, Result, Shape, Tick};
use serde::Serialize;

#[derive(Default, Serialize)]
struct Fraction {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

struct Fractions {
    v: u32,
}

impl Node for Fractions {
    const NAME: &'static str = "fractions";
    const VERSION: &'static str = "0.1.0";
    type Params = NoParams;

    fn shape(_: &NoParams, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock())
            .output(Output::rows("boxes").schema::<Fraction>())
            .pure())
    }

    fn init(_: NoParams, init: &Init) -> Result<Fractions> {
        Ok(Fractions {
            v: init.stream("v")?.id,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.v) {
            let row = Fraction { x: 1.5, y: 2.5, w: 10.25, h: 8.75 };
            out.row("boxes", frame.pts, &row)?;
        }
        Ok(())
    }
}

ffrwd_node::export!(Fractions);
