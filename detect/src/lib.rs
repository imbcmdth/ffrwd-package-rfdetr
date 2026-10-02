//! Object detection: one row per object per frame, the class as COCO label
//! text, the confidence, and the box in the frame's own pixels.
//!
//! The graph is RF-DETR Large's export, run through `wasi:nn` and bound to
//! the name `detect`. Each query's class is whichever of its logits is
//! largest.

use rfdetr_common::detector::{Detector, Head};

pub struct Coco;

impl Head for Coco {
    const NAME: &'static str = "detect";
    const VERSION: &'static str = "0.2.0";
    const SIDE: usize = 704;
    /// COCO's 91-slot category numbering.
    const CLASSES: usize = 91;

    fn classify(logits: &[f32]) -> (usize, f32) {
        let mut class = 0;
        let mut best = f32::NEG_INFINITY;
        for (index, logit) in logits.iter().enumerate() {
            if *logit > best {
                best = *logit;
                class = index;
            }
        }
        (class, best)
    }

    fn class_name(class: usize) -> String {
        rfdetr_common::class_name(class)
    }
}

ffrwd_node::export!(Detector<Coco>);

#[cfg(test)]
mod tests {
    use super::*;
    use rfdetr_common::detector::{decode, Found};

    /// What the real export returned for one 720x576 frame, captured from a
    /// run of the pinned graph: the five surest queries, each a box as a
    /// fraction of the picture beside the class logit that won it. Three
    /// people, a chair, a screen. The fixture is the decode's contract, not
    /// the model's.
    const CAPTURED: [(f32, f32, f32, f32, usize, f32); 5] = [
        (0.264, 0.548, 0.307, 0.898, 1, 2.976),
        (0.621, 0.857, 0.270, 0.279, 62, 1.966),
        (0.523, 0.824, 0.234, 0.347, 1, 1.629),
        (0.092, 0.901, 0.147, 0.196, 1, 0.364),
        (0.467, 0.414, 0.905, 0.644, 72, -0.253),
    ];

    /// The captured queries as the two tensors the graph hands back, plus the
    /// quiet queries a DETR head always returns beside them.
    fn captured(padding: usize) -> (Vec<f32>, Vec<f32>) {
        let mut boxes = Vec::new();
        let mut logits = Vec::new();
        for (cx, cy, w, h, class, logit) in CAPTURED {
            boxes.extend([cx, cy, w, h]);
            let mut scores = vec![-10.0f32; Coco::CLASSES];
            scores[class] = logit;
            logits.extend(scores);
        }
        for _ in 0..padding {
            boxes.extend([0.5, 0.5, 0.1, 0.1]);
            logits.extend(vec![-10.0f32; Coco::CLASSES]);
        }
        (boxes, logits)
    }

    fn decoded(conf: f32) -> Vec<Found> {
        let (boxes, logits) = captured(0);
        decode::<Coco>(&boxes, &logits, conf, 720, 576).expect("the tensors agree")
    }

    #[test]
    fn the_captured_tensors_decode_to_one_row_per_object() {
        let found = decoded(0.25);
        assert_eq!(found.len(), 5, "every captured query clears 0.25");
        assert_eq!(found[0].class, 1, "the surest object is a person");
        assert_eq!(found[1].class, 62, "and the second is the chair");
        for object in &found {
            assert!(object.x + object.w <= 720, "boxes stay on the picture");
            assert!(object.y + object.h <= 576);
        }
        // The centre is 0.264 of 720 and the width 0.307, so the left edge is
        // (0.264 - 0.1535) * 720.
        assert_eq!(found[0].x, 79);
        assert_eq!(found[0].y, 57);
    }

    #[test]
    fn the_threshold_drops_what_scores_under_it() {
        let found = decoded(0.5);
        assert_eq!(found.len(), 4, "the screen, at 0.437, is the one that goes");
        assert!(found.iter().all(|object| object.conf >= 0.5));
    }

    #[test]
    fn the_quiet_queries_a_detr_head_returns_never_decode() {
        // The head always returns its full set; the ones that found nothing
        // score far under any threshold.
        let (boxes, logits) = captured(295);
        let found = decode::<Coco>(&boxes, &logits, 0.25, 720, 576).expect("the tensors agree");
        assert_eq!(found.len(), 5);
    }

    #[test]
    fn a_box_naming_no_pixel_of_the_picture_is_dropped() {
        let boxes = [1.6, 0.5, 0.2, 0.2];
        let mut logits = vec![-10.0f32; Coco::CLASSES];
        logits[1] = 5.0;
        assert!(decode::<Coco>(&boxes, &logits, 0.25, 720, 576)
            .expect("the tensors agree")
            .is_empty());
    }

    #[test]
    fn a_class_is_named_as_coco_text() {
        assert_eq!(Coco::class_name(1), "person");
        assert_eq!(Coco::class_name(72), "tv");
    }
}
