//! Face detection: one row per face per frame, the class always `face`, the
//! confidence, and the box in the frame's own pixels. The rows are `detect`'s
//! rows, so `boxes_mask` and `draw_boxes` read them without knowing which
//! detector wrote them.
//!
//! The graph is RF-DETR Medium fine-tuned on one class, run through `wasi:nn`
//! and bound to the name `detect_faces`. The head is one class wide, so there
//! is no background slot and nothing to take an argmax over.

use rfdetr_common::detector::{Detector, Head};

/// How many times its own height a box has to be to count as wide.
const WIDE_ASPECT: f32 = 1.3;

/// The share of the frame a wide box has to cover to count as large.
const WIDE_AREA: f32 = 0.25;

pub struct Faces;

impl Head for Faces {
    const NAME: &'static str = "detect_faces";
    const VERSION: &'static str = "0.2.0";
    const SIDE: usize = 576;
    const CLASSES: usize = 1;

    fn classify(logits: &[f32]) -> (usize, f32) {
        (0, logits[0])
    }

    fn class_name(_: usize) -> String {
        "face".to_string()
    }

    /// The head's whole-frame answer arrives on a crowd at a middling
    /// confidence. A face is taller than it is wide, or near square when the
    /// head is tilted, so a box both this wide and this large is not one: a
    /// wide box that is small is a face half behind something, and a large
    /// box that is tall is a face close to the camera, and both stay.
    fn keeps(w: usize, h: usize, width: usize, height: usize) -> bool {
        let wide = w as f32 > h as f32 * WIDE_ASPECT;
        let large = (w * h) as f32 > (width * height) as f32 * WIDE_AREA;
        !(wide && large)
    }
}

ffrwd_node::export!(Detector<Faces>);

#[cfg(test)]
mod tests {
    use super::*;
    use rfdetr_common::detector::{decode, outputs, row, Found};

    /// The frame the capture was taken on: WIDER FACE validation image
    /// 2--Demonstration/2_Demonstration_Protesters_2_738.jpg, ten annotated
    /// faces, five of them over 96 px.
    const WIDTH: usize = 1024;
    const HEIGHT: usize = 1428;

    /// What the real export returned for that frame, captured from a run of
    /// the pinned graph: the five surest queries, each a box as a fraction of
    /// the picture beside the one class logit the head carries. The fixture
    /// is the decode's contract, not the model's.
    const CAPTURED: [(f32, f32, f32, f32, f32); 5] = [
        (0.951608, 0.792185, 0.096596, 0.131434, 1.986528),
        (0.055047, 0.925899, 0.110031, 0.147960, 1.979382),
        (0.768078, 0.763318, 0.119112, 0.115945, 1.794836),
        (0.297883, 0.663443, 0.120973, 0.103971, 1.708497),
        (0.327975, 0.066133, 0.058301, 0.058259, 1.286335),
    ];

    /// The captured queries as the two tensors the graph hands back, plus the
    /// quiet queries a DETR head always returns beside them.
    fn captured(padding: usize) -> (Vec<f32>, Vec<f32>) {
        let mut boxes = Vec::new();
        let mut logits = Vec::new();
        for (cx, cy, w, h, logit) in CAPTURED {
            boxes.extend([cx, cy, w, h]);
            logits.push(logit);
        }
        for _ in 0..padding {
            boxes.extend([0.5, 0.5, 0.1, 0.1]);
            logits.push(-10.0);
        }
        (boxes, logits)
    }

    fn decoded(conf: f32) -> Vec<Found> {
        let (boxes, logits) = captured(0);
        decode::<Faces>(&boxes, &logits, conf, WIDTH, HEIGHT).expect("the tensors agree")
    }

    #[test]
    fn the_captured_tensors_decode_to_one_row_per_face() {
        let found = decoded(0.25);
        assert_eq!(found.len(), 5, "every captured query clears 0.25");
        for face in &found {
            assert!(face.x + face.w <= WIDTH as u32, "boxes stay on the picture");
            assert!(face.y + face.h <= HEIGHT as u32);
            assert!(face.conf > 0.78, "the five surest are all well clear");
        }
        // The centre is 0.951608 of 1024 and the width 0.096596, so the left
        // edge is (0.951608 - 0.048298) * 1024 and the right runs off the
        // frame and clips to it.
        assert_eq!((found[0].x, found[0].w), (924, 100));
        // And 0.792185 of 1428 with a height of 0.131434.
        assert_eq!((found[0].y, found[0].h), (1037, 189));
    }

    #[test]
    fn the_threshold_drops_what_scores_under_it() {
        let found = decoded(0.8);
        assert_eq!(found.len(), 4, "the one at 0.7835 is what goes");
        assert!(found.iter().all(|face| face.conf >= 0.8));
        assert!(decoded(0.9).is_empty(), "and past the surest, nothing left");
    }

    #[test]
    fn the_quiet_queries_a_detr_head_returns_never_decode() {
        // The head always returns its full set, 300 queries, and the ones
        // that found nothing score far under any threshold.
        let (boxes, logits) = captured(295);
        assert_eq!(logits.len(), 300);
        let found =
            decode::<Faces>(&boxes, &logits, 0.25, WIDTH, HEIGHT).expect("the tensors agree");
        assert_eq!(found.len(), 5);
    }

    #[test]
    fn the_wide_whole_frame_box_goes_and_the_faces_beside_it_stay() {
        // Wide and most of the picture: the head's whole-frame answer. Beside
        // it a close-up, as large but taller than wide, and a face half behind
        // something, as wide but small: a row each.
        let boxes = [
            0.5, 0.5, 0.95, 0.4, //
            0.5, 0.5, 0.5, 0.6, //
            0.5, 0.5, 0.3, 0.1,
        ];
        let logits = [5.0, 5.0, 5.0];
        let found =
            decode::<Faces>(&boxes, &logits, 0.25, WIDTH, HEIGHT).expect("the tensors agree");
        assert_eq!(found.len(), 2, "only the wide whole-frame box goes");
        assert_eq!((found[0].w, found[0].h), (512, 858), "the close-up stays");
        assert_eq!(
            (found[1].w, found[1].h),
            (308, 144),
            "and so does the small"
        );
    }

    #[test]
    fn a_box_naming_no_pixel_of_the_picture_is_dropped() {
        let boxes = [1.6, 0.5, 0.2, 0.2];
        let logits = [5.0];
        assert!(decode::<Faces>(&boxes, &logits, 0.25, WIDTH, HEIGHT)
            .expect("the tensors agree")
            .is_empty());
    }

    #[test]
    fn tensors_that_disagree_on_how_many_queries_are_refused_by_name() {
        let boxes = [0.5, 0.5, 0.2, 0.2, 0.4, 0.4, 0.2, 0.2];
        let logits = [1.0];
        let error = decode::<Faces>(&boxes, &logits, 0.25, WIDTH, HEIGHT)
            .expect_err("two boxes, one query");
        assert!(error.starts_with("detect_faces: "), "{error}");
    }

    #[test]
    fn a_row_spells_the_class_face_and_rounds_the_confidence() {
        let face = Found {
            class: 0,
            conf: 0.87941,
            x: 924,
            y: 1037,
            w: 100,
            h: 189,
        };
        assert_eq!(
            serde_json::to_string(&row::<Faces>(&face)).unwrap(),
            r#"{"class":"face","conf":0.8794,"x":924,"y":1037,"w":100,"h":189}"#
        );
    }

    #[test]
    fn the_two_returned_tensors_are_told_apart_by_shape_in_either_order() {
        assert_eq!(
            outputs::<Faces>(&[vec![1, 300, 4], vec![1, 300, 1]]).expect("both found"),
            (0, 1)
        );
        assert_eq!(
            outputs::<Faces>(&[vec![1, 300, 1], vec![1, 300, 4]]).expect("both found"),
            (1, 0)
        );
        let error =
            outputs::<Faces>(&[vec![1, 300, 4], vec![1, 300, 91]]).expect_err("a COCO head");
        assert!(error.starts_with("detect_faces: "), "{error}");
    }
}
