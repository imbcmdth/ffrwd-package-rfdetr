//! Licence plate detection: one row per plate per frame, the class always
//! `plate`, the confidence, and the box in the frame's own pixels. The rows
//! are `detect`'s rows, so `boxes_mask` and `draw_boxes` read them without
//! knowing which detector wrote them.
//!
//! The graph is RF-DETR Medium fine-tuned on one class, run through `wasi:nn`
//! and bound to the name `detect_plates`.

use rfdetr_common::detector::{Detector, Head};

/// Which of a query's logits is the plate.
const PLATE_LOGIT: usize = 0;

pub struct Plates;

impl Head for Plates {
    const NAME: &'static str = "detect_plates";
    const VERSION: &'static str = "0.2.0";
    const SIDE: usize = 576;
    /// The head was trained one class wide by `rfdetr`, which lays the head
    /// out as the classes plus one spare slot, so the export carries two
    /// logits and the plate is the first. There is nothing to take an argmax
    /// over: the spare slot was never trained toward anything.
    const CLASSES: usize = 2;

    fn classify(logits: &[f32]) -> (usize, f32) {
        (0, logits[PLATE_LOGIT])
    }

    fn class_name(_: usize) -> String {
        "plate".to_string()
    }
}

ffrwd_node::export!(Detector<Plates>);

#[cfg(test)]
mod tests {
    use super::*;
    use rfdetr_common::detector::{decode, outputs, row, Found};

    const WIDTH: usize = 1280;
    const HEIGHT: usize = 720;

    /// Three queries: a sure plate, a doubtful one, and a query whose spare
    /// slot is the loud one, which must not count.
    fn queries() -> (Vec<f32>, Vec<f32>) {
        let boxes = vec![
            0.5, 0.6, 0.10, 0.04, // a plate mid-frame, wide and low
            0.2, 0.2, 0.05, 0.02, // a doubtful one
            0.8, 0.8, 0.20, 0.20, // the spare slot's answer
        ];
        let logits = vec![
            3.0, -6.0, // plate 0.95, spare quiet
            -1.0, -6.0, // plate 0.27
            -6.0, 4.0, // plate 0.002, spare loud
        ];
        (boxes, logits)
    }

    #[test]
    fn reads_the_plate_logit_and_leaves_the_spare_slot_alone() {
        let (boxes, logits) = queries();
        let found = decode::<Plates>(&boxes, &logits, 0.25, WIDTH, HEIGHT).unwrap();
        assert_eq!(found.len(), 2);
        assert!(found[0].conf > 0.94 && found[0].conf < 0.96);
        // 0.58 and 0.62 of 720 are 417.6 and 446.4; the box takes the whole pixels they touch.
        assert_eq!(
            (found[0].x, found[0].y, found[0].w, found[0].h),
            (576, 417, 128, 30)
        );
        assert!(found[1].conf > 0.26 && found[1].conf < 0.28);
    }

    #[test]
    fn a_higher_threshold_keeps_only_the_sure_one() {
        let (boxes, logits) = queries();
        let found = decode::<Plates>(&boxes, &logits, 0.5, WIDTH, HEIGHT).unwrap();
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn a_wide_plate_is_not_mistaken_for_a_crowd() {
        // The face module drops a box that is both wide and a quarter of the
        // frame. A plate close to the camera is exactly that, and stays.
        let boxes = vec![0.5, 0.5, 0.8, 0.4];
        let logits = vec![4.0, -6.0];
        let found = decode::<Plates>(&boxes, &logits, 0.25, WIDTH, HEIGHT).unwrap();
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn a_row_says_plate() {
        let plate = Found {
            class: 0,
            conf: 0.87654,
            x: 1,
            y: 2,
            w: 3,
            h: 4,
        };
        assert_eq!(
            serde_json::to_string(&row::<Plates>(&plate)).unwrap(),
            r#"{"class":"plate","conf":0.8765,"x":1,"y":2,"w":3,"h":4}"#
        );
    }

    #[test]
    fn outputs_are_told_apart_by_their_last_dimension() {
        let (boxes, logits) = outputs::<Plates>(&[vec![1, 300, 2], vec![1, 300, 4]]).unwrap();
        assert_eq!((boxes, logits), (1, 0));
        assert!(outputs::<Plates>(&[vec![1, 300, 4], vec![1, 300, 1]]).is_err());
    }
}
