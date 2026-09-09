//! What the rfdetr modules share: the COCO class names at the label indices
//! the graphs use, the box math that brings a normalized coordinate back onto
//! the frame, and the tap map a separable resize reads. Nothing here touches
//! `wasi:nn` or the wit bindings, so every module compiles it in as plain
//! Rust and its tests run on the host.
//!
//! Turning a frame into a tensor is not here: a module asks the host for
//! `rgba` and `ffrwd-frame` crops, resizes and normalizes it.

/// The classes the graphs were trained on, at the label index each graph
/// reports: COCO's 91-slot category numbering, not the 80 contiguous names.
/// An unused slot is empty.
pub const COCO: [&str; 91] = [
    "",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "",
    "backpack",
    "umbrella",
    "",
    "",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "",
    "dining table",
    "",
    "",
    "toilet",
    "",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// The name of a class by the graph's own label index. A slot the numbering
/// leaves empty, or one past the table, is named by number rather than
/// guessed at.
pub fn class_name(index: usize) -> String {
    match COCO.get(index) {
        Some(name) if !name.is_empty() => (*name).to_string(),
        _ => index.to_string(),
    }
}

/// The graph's label index for a class name, or None for a name it was not
/// trained on.
pub fn class_index(name: &str) -> Option<usize> {
    if name.is_empty() {
        return None;
    }
    COCO.iter().position(|known| *known == name)
}

/// A box the graph reported - centre, width and height, each a fraction of
/// the picture - brought onto the frame and clipped to it: `(x0, y0, x1, y1)`
/// in frame pixels, exclusive on the right and bottom. Empty - `x0 == x1` or
/// `y0 == y1` - when the box names no pixel of the picture.
pub fn frame_box(
    normalized: (f32, f32, f32, f32),
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    let (cx, cy, w, h) = normalized;
    let clip = |value: f32, limit: usize| value.clamp(0.0, limit as f32) as usize;
    let x0 = clip(((cx - w / 2.0) * width as f32).floor(), width);
    let x1 = clip(((cx + w / 2.0) * width as f32).ceil(), width);
    let y0 = clip(((cy - h / 2.0) * height as f32).floor(), height);
    let y1 = clip(((cy + h / 2.0) * height as f32).ceil(), height);
    (x0, y0, x1.max(x0), y1.max(y0))
}

/// The logistic function, which turns one of the graph's class logits into
/// the confidence a row carries.
pub fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

/// Where each of `count` output steps reads from along one axis: the two
/// samples it falls between, and how far along it sits. Every row of a resize
/// reads the same columns, so a column map is built once rather than per row.
pub struct Taps {
    pub low: Vec<usize>,
    pub high: Vec<usize>,
    pub fraction: Vec<f32>,
}

impl Taps {
    /// `at` gives the source coordinate of each step, on a source `extent`
    /// samples long.
    pub fn build(count: usize, extent: usize, at: impl Fn(usize) -> f32) -> Taps {
        let mut map = Taps {
            low: Vec::with_capacity(count),
            high: Vec::with_capacity(count),
            fraction: Vec::with_capacity(count),
        };
        for step in 0..count {
            let f = at(step).clamp(0.0, (extent - 1) as f32);
            let base = f.floor() as usize;
            map.low.push(base);
            map.high.push((base + 1).min(extent - 1));
            map.fraction.push(f - base as f32);
        }
        map
    }
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
pub fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_class_is_named_by_the_graphs_own_label_index() {
        assert_eq!(class_name(1), "person");
        assert_eq!(class_name(6), "bus");
        assert_eq!(class_name(62), "chair");
        assert_eq!(class_name(72), "tv");
        assert_eq!(class_name(90), "toothbrush");
        assert_eq!(
            class_name(0),
            "0",
            "a slot the numbering leaves empty is numbered, not guessed at"
        );
        assert_eq!(class_name(12), "12", "and so is one inside the range");
        assert_eq!(class_name(91), "91", "and so is one past the table");
    }

    #[test]
    fn a_class_name_finds_the_label_index_it_names() {
        assert_eq!(class_index("person"), Some(1));
        assert_eq!(class_index("skateboard"), Some(41));
        assert_eq!(class_index("toothbrush"), Some(90));
        assert_eq!(class_index("warp core"), None);
        assert_eq!(class_index(""), None, "an empty slot is not a class");
    }

    #[test]
    fn the_table_holds_the_eighty_coco_names_over_ninety_one_slots() {
        assert_eq!(COCO.len(), 91);
        assert_eq!(COCO.iter().filter(|name| !name.is_empty()).count(), 80);
    }

    #[test]
    fn a_normalized_box_lands_on_the_frame_in_its_own_pixels() {
        // Centred, a quarter of the picture each way, on a 720x576 frame.
        let (x0, y0, x1, y1) = frame_box((0.5, 0.5, 0.25, 0.25), 720, 576);
        assert_eq!((x0, x1), (270, 450));
        assert_eq!((y0, y1), (216, 360));
    }

    #[test]
    fn a_box_running_off_the_picture_is_clipped_to_it() {
        let (x0, y0, x1, y1) = frame_box((0.1, 0.1, 0.5, 0.5), 720, 576);
        assert_eq!((x0, y0), (0, 0), "the corner outside the frame clips to it");
        assert_eq!((x1, y1), (252, 202));

        let (x0, _, x1, _) = frame_box((0.95, 0.5, 0.5, 0.2), 720, 576);
        assert_eq!(x1, 720, "and so does the far edge");
        assert_eq!(x0, 504);
    }

    #[test]
    fn a_box_entirely_off_the_picture_is_empty() {
        let (x0, y0, x1, y1) = frame_box((1.4, 0.5, 0.2, 0.2), 720, 576);
        assert_eq!(x0, x1, "nothing of it is on the frame");
        assert!(y0 < y1);
    }

    #[test]
    fn the_confidence_is_the_logit_through_the_logistic_curve() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(-10.0) < 0.0001);
        assert!(sigmoid(10.0) > 0.9999);
    }

    #[test]
    fn tensor_floats_come_back_out_of_their_bytes() {
        let bytes: Vec<u8> = [1.5f32, -2.25, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(le_f32s(&bytes), vec![1.5, -2.25, 0.0]);
    }
}
