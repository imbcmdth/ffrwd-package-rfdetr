//! What the rfdetr modules share: the resize into the model's square, the
//! frame-to-tensor preprocessing, the COCO class names at the label indices
//! the graphs use, and the box math that brings a normalized coordinate back
//! onto the frame. Nothing here touches `wasi:nn` or the wit bindings, so
//! every module compiles it in as plain Rust and its tests run on the host.

/// What the graphs normalize with: ImageNet's mean and standard deviation,
/// over red, green and blue in 0..1. The exports do not bake this in - their
/// first node reads the input tensor straight - so the caller applies it.
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

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

/// The pixel format an instance was opened for, fixed for its life.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PixFmt {
    Yuv420p,
    Rgba,
}

impl PixFmt {
    /// The format the host named, or an error naming what it was.
    pub fn parse(named: &str, module: &str) -> Result<PixFmt, String> {
        match named {
            "yuv420p" => Ok(PixFmt::Yuv420p),
            "rgba" => Ok(PixFmt::Rgba),
            other => Err(format!("{module} does not accept pixel format {other}")),
        }
    }
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

/// One frame row as red, green and blue, a channel at a time so each is
/// contiguous.
fn row_to_rgb(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    height: usize,
    y: usize,
    out: &mut [f32],
) {
    let (red, rest) = out.split_at_mut(width);
    let (green, blue) = rest.split_at_mut(width);
    match pix_fmt {
        PixFmt::Rgba => {
            for (x, pixel) in frame[y * width * 4..(y + 1) * width * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
            {
                red[x] = f32::from(pixel[0]);
                green[x] = f32::from(pixel[1]);
                blue[x] = f32::from(pixel[2]);
            }
        }
        PixFmt::Yuv420p => {
            let pixels = width * height;
            let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
            let chroma = cw * ch;
            let luma = &frame[y * width..(y + 1) * width];
            let crow = (y / 2).min(ch - 1) * cw;
            for x in 0..width {
                let l = f32::from(luma[x]);
                let ci = crow + (x / 2).min(cw - 1);
                let u = f32::from(frame[pixels + ci]) - 128.0;
                let v = f32::from(frame[pixels + chroma + ci]) - 128.0;
                // The usual BT.601 inverse, in full range: the frames a module
                // is handed are what the host decoded, not studio-swing video.
                red[x] = l + 1.402 * v;
                green[x] = l - 0.344_136 * u - 0.714_136 * v;
                blue[x] = l + 1.772 * u;
            }
        }
    }
}

/// One frame row resized to the square's columns, channel by channel.
fn resize_rgb_row(rgb: &[f32], columns: &Taps, width: usize, out: &mut [f32]) {
    let count = columns.low.len();
    for channel in 0..3 {
        let source = &rgb[channel * width..(channel + 1) * width];
        let target = &mut out[channel * count..(channel + 1) * count];
        for (((sample, low), high), fraction) in target
            .iter_mut()
            .zip(&columns.low)
            .zip(&columns.high)
            .zip(&columns.fraction)
        {
            let (a, b) = (source[*low], source[*high]);
            *sample = a + (b - a) * fraction;
        }
    }
}

/// The frame stretched into the square the graph takes and laid out as the
/// planar fp32 tensor it expects: red, green and blue in turn, each rescaled
/// to 0..1 and then normalized by `MEAN` and `STD`. The whole frame fills the
/// square - the shape is not kept, which is what the exports were trained and
/// evaluated with - so a box the graph reports as a fraction of the square is
/// the same fraction of the frame.
///
/// The resize is separable, so each frame row is turned into the square's
/// columns once and the two square rows that read it mix the same numbers.
/// Slots go by parity, and a square row mixes frame rows `y` and `y + 1`,
/// which never share one.
pub fn to_input(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    height: usize,
    side: usize,
) -> Vec<u8> {
    let plane = side * side;
    let mut planes = vec![0f32; plane * 3];

    let scale_x = side as f32 / width as f32;
    let scale_y = side as f32 / height as f32;
    let columns = Taps::build(side, width, |sx| (sx as f32 + 0.5) / scale_x - 0.5);
    let rows = Taps::build(side, height, |sy| (sy as f32 + 0.5) / scale_y - 0.5);
    let mut rgb = vec![0f32; width * 3];
    let mut resized = [vec![0f32; side * 3], vec![0f32; side * 3]];
    let mut held: [Option<usize>; 2] = [None, None];

    for sy in 0..side {
        for y in [rows.low[sy], rows.high[sy]] {
            let slot = y % 2;
            if held[slot] != Some(y) {
                row_to_rgb(frame, pix_fmt, width, height, y, &mut rgb);
                resize_rgb_row(&rgb, &columns, width, &mut resized[slot]);
                held[slot] = Some(y);
            }
        }
        let (top_row, bottom_row) = (&resized[rows.low[sy] % 2], &resized[rows.high[sy] % 2]);
        let ty = rows.fraction[sy];

        for channel in 0..3 {
            let top = &top_row[channel * side..(channel + 1) * side];
            let bottom = &bottom_row[channel * side..(channel + 1) * side];
            let at = channel * plane + sy * side;
            let target = &mut planes[at..at + side];
            let (mean, std) = (MEAN[channel], STD[channel]);
            for ((sample, a), b) in target.iter_mut().zip(top).zip(bottom) {
                let value = (a + (b - a) * ty).clamp(0.0, 255.0) / 255.0;
                *sample = (value - mean) / std;
            }
        }
    }

    let mut bytes = vec![0u8; planes.len() * 4];
    let (words, _) = bytes.as_chunks_mut::<4>();
    for (word, value) in words.iter_mut().zip(&planes) {
        *word = value.to_le_bytes();
    }
    bytes
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
pub fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The side the detection export is run at, which is what these tests
    /// build tensors against.
    const SIDE: usize = 704;

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
    fn the_input_tensor_is_the_whole_frame_stretched_and_normalized() {
        // A flat mid-grey frame: every sample is the same normalized value,
        // whatever the frame's shape, since the whole of it fills the square.
        let (width, height) = (32usize, 64usize);
        let frame = vec![128u8; width * height * 4];
        let bytes = to_input(&frame, PixFmt::Rgba, width, height, SIDE);
        assert_eq!(bytes.len(), SIDE * SIDE * 3 * 4);
        let (words, _) = bytes.as_chunks::<4>();
        let value = |channel: usize, x: usize, y: usize| {
            f32::from_le_bytes(words[channel * SIDE * SIDE + y * SIDE + x])
        };
        for channel in 0..3 {
            let want = (128.0 / 255.0 - MEAN[channel]) / STD[channel];
            for (x, y) in [(0, 0), (SIDE / 2, SIDE / 2), (SIDE - 1, SIDE - 1)] {
                assert!(
                    (value(channel, x, y) - want).abs() < 1e-3,
                    "channel {channel} at ({x},{y}): {} wanted {want}",
                    value(channel, x, y)
                );
            }
        }
    }

    #[test]
    fn the_stretch_puts_a_frame_corner_in_the_squares_corner() {
        // Black everywhere but the top-left quarter, which is white: the
        // square's top-left quarter is white and the rest black, whatever the
        // frame's shape - there is no padding to fall on.
        let (width, height) = (64usize, 32usize);
        let mut frame = vec![0u8; width * height * 4];
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                frame[(y * width + x) * 4..(y * width + x) * 4 + 3].fill(255);
            }
        }
        let bytes = to_input(&frame, PixFmt::Rgba, width, height, SIDE);
        let (words, _) = bytes.as_chunks::<4>();
        let red = |x: usize, y: usize| f32::from_le_bytes(words[y * SIDE + x]);
        let white = (1.0 - MEAN[0]) / STD[0];
        let black = -MEAN[0] / STD[0];
        assert!(
            (red(SIDE / 4, SIDE / 4) - white).abs() < 1e-3,
            "the lit quarter"
        );
        assert!(
            (red(3 * SIDE / 4, SIDE / 4) - black).abs() < 1e-3,
            "right of it"
        );
        assert!(
            (red(SIDE / 4, 3 * SIDE / 4) - black).abs() < 1e-3,
            "below it"
        );
    }

    #[test]
    fn a_grey_yuv_frame_and_a_grey_rgba_frame_reach_the_same_tensor() {
        let (width, height) = (16usize, 16usize);
        let side = 32;
        let mut yuv = vec![128u8; width * height + 2 * (width / 2) * (height / 2)];
        yuv[..width * height].fill(200);
        let rgba: Vec<u8> = (0..width * height)
            .flat_map(|_| [200u8, 200, 200, 255])
            .collect();
        let from_yuv = to_input(&yuv, PixFmt::Yuv420p, width, height, side);
        let from_rgba = to_input(&rgba, PixFmt::Rgba, width, height, side);
        let (a, _) = from_yuv.as_chunks::<4>();
        let (b, _) = from_rgba.as_chunks::<4>();
        for (left, right) in a.iter().zip(b) {
            let (l, r) = (f32::from_le_bytes(*left), f32::from_le_bytes(*right));
            assert!((l - r).abs() < 1e-4, "{l} vs {r}");
        }
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
