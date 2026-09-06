# ffrwd/rfdetr

RF-DETR detection and instance segmentatio. Same API as YOLO26: `detect` produces rows,
`segment_mask` produces a matte, the utilities turn rows into mattes or
drawn overlays, and everything downstream is native ffmpeg.

## Model exports

- `detect(v, conf DEFAULT 0.25)` returns
  `STRUCT(v video_stream, boxes STRUCT(class text, conf number, x number, y number, w number, h number)[])` -
  the picture untouched, one row per object per frame, boxes in the
  frame's own pixels, classes as COCO label text.
- `segment_mask(v, class DEFAULT NULL, conf DEFAULT 0.25)` returns the
  found instances as one grayscale matte, optionally narrowed to one
  class name.

A DETR head returns no duplicates, so there is no NMS anywhere; the
module resizes each frame to the model's square, normalizes it, and
reads the boxes and per-query masks back. Narrow `detect`'s rows at
run time with the gather spelling:

```sql
ARRAY(SELECT r FROM unnest(ffrwd.rfdetr.detect(v).boxes) r
      WHERE r.class = 'person' AND r.conf >= 0.5)
```

## Utilities

- `boxes_mask(v, boxes, grow DEFAULT 0, feather DEFAULT 0)` - the rows
  rasterized into a matte; `grow` pads each box in pixels, `feather`
  softens the edge.
- `draw_boxes(v, boxes, thickness DEFAULT 2)` - the boxes drawn on the
  picture.

## Composition

The composition layer lives in `ffrwd/mask_tools` - `blur_where`,
`mosaic_where`, `spotlight`, `cutout` and the `masked` spelling they
share - because it is model-agnostic: any grayscale matte beside any
stream, all native ffmpeg.

## Recipes

`blur-people`, `mosaic-people`, `spotlight`, `replace-background`,
`draw`, `detections` - run `ffrwd list` for each one's variables, or
read the header of the recipe file.

```
ffrwd run ffrwd/rfdetr:blur-people -v source=street.mp4 -v dest=blurred.mp4
```

## Building

The modules build against the wit from the installed `ffrwd/wasm`
package:

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```

## License

This package is **Apache-2.0**, and so are the weights. RF-DETR is
Roboflow's real-time detection transformer; its Nano through Large
sizes are released under Apache-2.0, and this package pins two of the
Large ones.

The weights are not in the archive: the manifest pins them - repo,
revision, file and sha256 - and `ffrwd install` fetches and verifies
them. Both are fp32 ONNX exports made with Roboflow's own exporter
from the official checkpoints, and live in
[imbcmdth/rfdetr-onnx](https://huggingface.co/imbcmdth/rfdetr-onnx)
with the checkpoint hashes and the export script beside them: RF-DETR
Large at 704x704 for detection and RF-DETR Seg Large at 504x504 for
segmentation, about 266 MB together, run through `wasi:nn` on the
machine's own ONNX Runtime.

