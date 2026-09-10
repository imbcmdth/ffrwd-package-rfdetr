# ffrwd/rfdetr

RF-DETR detection, instance segmentation, face detection and licence
plate detection. Same API as YOLO26: `detect`, `detect_faces` and
`detect_plates` produce rows, `segment_mask` produces a matte, the utilities turn rows into mattes or drawn
overlays, and everything downstream is native ffmpeg.

## Model exports

- `detect(v, conf DEFAULT 0.25)` returns
  `STRUCT(v video_stream, boxes STRUCT(class text, conf number, x number, y number, w number, h number)[])` -
  the picture untouched, one row per object per frame, boxes in the
  frame's own pixels, classes as COCO label text.
- `detect_faces(v, conf DEFAULT 0.25)` returns the same record with
  one row per face, every row's class the text `face`. A different
  model behind the same shape, so `boxes_mask`, `draw_boxes` and the
  gather spelling below read its rows unchanged.
- `detect_plates(v, conf DEFAULT 0.25)` returns the same record again
  with one row per licence plate, every row's class the text `plate`.
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

## Faces

`detect_faces` is for the face itself, not the person around it: the
box it returns is the head, which is what a privacy blur has to cover
and what an age or identity model reads. COCO has no face class, so
this is a second RF-DETR, Medium size, fine-tuned for one class.

It runs at a single 576x576 pass over the whole frame, so how much of
the frame a face fills decides whether it is found. On WIDER FACE's
validation set, at the default confidence, it finds 98 in 100 faces
over 96 px across, 82 in 100 between 32 and 96 px, and 17 in 100
under 32 px. Footage where the faces that matter are small - a wide
shot of a room - wants a closer camera or a crop before the
detector, not a lower threshold.

One answer the head gives is not a face: on a crowd it sometimes
returns a box covering nearly the whole frame at a middling
confidence. A face is taller than it is wide, so the module drops a
box that is both wider than 1.3 times its height and larger than a
quarter of the frame. A wide box that is small, a face half behind
something, stays; so does a large box that is tall, a face close to
the camera.

For a blur, pad the box: `boxes_mask`'s `grow` adds pixels around
each face, and its `feather` softens the edge, so hair and the jaw
line go with the face. The `blur-faces` recipe defaults to 8 and 4.

## Plates

`detect_plates` finds the licence plate itself, the rectangle a
redaction has to cover, on any vehicle from any angle the plate is
legible from. COCO has no plate class either, so this is a third
RF-DETR, Medium size, fine-tuned for one class - by us, on Open Images'
"Vehicle registration plate" boxes, since no plate fine-tune with a
licence the registry could carry existed.

It runs at a single 576x576 pass over the whole frame, so how much of
the frame a plate fills decides whether it is found. On Open Images'
own test split, at the default confidence, it finds 98 in 100 plates
over 96 px across, 96 in 100 between 32 and 96 px, and 47 in 100 under
32 px, and 91 of every 100 boxes it returns are plates. A plate small
in a wide shot - the far lane of a dashcam, the back of a car park -
wants a crop before the detector, not a lower threshold; a plate at
night or under a headlight's glare is the other thing this training
data had few of.

Unlike a face, a plate is wider than it is tall, so the crowd rule
above does not apply here: a wide box that is a quarter of the frame is
a plate close to the camera, and it stays.

For a mosaic, pad the box the same way: the `blur-plates` and
`mosaic-plates` recipes default `grow` to 8 and `feather` to 4, so the
plate's frame and the screws go with it. `redact` masks faces and
plates in one pass, the plates over the frame the faces were already
masked on.

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

People: `blur-people`, `mosaic-people`, `spotlight`,
`replace-background`, `draw`, `detections`. Faces: `blur-faces`,
`mosaic-faces`, `draw-faces`, `faces`. Plates: `blur-plates`,
`mosaic-plates`, `draw-plates`, `plates`. Both at once: `redact`. Run
`ffrwd list ffrwd/rfdetr` for each one's variables, or read the header
of the recipe file.

```
ffrwd run ffrwd/rfdetr:blur-people -v source=street.mp4 -v dest=blurred.mp4
ffrwd run ffrwd/rfdetr:blur-faces -v source=class.mp4 -v dest=blurred.mp4
ffrwd run ffrwd/rfdetr:redact -v source=dashcam.mp4 -v dest=redacted.mp4
```

## Building

The modules build against the wit from the installed `ffrwd/wasm`
package, and the ones that read a picture take the frame as `rgba`,
converted upstream by ffmpeg with the stream's own range and matrix,
and turn it into the model's tensor with the
[ffrwd-frame](https://github.com/imbcmdth/ffrwd-frame) crate:

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```

## License

This package is **Apache-2.0**, and so are the weights. RF-DETR is
Roboflow's real-time detection transformer; its Nano through Large
sizes are released under Apache-2.0, and this package pins two of the
Large ones. The face detector is RF-DETR Medium as fine-tuned by
[Herojayjay/RFDETR-Face-Detection](https://huggingface.co/Herojayjay/RFDETR-Face-Detection),
also Apache-2.0, on a Kaggle face dataset of about 16,700 images. The
plate detector is RF-DETR Medium fine-tuned by us, from Roboflow's
Medium checkpoint, on Open Images V7's "Vehicle registration plate"
boxes - annotations CC BY 4.0, images CC BY 2.0 - and released under
the same Apache-2.0.

The weights are not in the archive: the manifest pins them - repo,
revision, file and sha256 - and `ffrwd install` fetches and verifies
them. All four are fp32 ONNX exports made with Roboflow's own
exporter from their checkpoints, and live in
[imbcmdth/rfdetr-onnx](https://huggingface.co/imbcmdth/rfdetr-onnx)
with the checkpoint hashes and the export scripts beside them: RF-DETR
Large at 704x704 for detection, RF-DETR Seg Large at 504x504 for
segmentation and RF-DETR Medium at 576x576 for faces and again for
plates, about 520 MB together, run through `wasi:nn` on the machine's
own ONNX Runtime.
