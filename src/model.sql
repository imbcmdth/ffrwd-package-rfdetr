-- The model exports, hosted as wasm modules the package ships. The weights
-- themselves are pinned in the manifest and land beside each module at
-- install.
--
-- `detect` returns a row per object per frame: the class as COCO label text,
-- the confidence, and the box in the frame's own pixels. It returns the rows
-- alone, and a reader takes the picture from where it already is:
-- `draw_boxes(v, detect(v))`. `detect_faces` returns those same rows off a
-- model trained on one class, the class always `face`, and everything that
-- reads `detect`'s boxes reads these; `detect_plates` is the same again for
-- licence plates, the class always `plate`. `segment_mask` returns the found
-- instances as one grayscale matte, optionally narrowed to one class name,
-- ready for `ffrwd/mask_tools` and everything else that reads a mask beside
-- the picture.
CREATE FUNCTION detect(v video_stream, conf number DEFAULT 0.25)
RETURNS STRUCT(class text, conf number, x number, y number, w number, h number)[]
  AS 'target/wasm32-wasip2/release/detect.wasm', 'detect' LANGUAGE wasm;

CREATE FUNCTION detect_faces(v video_stream, conf number DEFAULT 0.25)
RETURNS STRUCT(class text, conf number, x number, y number, w number, h number)[]
  AS 'target/wasm32-wasip2/release/detect_faces.wasm', 'detect_faces' LANGUAGE wasm;

CREATE FUNCTION detect_plates(v video_stream, conf number DEFAULT 0.25)
RETURNS STRUCT(class text, conf number, x number, y number, w number, h number)[]
  AS 'target/wasm32-wasip2/release/detect_plates.wasm', 'detect_plates' LANGUAGE wasm;

CREATE FUNCTION segment_mask(v video_stream, class text DEFAULT NULL,
                             conf number DEFAULT 0.25)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/segment_mask.wasm', 'segment_mask' LANGUAGE wasm;
