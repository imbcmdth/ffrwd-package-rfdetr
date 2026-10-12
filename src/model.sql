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

-- The GPU builds of the same four exports: the host makes each frame into the
-- graph's input, on the device when `memory` is 'gpu', where the frames are
-- held. The detectors' rows are the same rows, decoded on the host; the
-- matte of `segment_mask_gpu` is painted on the device and leaves as rgba.
-- With 'host' the frames stay in host memory and the input is made on the
-- CPU. Each names the same export as its host function, so it loads the same
-- model.
CREATE FUNCTION detect_gpu(v video_stream, conf number DEFAULT 0.25,
                           memory text DEFAULT 'gpu')
RETURNS STRUCT(class text, conf number, x number, y number, w number, h number)[]
  AS 'target/wasm32-wasip2/release/detect_gpu.wasm', 'detect' LANGUAGE wasm;

CREATE FUNCTION detect_faces_gpu(v video_stream, conf number DEFAULT 0.25,
                                 memory text DEFAULT 'gpu')
RETURNS STRUCT(class text, conf number, x number, y number, w number, h number)[]
  AS 'target/wasm32-wasip2/release/detect_faces_gpu.wasm', 'detect_faces' LANGUAGE wasm;

CREATE FUNCTION detect_plates_gpu(v video_stream, conf number DEFAULT 0.25,
                                  memory text DEFAULT 'gpu')
RETURNS STRUCT(class text, conf number, x number, y number, w number, h number)[]
  AS 'target/wasm32-wasip2/release/detect_plates_gpu.wasm', 'detect_plates' LANGUAGE wasm;

CREATE FUNCTION segment_mask_gpu(v video_stream, class text DEFAULT NULL,
                                 conf number DEFAULT 0.25, memory text DEFAULT 'gpu')
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/segment_mask_gpu.wasm', 'segment_mask' LANGUAGE wasm;
