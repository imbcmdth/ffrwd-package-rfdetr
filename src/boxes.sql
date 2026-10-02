-- The row readers: pure Rust modules, no model. Each reads the picture and
-- the rows a detector wrote about it, and names only the fields it uses, so
-- any rows carrying a box will do: a detector's, or a tracker's with fields
-- of its own beside the box.
--
-- `boxes_mask` rasterizes the boxes into a grayscale matte the size of the
-- picture: `grow` pads each box outward in pixels, `feather` softens the edge
-- over that many. `draw_boxes` draws the boxes on the picture as green
-- outlines, and reads its coordinates as whole pixels.
CREATE FUNCTION boxes_mask(v video_stream,
                           boxes STRUCT(x number, y number, w number, h number)[],
                           grow number DEFAULT 0, feather number DEFAULT 0)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/boxes_mask.wasm', 'boxes_mask' LANGUAGE wasm;

CREATE FUNCTION draw_boxes(v video_stream,
                           boxes STRUCT(x number, y number, w number, h number)[],
                           thickness number DEFAULT 2)
RETURNS video_stream
  AS 'target/wasm32-wasip2/release/draw_boxes.wasm', 'draw_boxes' LANGUAGE wasm;
