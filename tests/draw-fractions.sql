-- draw_boxes handed boxes in fractions of a pixel: refused when compiled,
-- naming the field, since draw_boxes reads whole pixels. The fixture is
-- `tests/fractions`, which the workspace builds beside the modules.
CREATE FUNCTION fractions(v video_stream)
RETURNS STRUCT(x number, y number, w number, h number)[]
  AS 'target/wasm32-wasip2/release/fractions.wasm', 'fractions' LANGUAGE wasm;

COPY (
  SELECT ffrwd.rfdetr.draw_boxes(v, fractions(v))
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = 1
) TO :'dest' WITH (video_codec 'libx264', crf 20)
