-- The same call over a detector's rows, which are whole pixels: compiles.
COPY (
  SELECT ffrwd.rfdetr.draw_boxes(v, ffrwd.rfdetr.detect_faces(v))
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = 1
) TO :'dest' WITH (video_codec 'libx264', crf 20)
