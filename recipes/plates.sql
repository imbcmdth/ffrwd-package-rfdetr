-- Every plate as NDJSON, one line per plate per frame - the class `plate`, the
-- confidence, and the box in pixels. No video is written.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), track (video track index, defaults to the first), dest (output path, a .ndjson file)
-- example: ffrwd compile -f packages/ffrwd/rfdetr/recipes/plates.sql -v source=dashcam.mp4 -v dest=plates.ndjson
COPY (
  SELECT ffrwd.rfdetr.detect_plates(v, :conf)
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest'
