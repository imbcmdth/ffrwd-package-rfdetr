-- Every face as NDJSON, one line per face per frame - the class `face`, the
-- confidence, and the box in pixels. No video is written.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), track (video track index, defaults to the first), dest (output path, a .ndjson file)
-- example: ffrwd compile -f packages/ffrwd/rfdetr/recipes/faces.sql -v source=class.mp4 -v dest=faces.ndjson
COPY (
  SELECT ffrwd.rfdetr.detect_faces(v, :conf)
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest'
