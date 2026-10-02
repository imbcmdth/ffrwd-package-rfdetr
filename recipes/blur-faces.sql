-- Blur every face the detector finds, on the first video track, or the one
-- `track` names. `grow` pads each box so hair and the jaw line go with it.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), grow (pixels added around each box, defaults to 8), feather (how far the edge softens in pixels, defaults to 4), sigma (blur strength, defaults to 12), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/rfdetr/recipes/blur-faces.sql -v source=class.mp4 -v dest=blurred.mp4
COPY (
  SELECT ffrwd.mask_tools.blur_where(
           v, ffrwd.rfdetr.boxes_mask(v, ffrwd.rfdetr.detect_faces(v, :conf),
                                      grow => COALESCE(:grow, 8),
                                      feather => COALESCE(:feather, 4)),
           :sigma), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
