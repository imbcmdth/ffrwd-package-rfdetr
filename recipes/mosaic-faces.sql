-- Mosaic every face the detector finds, on the first video track, or the one
-- `track` names. `grow` pads each box so hair and the jaw line go with it.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), grow (pixels added around each box, defaults to 8), feather (how far the edge softens in pixels, defaults to 4), size (mosaic block size, defaults to 16), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/rfdetr/recipes/mosaic-faces.sql -v source=class.mp4 -v dest=mosaic.mp4
COPY (
  SELECT ffrwd.mask_tools.mosaic_where(
           v, ffrwd.rfdetr.boxes_mask(ffrwd.rfdetr.detect_faces(v, :conf),
                                      COALESCE(:grow, 8), COALESCE(:feather, 4)),
           :size), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
