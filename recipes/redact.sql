-- Redact the two things a frame most often identifies someone by: every face
-- and every licence plate the detectors find are mosaicked, on the first video
-- track or the one `track` names. Two detectors, two mattes, one picture: the
-- plates are mosaicked over the frame the faces were already mosaicked on.
-- `grow` pads each box so hair, the jaw line and the plate's frame go with it.
-- variables: source (input media path), conf (confidence threshold for both detectors, defaults to 0.25), grow (pixels added around each box, defaults to 8), feather (how far the edge softens in pixels, defaults to 4), size (mosaic block size, defaults to 16), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/rfdetr/recipes/redact.sql -v source=street.mp4 -v dest=redacted.mp4
COPY (
  SELECT ffrwd.mask_tools.mosaic_where(
           ffrwd.mask_tools.mosaic_where(
             v, ffrwd.rfdetr.boxes_mask(ffrwd.rfdetr.detect_faces(v, :conf),
                                        COALESCE(:grow, 8), COALESCE(:feather, 4)),
             COALESCE(:size, 16)),
           ffrwd.rfdetr.boxes_mask(ffrwd.rfdetr.detect_plates(v, :conf),
                                   COALESCE(:grow, 8), COALESCE(:feather, 4)),
           COALESCE(:size, 16)),
         f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20)
