-- Blur every face the detector finds, as `blur-faces` does, with every step
-- on the GPU: the picture is decoded onto the device, the detector's input
-- is made there, and the matte and the blur are painted there.
-- The picture leaves the GPU as rgba, so `pix_fmt` asks libx264 for 4:2:0,
-- the format the host recipe writes.
-- variables: source (input media path), conf (confidence threshold, defaults to 0.25), grow (pixels added around each box, defaults to 8), feather (how far the edge softens in pixels, defaults to 4), sigma (blur strength, defaults to 12), track (video track index, defaults to the first), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/rfdetr/recipes/blur-faces-gpu.sql -v source=class.mp4 -v dest=blurred.mp4
COPY (
  SELECT ffrwd.mask_tools.blur_where_gpu(
           v, ffrwd.rfdetr.boxes_mask_gpu(v, ffrwd.rfdetr.detect_faces_gpu(v, :conf),
                                          grow => COALESCE(:grow, 8),
                                          feather => COALESCE(:feather, 4)),
           :sigma), f.audio
  FROM input(:'source') f, unnest(f.video) v
  WHERE v.index = COALESCE(:track, 1)
) TO :'dest' WITH (video_codec 'libx264', crf 20, pix_fmt 'yuv420p')
