# Video Alignment functions for Vapoursynth

Examples: https://slow.pics/c/rqeq3D97

### Requirements
* pip install numpy
* pip install opencv-python
* [pytorch](https://pytorch.org/)
* [vstools](https://github.com/Jaded-Encoding-Thaumaturgy/vs-tools)
* [julek-plugin](https://github.com/dnjulek/vapoursynth-julek-plugin) (optional, for using  butteraugli when setting temporal alignment to precision=2)
* [tivtc](https://github.com/dubhater/vapoursynth-tivtc) (optional, for temporal alignment resampling)

### Setup
Drop the entire "vs_align" folder to where you typically load scripts from.

<br />

## Spatial Alignment
Aligns the content of a frame to a reference frame using a modified [Rife](https://github.com/megvii-research/ECCV2022-RIFE) AI model. Frames should have vague alignment and no black borders before using. Output clip will have the same dimensions as reference clip. Resize reference clip to get desired output scale.

    import vs_align
    clip = vs_align.spatial(clip, ref, precision="100", iterations=1, blur_strength=0, device="cuda")

__*clip*__  
Misaligned clip. Must be in RGBS format.

__*ref*__  
Reference clip that misaligned clip will be aligned to. Must be in RGBS format.

__*precision*__  
Possible values in % are "50", "100", "200", "400" and "800". Higher values will internally align at higher resolutions to increase precision, which will in turn increase processing time and VRAM usage fast. Lower values are less precise, but can align over larger distances. For problematic cases it can also be helpful to chain multiple alignment calls with increasing precision.  
If the alignment is very close, try a high value.  
If the alignment is not very close, try a low value.  

__*iterations*__  
Runs the alignment multiple times. With more than around 4 passes, artifacts can appear.

__*blur_strength*__  
Blur is only used internally and will not be visible on the output. It can help to ignore strong degredations like compression, halos or noise. If lines on the output get thinner or thicker, try to increase blur a little. It will reduce accuracy, so try to keep it as low as possible. Good values are 0-10. The best alignment will be at Blur 0. 

__*device*__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". This will be very slow on CPU.

<br />

## Temporal Alignment
Aligns clips timewise by searching through one clip and selecting the frame that most closely matches a reference frame in another clip, based on the smallest differences. It is recommended trying to minimize the difference between the two clips by preprocessing. For example removing black borders, cropping to the overlapping region, rough color matching, dehaloing. The closer the clips look to each other, the better the temporal alignment will be. Adapted from [decimatch](https://gist.github.com/po5/b6a49662149005922b9127926f96e68b) by po5.

Example usage if clips have the same frame rate:

    import vs_align
    clip = vs_align.temporal(clip, ref, clip2, tr=20, fallback, thresh=40, precision=1, debug=False)

Example usage if clips have different frame rate:

    import vs_align
    clip = vs_align.temporal(clip, ref, clip2, tr=20, fallback, thresh=40, precision=1, clip_num=30000, clip_den=1001, ref_num=24000, ref_den=1001, debug=False)

__*clip*__  
Misaligned clip. Must be same format and dimensions as ref.

__*ref*__  
Reference clip that misaligned clip will be aligned to. Must be same format and dimensions as clip.

__*clip2*__  
Clip and ref will be used for processing, but the actual output frame is copied from clip2 if set. This is useful if you would like to do preprocessing on clip and ref (like downsizing to increase speed), but would like the ouput frame to be unaltered.

__*tr*__  
Temporal radius. How many frames it will search forward and back to find a match.

__*fallback*__  
Optional fallback clip in case no frame below thresh can be found. Must have the same format and dimensions as clip (or clip2 if it is set).

__*thresh*__  
Threshold for fallback clip. If frame difference is higher than this value, fallback clip is used. Does nothing if no fallback clip is set. Use debug=True to get an idea for the values.

__*precision*__  
1 = more wrong matches, very fast, fine if clips are basically identical besides the temporal misalignment, uses Vapoursynth's [PlaneStats](https://www.vapoursynth.com/doc/functions/video/planestats.html).  
2 = less wrong matches, much slower, much better if clips look more different, at half resolution still better than 1 at full, uses [Butteraugli](https://github.com/dnjulek/vapoursynth-julek-plugin/wiki/Butteraugli).

__*debug*__  
Shows computed difference values for all frames and the best match directly on the frame.

__*clip_num, clip_den, ref_num, ref_den*__  
Resamples clip to ref. Fps Numerator and Denominator for clip and ref (clip2 uses the same as clip).  
This is __optional__ and should __only__ be set if clip and ref have different frame rates (for example 23.976fps and 29.97fps), as it will double processing time. If set, all input clips must be in YUV8..16 format.  
If set, frames will be doubled internally, then resampled, then aligned, then halved again. This is done to make sure no frames are lost, but means processing will take double as long, so only set this if needed!






