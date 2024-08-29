# Video Alignment functions for Vapoursynth
Useful when two sources are available and you would like to combine them in curtain ways, which would only become possible once they are perfectly aligned. For example doing a color transfer, replacing a logo/hardsubs, creating a paired dataset, combining high resolution Bluray chroma with better DVD luma, or similar.

### Requirements
* [pytorch](https://pytorch.org/)
* `pip install numpy`
* `pip install pyiqa && pip install -U setuptools` (optional, only for temporal alignment precision=3)
* [julek-plugin](https://github.com/dnjulek/vapoursynth-julek-plugin) (optional, only for temporal alignment precision=2)
* [tivtc](https://github.com/dubhater/vapoursynth-tivtc) (optional, only for temporal alignment with different frame rates)

### Setup
Put the entire "vs_align" folder into your scripts folder, or where you typically load scripts from.

<br />

## Spatial Alignment
Aligns the content of a frame to a reference frame using a modified [Rife](https://github.com/megvii-research/ECCV2022-RIFE) AI model. Frames should have no black borders before using. Output clip will have the same dimensions as reference clip. Resize reference clip to get desired output scale. Examples: https://slow.pics/c/rqeq3D97
<p align="center">
  <img src="README_img1.png" width="500" />
</p>

    import vs_align
    clip = vs_align.spatial(clip, ref, precision=3, iterations=1, blur_strength=0, device="cuda")

__*`clip`*__  
Misaligned clip. Must be in RGBS format.

__*`ref`*__  
Reference clip that misaligned clip will be aligned to. Must be in RGBS format.

__*`precision`*__  
1, 2, 3, 4, or 5. Higher values will internally align at higher resolutions to increase precision. Each step up doubles the internal resolution, which will in turn increase processing time and VRAM usage. Lower values are less precise, but can correct larger misalignments.
3 works great in most cases.  

__*`iterations`* (optional)__  
Runs the alignment multiple times to dial it in even further. With more than around 5 passes, artifacts can appear.

__*`blur_strength`* (optional)__  
Blur is only used internally and will not be visible on the output. It can help to ignore small details in the alignment process (like compression, noise or halos) and focus more on the general shapes. If lines on the output get thinner or thicker, try to increase blur a little. It will reduce accuracy, so try to keep it as low as possible. Good values are 0-10. The best alignment will be at blur 0. 

__*`device`* (optional)__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". This will be very slow on CPU.

<br />

## Temporal Alignment
Syncs two clips timewise by searching through one clip and selecting the frame that most closely matches the reference clip frame. It is recommended trying to minimize the difference between the two clips by preprocessing. For example removing black borders, cropping to the overlapping region, rough color matching, dehaloing. The closer the clips look to each other, the better the temporal alignment will be. Adapted from [decimatch](https://gist.github.com/po5/b6a49662149005922b9127926f96e68b) by po5.
<p align="center">
  <img src="README_img2.png" width="670" />
</p>

    import vs_align
    clip = vs_align.temporal(clip, ref, clip2, tr=20, precision=1, fallback, thresh=40, device="cuda", debug=False)

__*`clip`*__  
Misaligned clip. Must be same format and dimensions as ref.

__*`ref`*__  
Reference clip that misaligned clip will be aligned to. Must be same format and dimensions as clip.

__*`clip2`* (optional)__  
Clip and ref will be used for the calculations, but the actual output frame is then copied from clip2 if set. This is useful if you would like to do preprocessing on clip and ref (like downsizing to increase speed), but would like the ouput frame to be unaltered.

__*`tr`*__  
Temporal radius. How many frames it will search forward and back to find a match.

__*`precision`*__  
| Value | Precision | Speed     | Usecase                                                                           | Method
| ----- | --------- | --------- | --------------------------------------------------------------------------------- | ------
| 1     | worst     | very fast | when clips are identical besides the temporal misalignment                        | [PlaneStats](https://www.vapoursynth.com/doc/functions/video/planestats.html)
| 2     | better    | slow      | more robust to differences between clips                                          | [Butteraugli](https://github.com/dnjulek/vapoursynth-julek-plugin/wiki/Butteraugli)
| 3     | best      | very slow | extremely accurate with large differences and spatial misalignments between clips | [TOPIQ](https://github.com/chaofengc/IQA-PyTorch/tree/main)

__*`fallback`* (optional)__  
Optional fallback clip in case no frame below thresh can be found. Must have the same format and dimensions as clip (or clip2 if it is set).

__*`thresh`* (optional)__  
Threshold for fallback clip. If frame difference is higher than this value, fallback clip is used. Use "debug=True" to get an idea for the values.  
Does nothing if no fallback clip is set.

__*`device`* (optional)__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu".  
Only has an effect with "precision=3", which will be very slow on CPU.  

__*`debug`* (optional)__  
Overlays computed difference values for all surrounding frames and the best match directly onto the frame.

__*`clip_num`, `clip_den`, `ref_num`, `ref_den`* (optional)__   
Resamples clip to match ref's frame rate. Numerator and Denominator for clip and ref (clip2 uses the same as clip). Set this only if clip and ref have different frame rates (e.g., 29.97fps and 23.976fps), as it will double processing time. Requires all input clips to be in YUV8..16 format.  
To avoid removal of the wrong frames during resampling, frames are doubled, resampled, aligned, then halved again.  
Example: `clip_num=30000, clip_den=1001, ref_num=24000, ref_den=1001`

<br />

## Tips & Troubleshooting
* Enums are available in vs_align/enums.py if needed.
* For problematic cases of spatial misalignment, it can be helpful to chain multiple alignment calls with increasing precision.
* Temporal Alignment precision=3 may need a little time on the first run, as the model needs to download first.
* Temporal Alignment precision=2 and 3 are at half or quarter resolution still better than precision 1.
