import vapoursynth as vs
import torch
import numpy as np
from .enums import TemporalPrecision, Device

core = vs.core

#adapted from "decimatch" by po5 https://gist.github.com/po5/b6a49662149005922b9127926f96e68b

def frame_to_tensor(frame: vs.VideoFrame, device: str, fp16: bool = False) -> torch.Tensor:
    dtype = np.float16 if fp16 else np.float32
    array = np.empty((frame.height, frame.width, 3), dtype=dtype)
    for p in range(frame.format.num_planes):
        array[..., p] = np.asarray(frame[p], dtype=dtype)
    tensor = torch.from_numpy(array).to(device)
    tensor.clamp_(0, 1)
    return tensor.permute(2, 0, 1).unsqueeze(0)

def vs_pyiqa(clip, ref, iqa_model, prop_key, device, fp16):
    def _evaluate_frame(n, f):
        ref_frame = ref.get_frame(n)
        ref_tensor = frame_to_tensor(ref_frame, device, fp16)
        clip_tensor = frame_to_tensor(f, device, fp16)
        score = 1 - iqa_model(clip_tensor, ref_tensor)
        score = score.cpu().item()
        output_clip = core.std.SetFrameProp(clip, prop=prop_key, floatval=score)
        return output_clip
    return core.std.FrameEval(clip, eval=_evaluate_frame, prop_src=[clip])

def temporal(clip, ref, clip2=None, tr=30, precision=1, fallback=None, thresh=40, clip_num=None, clip_den=None, ref_num=None, ref_den=None, device="cuda" if torch.cuda.is_available() else "cpu", fp16=False, debug=False):
    from pyiqa import create_metric
    
    # convert enums
    if isinstance(device, Device):
        device = device.value
    if isinstance(precision, TemporalPrecision):
        precision = precision.value

    # checks
    if fp16 and device == "cpu":
        raise ValueError("Fp16 does not work on CPU.")
    if clip.format.id != ref.format.id:
        raise ValueError("Clip and ref must be the same format.")
    if clip.width != ref.width or clip.height != ref.height:
        raise ValueError("Clip and ref must have the same dimensions.")
    if clip2 is not None and fallback is not None:
        if clip2.format.id != fallback.format.id:
            raise ValueError("Clip2 and fallback must be the same format.")
        if clip2.width != fallback.width or clip2.height != fallback.height:
            raise ValueError("Clip2 and fallback must have the same dimensions.")
    if clip2 is None and fallback is not None:
        if clip.format.id != fallback.format.id:
            raise ValueError("Clip and fallback must be the same format.")
        if clip.width != fallback.width or clip.height != fallback.height:
            raise ValueError("Clip and fallback must have the same dimensions.")
    if clip_num is not None and clip_den is not None and ref_num is not None and ref_den is not None:
        if clip.format.color_family != vs.YUV:
            raise ValueError("All input clips must be in YUV format when using clips with different frame rates.")

    # if clip2 is not set, use clip
    if clip2 is None:
        clip2 = clip

    # precision modes
    iqa_model = None
    if precision == 3:
        metric = 'topiq_fr'
        iqa_model = create_metric('topiq_fr', metric_mode='FR', device=torch.device(device)).half() if fp16 else create_metric('topiq_fr', metric_mode='FR', device=torch.device(device))
        process = vs_pyiqa
        prop_key = "pyiqa_TOPIQ"
        process_name = "TOPIQ"
        select_func = min
        compare_func = lambda score: score < thresh
    elif precision == 2:
        process = core.julek.Butteraugli
        prop_key = "_FrameButteraugli"
        process_name = "Butteraugli"
        select_func = min
        compare_func = lambda score: score < thresh
    else:  #precision == 1
        process = core.std.PlaneStats
        prop_key = "PlaneStatsDiff"
        process_name = "PlaneStats"
        select_func = min
        compare_func = lambda score: score < thresh

    if clip_num is not None and clip_den is not None and ref_num is not None and ref_den is not None:
        #double framerate to make sure no frames are lost during resampling
        ref = core.resize.Bicubic(ref, width=clip.width, height=clip.height)
        clip = core.std.Interleave([clip, clip])
        ref = core.std.Interleave([ref, ref])
        clip2 = core.std.Interleave([clip2, clip2]) if clip2 else clip2
        
        #resample clip to same fps as ref (but still doubled)
        clip = core.resize.Point(clip, format=vs.YUV444P8, range_in_s='full', range_s='full')
        ref = core.std.AssumeFPS(ref, fpsnum=ref_num, fpsden=ref_den)
        clip = core.std.Interleave([clip, clip])
        clip = core.std.AssumeFPS(clip, fpsnum=clip_num*2, fpsden=clip_den)
        clip_tivtc = core.tivtc.TDecimate(clip, rate=ref_num/ref_den, mode=7)
        clip_tivtc = core.std.AssumeFPS(clip_tivtc, fpsnum=ref_num, fpsden=ref_den)
        
        clip2 = core.std.Interleave([clip2, clip2])
        clip2 = core.std.AssumeFPS(clip2, fpsnum=clip_num*2, fpsden=clip_den)
        clip2 = core.tivtc.TDecimate(clip=clip, clip2=clip2, rate=ref_num/ref_den, mode=7)
        clip2 = core.std.AssumeFPS(clip2, fpsnum=ref_num, fpsden=ref_den)
        clip = clip_tivtc

    #convert to the appropriate format for butteraugli and pyiqa
    if precision in [2, 3]:
        format_id = vs.RGBH if precision == 3 and fp16 else vs.RGBS if precision == 3 else vs.RGB24
        if clip.format.id != format_id:
            if clip.format.color_family == vs.YUV:
                clip = core.resize.Point(clip, format=format_id, matrix_in_s="709", range_in_s='full', range_s='full')
                ref = core.resize.Point(ref, format=format_id, matrix_in_s="709", range_in_s='full', range_s='full')
            elif clip.format.color_family == vs.RGB:
                clip = core.resize.Point(clip, format=format_id)
                ref = core.resize.Point(ref, format=format_id)
            elif clip.format.color_family == vs.GRAY:
                clip = core.resize.Point(clip, format=format_id)
                ref = core.resize.Point(ref, format=format_id)
            else:
                raise ValueError("Unsupported clip format.")

    #helper function to generate shifts
    def gen_shifts(c, n, forward=True, backward=True):
        shifts = [c]
        for cur in range(1, n+1):
            if forward:
                shifts.append(c[cur:]+c[0]*cur)
            if backward:
                shifts.append(c.std.DuplicateFrames([0]*cur)[:-1*cur])
        return shifts

    #generate shifted clips
    if clip_num is not None and clip_den is not None and ref_num is not None and ref_den is not None:
        clip = gen_shifts(clip, tr*2)
        clip2 = gen_shifts(clip2, tr*2)
    else:
        clip = gen_shifts(clip, tr)
        clip2 = gen_shifts(clip2, tr)

    diffs = [process(c, ref, iqa_model, prop_key, device, fp16) if iqa_model else process(c, ref) for c in clip]
    indices = list(range(len(diffs)))
    do_debug, alignment = debug if isinstance(debug, tuple) else (debug, 7)

    def _select(n, f):
        scores = [float(diff.props[prop_key]) for diff in f]
        best = select_func(indices, key=lambda i: scores[i])

        if fallback and any(compare_func(score) for score in scores):
            best_clip = clip2[best]
        elif fallback:
            best_clip = fallback
            best = "fallback"
        else:
            best_clip = clip2[best]
        
        if do_debug:
            return best_clip.text.Text(
                "\n".join([f"{process_name} ({prop_key})", f"Best: {best}", *[f"{i}: {s}" for i, s in enumerate(scores)]]), alignment
            )
        return best_clip

    result = core.std.FrameEval(clip2[0], _select, diffs)

    #half framerate to get original one
    if clip_num is not None and clip_den is not None and ref_num is not None and ref_den is not None:
        result = core.std.SelectEvery(clip=result, cycle=2, offsets=0)
    return result
