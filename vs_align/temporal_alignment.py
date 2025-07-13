import vapoursynth as vs
import numpy as np
import torch
import math
import os
import functools
from .enums import TemporalPrecision, Device

core = vs.core


# generates a list of frame numbers that need to be duplicated to resample ref to clip
def gen_duplicates(clip_num, clip_den, ref_num, ref_den, clip_length, ref_length):
    clip_frame_duration = clip_den / clip_num
    ref_frame_duration = ref_den / ref_num
    duplicates = []
    last_ref_frame = None
    for clip_frame in range(clip_length):
        clip_time = clip_frame * clip_frame_duration
        ref_frame = int(clip_time // ref_frame_duration)
        if ref_frame >= ref_length:
            break
        if ref_frame == last_ref_frame:
            duplicates.append(ref_frame)
        last_ref_frame = ref_frame
    return duplicates


# generates a list of frame numbers that need to be deleted to get back to refs original framerate
# each duplicate shifts subsequent frames by the count of previous duplicates
def gen_deletions(duplicates):
    cumulative_duplicates = 0
    deletions = []
    for position in duplicates:
        deletions.append(position + cumulative_duplicates + 1)
        cumulative_duplicates += 1
    return deletions


# using topiq for frame matching
def topiq(clip, ref, out, tr, fallback, thresh, device, fp16, batch_size, debug):
    import timm
    import logging
    import threading
    from collections import OrderedDict
    from urllib.parse import urlparse
    from torch.hub import download_url_to_file
    from .topiq.topiq_arch import CFANet

    # reduces pytorch reserved gpu memory 
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.2'
    torch.cuda.set_per_process_memory_fraction(fraction=1.0)
    
    class LRUFeatureCache:
        def __init__(self, capacity: int):
            self.capacity = capacity
            self.store = OrderedDict()
            self.lock = threading.Lock()

        def __getitem__(self, key):
            with self.lock:
                value = self.store.pop(key, None)
                if value is not None:
                    self.store[key] = value
                return value

        def __setitem__(self, key, value):
            with self.lock:
                if key in self.store:
                    self.store.pop(key)
                elif len(self.store) >= self.capacity:
                    self.store.popitem(last=False)
                self.store[key] = value

        def get(self, key, default=None):
            value = self.__getitem__(key)
            if value is None:
                return default
            return value

        def __len__(self):
            with self.lock:
                return len(self.store)

    def frame_to_tensor(frame: vs.VideoFrame, device: str, fp16: bool = False) -> torch.Tensor:
        dtype = np.float16 if fp16 else np.float32
        array = np.empty((frame.height, frame.width, 3), dtype=dtype)
        for p in range(frame.format.num_planes):
            array[..., p] = np.asarray(frame[p], dtype=dtype)
        tensor = torch.from_numpy(array).to(device)
        return tensor.permute(2, 0, 1).unsqueeze(0)

    def fix_bn(model):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def download_model(url, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        file_name = os.path.basename(urlparse(url).path)
        cached_file = os.path.abspath(os.path.join(model_dir, file_name))
        if not os.path.exists(cached_file):
            logging.info("vs_align: Downloading TOPIQ model into vs_align/topiq.")
            download_url_to_file(url, cached_file, hash_prefix=None, progress=True)
        return cached_file

    def load_models(device, fp16):
        semantic_model = timm.create_model("resnet50", pretrained=False, features_only=True)
        semantic_model.to(device)
        semantic_model.eval()
        fix_bn(semantic_model)
        feature_dim_list = semantic_model.feature_info.channels()
        
        cfanet_url = "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/cfanet_fr_kadid_res50-2c4cc61d.pth"
        
        current_folder = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_folder, "topiq")
        model_path = download_model(cfanet_url, model_dir=model_dir)
        cfanet_model = CFANet(semantic_model=semantic_model, feature_dim_list=feature_dim_list)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        cfanet_model.load_state_dict(state_dict['params'], strict=True)
        cfanet_model.to(device)
        cfanet_model.eval()

        if fp16:
            cfanet_model.half()
            semantic_model.half()
        return cfanet_model, semantic_model

    def _frame_match(n, f):
        with torch.inference_mode():
            # generate features for ref
            ref_tensor   = frame_to_tensor(f[3], device, fp16) # convert frames to tensors
            ref_tensor   = (ref_tensor - mean) / std           # normalize tensors
            ref_features = semantic_model(ref_tensor)          # compute features

            # collect all features for all candidates
            clip_features_collected = []
            clip_features_indices   = []
            for i in range(max(0, n - tr), min(clip.num_frames, n + tr + 1)):
                # get features for clip from cache
                clip_features = FEATURE_CACHE.get(i, None)
                
                # if features not in cache, request frame and generate features
                if clip_features is None:
                    if i == n + tr:  # get forward frame trough vs cache since in subsequent runs of _frame_match this is the only new frame needed
                        clip_tensor  = frame_to_tensor(f[2], device, fp16)
                    else:            # request frames at the start with get_frame
                        clip_tensor  = frame_to_tensor(clip.get_frame(i), device, fp16) # convert frames to tensors
                    clip_tensor      = (clip_tensor - mean) / std                       # normalize tensors
                    clip_features    = semantic_model(clip_tensor)                      # compute features
                    FEATURE_CACHE[i] = clip_features                                    # cache features

                # append frame features to collected features list
                clip_features_collected.append(clip_features)
                clip_features_indices.append(i)

            # features for each frame are actually a list of feature tensors at 5 different scales
            # this groups the features by scale instead, so that each list contains the features of all frames for one scale
            frames_amount = len(clip_features_collected)
            scalewise_candidates = [[] for _ in range(5)]
            for feats_for_i in clip_features_collected:
                for scale in range(5):
                    scalewise_candidates[scale].append(feats_for_i[scale])

            # do inference in multiple batches to limit vram usage
            scores = []
            for start in range(0, frames_amount, batch_size):
                end = min(start + batch_size, frames_amount)

                # concatenate along batch dimension
                clip_features_batched = []
                ref_features_batched  = []
                for scale in range(5):
                    clip_scale = torch.cat(scalewise_candidates[scale][start:end], dim=0)  # shape [current batch_size, C, H, W]
                    ref_scale  = ref_features[scale].repeat(end - start, 1, 1, 1)          # replicate reference
                    clip_features_batched.append(clip_scale)
                    ref_features_batched.append(ref_scale)

                # compare all features in batch
                batch_scores = cfanet_model(clip_features_batched, ref_features_batched)
                scores.append(1 - batch_scores.squeeze(dim=-1))  # shape [current batch_size]
            
            scores = torch.cat(scores, dim=0)  # shape [frames_amount]
            below_thresh = (scores < thresh)
            if not below_thresh.any():
                best_index = None # fallback
            else:
                best_in_frames = scores[below_thresh].argmin().item()      # find best from current frames
                best_index = below_thresh.nonzero()[best_in_frames].item() # find absolute frame number of that one

            # if no match below tresh, use fallback
            if best_index is None:
                final_frame_idx = n
                final_clip      = fallback
            else:
                final_frame_idx = clip_features_indices[best_index]
                final_clip      = out
            
            # sort and overlay debug info 
            if debug:
                scores_dict = dict(zip(clip_features_indices, scores.tolist()))
                ordered_scores = []
                for off in range(len(scores_dict)):
                    fwd, bck = n + off, n - off
                    if fwd < clip.num_frames and fwd in scores_dict:
                        ordered_scores.append((fwd, scores_dict[fwd]))
                    if bck >= 0 and bck != fwd and bck in scores_dict:
                        ordered_scores.append((bck, scores_dict[bck]))
                
                if best_index is None:
                    best_str = "fallback"
                else:
                    best_frame = clip_features_indices[best_index]
                    offset = best_frame - n
                    best_str = "n" if offset == 0 else f"n{offset:+d}"
                
                lines = [f"{'n  ' if frm == n else f'n{frm - n:+d}'}: {val}" for frm, val in ordered_scores]
                text_debug = "\n".join(["Method: TOPIQ", f"Best Match: {best_str}"] + lines)
                return core.text.Text([final_clip[final_frame_idx]], text=text_debug).get_frame(0)

            return final_clip.get_frame(final_frame_idx)
        
    FEATURE_CACHE = LRUFeatureCache(capacity=tr * 2 + 3)
    cfanet_model, semantic_model = load_models(device, fp16)
    mean    = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16 if fp16 else torch.float32).view(1, 3, 1, 1)
    std     = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16 if fp16 else torch.float32).view(1, 3, 1, 1)
    shifted = core.std.Trim(clip, first=tr)
    return core.std.ModifyFrame(out, clips=[clip, out, shifted, ref], selector=_frame_match)


def temporal(clip, ref, out=None, precision=1, tr=20, fallback=None, thresh=100.0, clip_num=None, clip_den=None, ref_num=None, ref_den=None, batch_size=None, device="cuda", debug=False):

    # convert enums
    if isinstance(device, Device):
        device = device.value
    if isinstance(precision, TemporalPrecision):
        precision = precision.value

    # checks for inputs
    if tr < 0:
        raise ValueError("vs_align.temporal: Temporal radius (tr) can not be negative.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("vs_align.temporal: Batch_size must be at least 1 or None. None uses maximum possible batch_size.")
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_align.temporal: Clip is not a vapoursynth clip.")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("vs_align.temporal: Ref is not a vapoursynth clip.")
    if out is not None and not isinstance(out, vs.VideoNode):
        raise TypeError("vs_align.temporal: Out is not a vapoursynth clip.")
    if fallback is not None and not isinstance(fallback, vs.VideoNode):
        raise TypeError("vs_align.temporal: Fallback is not a vapoursynth clip.")

    # checks for format and dimensions
    if clip.format.id != ref.format.id:
        raise ValueError("vs_align.temporal: Clip and ref must have the same format.")
    if clip.width != ref.width or clip.height != ref.height:
        raise ValueError("vs_align.temporal: Clip and ref must have the same dimensions.")
    
    if out is not None and fallback is not None:
        if out.format.id != fallback.format.id:
            raise ValueError("vs_align.temporal: Out and fallback must have the same format.")
        if out.width != fallback.width or out.height != fallback.height:
            raise ValueError("vs_align.temporal: Out and fallback must have the same dimensions.")
    
    if out is None and fallback is not None:
        if clip.format.id != fallback.format.id:
            raise ValueError("vs_align.temporal: Clip and fallback must have the same format.")
        if clip.width != fallback.width or clip.height != fallback.height:
            raise ValueError("vs_align.temporal: Clip and fallback must have the same dimensions.")
    
    # checks for clip length
    if fallback is not None and fallback.num_frames < ref.num_frames:
        raise ValueError("vs_align.temporal: Fallback must be at least as long as ref.")
    if out is not None and out.num_frames != clip.num_frames:
        raise ValueError("vs_align.temporal: Out must be the same length as clip.")
    
    # checks for resampling
    resample_all_none = all(param is None for param in [clip_num, clip_den, ref_num, ref_den])
    resample_all_set  = all(param is not None for param in [clip_num, clip_den, ref_num, ref_den])
    if not resample_all_none and not resample_all_set:
        raise ValueError("vs_align.temporal: Parameters clip_num, clip_den, ref_num, and ref_den are used together. Set all of them or none.")
    resample_clip = resample_all_set and clip_num / clip_den < ref_num / ref_den
    resample_ref  = resample_all_set and clip_num / clip_den > ref_num / ref_den
    
    # defaults
    if out is None:
        out = clip
    if fallback is None:
        thresh = float('inf')
    if batch_size is None:
        batch_size = tr
    batch_size = batch_size * 2 + 1 # make batch_size scale like tr
    fp16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 7
    ref_orig_length = ref.num_frames

    ##### prepare clips #####
    
    # if clip's framerate is lower than ref's, resample clip
    if resample_clip:
        tr = math.ceil(tr * ((ref_num / ref_den) / (clip_num / clip_den))) # make tr higher to compensate for duplicated frames - not the best solution as this makes it slower (todo: find an easy way to skip comparing against duplicated frames instead)
        duplicates = gen_duplicates(ref_num, ref_den, clip_num, clip_den, ref.num_frames, clip.num_frames)
        if duplicates: # if framerates are very close it can happen that no duplicates are generated
            clip = core.std.DuplicateFrames(clip, frames=duplicates)
            out  = core.std.DuplicateFrames(out,  frames=duplicates)
    
    # if clip's framerate is higher than ref's, resample ref
    if resample_ref:
        duplicates = gen_duplicates(clip_num, clip_den, ref_num, ref_den, clip.num_frames, ref.num_frames)
        if duplicates: # if framerates are very close it can happen that no duplicates are generated
            ref = core.std.DuplicateFrames(ref, frames=duplicates)
            if fallback:
                fallback = core.std.DuplicateFrames(fallback, frames=duplicates)
        
    # clamp if clip or ref is already float
    if clip.format.sample_type == vs.FLOAT and precision == 3:
        clip = core.std.Expr(clip, "x 0 max 1 min")
    if ref.format.sample_type  == vs.FLOAT and precision == 3:
        ref  = core.std.Expr(ref,  "x 0 max 1 min")

    # convert to appropriate format for topiq and butteraugli
    if precision in [2, 3]:
        if   precision == 2 and device == "cpu":
            format_id = vs.RGB24 # butteraugli cpu
        elif precision == 2 and device == "cuda":
            format_id = vs.RGBS  # butteraugli gpu
        elif precision == 3 and fp16:
            format_id = vs.RGBH  # topiq fp16
        elif precision == 3 and not fp16:
            format_id = vs.RGBS  # topiq fp32
        if clip.format.id != format_id:
            if clip.format.color_family == vs.YUV:
                clip = core.resize.Point(clip, format=format_id, matrix_in_s="709")
                ref  = core.resize.Point(ref,  format=format_id, matrix_in_s="709")
            else:
                clip = core.resize.Point(clip, format=format_id)
                ref  = core.resize.Point(ref,  format=format_id)

    #convert to something other than RGBH for planestats
    if precision == 1 and clip.format.id == vs.RGBH:
        clip = core.resize.Point(clip, format=vs.RGBS)
        ref  = core.resize.Point(ref,  format=vs.RGBS)

    # extend clip and out to match ref's length
    if clip.num_frames < ref.num_frames:
        clip = core.std.Splice([clip, core.std.BlankClip(clip, length=ref.num_frames - clip.num_frames, keep=True)])
        out  = core.std.Splice([out,  core.std.BlankClip(out,  length=ref.num_frames - out.num_frames,  keep=True)])

    ##### frame matching #####

    # do temporal alignment with topiq
    if precision == 3:
        result = topiq(clip, ref, out, tr, fallback, thresh, device, fp16, batch_size, debug)
    
    # else do temporal aligment with butteraugli or planestats
    # based on "decimatch" by po5 https://gist.github.com/po5/b6a49662149005922b9127926f96e68b
    else:
        if   precision == 1:
            method_name = "PlaneStats"
            method_func = core.std.PlaneStats
            prop_key    = "PlaneStatsDiff"
        elif precision == 2 and device == "cpu":
            method_name = "Butteraugli"
            method_func = functools.partial(core.julek.Butteraugli, intensity_target=80)
            prop_key    = "_FrameButteraugli"
        elif precision == 2 and device == "cuda":
            method_name = "Butteraugli"
            method_func = functools.partial(core.vship.BUTTERAUGLI, intensity_multiplier=80)
            prop_key    = "_BUTTERAUGLI_INFNorm"
        else:
            raise TypeError("vs_align.temporal: Precision must be 1, 2, or 3.")
        
        # generates a list of clips, each one shifted 1 frame forwards or backwards
        def gen_shifts(c, n, forward=True, backward=True):
            shifts = [c]
            for cur in range(1, n+1):
                if forward:
                    shifts.append(c[cur:]+c[0]*cur)
                if backward:
                    shifts.append(c.std.DuplicateFrames([0]*cur)[:-1*cur])
            return shifts
    
        # selects the clip with the best match to ref for the current frame
        def _select(n, f):
        
            # if tr=0, make sure f is still an array
            if not isinstance(f, list):
                f = [f]

            scores = [float(diff.props[prop_key]) for diff in f]
            best   = min(indices, key=lambda i: scores[i])

            if fallback and any(score < thresh for score in scores):
                best_clip = shifts_out[best]
            elif fallback:
                best_clip = fallback
                best      = "fallback"
            else:
                best_clip = shifts_out[best]
            
            if do_debug:
                def shift_label(i: int) -> str:
                    if i == 0:
                        return "n  "
                    magnitude = (i + 1) // 2
                    sign = 1 if i % 2 == 1 else -1
                    return f"n{sign * magnitude:+d}"
                shift_labels = [shift_label(i) for i in range(len(scores))]
                if isinstance(best, int):
                    best_label = shift_labels[best]
                else:
                    best_label = str(best)
                debug_lines = [f"Method: {method_name} ({prop_key})",f"Best Match: {best_label}",*[f"{shift_labels[i]}: {s}" for i, s in enumerate(scores)]]
                return best_clip.text.Text("\n".join(debug_lines), alignment)

            return best_clip
        
        shifts_clip = gen_shifts(clip, tr)
        shifts_out  = gen_shifts(out,  tr)
        diffs       = [method_func(c, ref) for c in shifts_clip]
        indices     = list(range(len(diffs)))
        do_debug, alignment = debug if isinstance(debug, tuple) else (debug, 7)
        result = core.std.FrameEval(shifts_out[0], _select, diffs)

    ##### prepare output #####

    # delete duplicate frames from resampling, so that output matches ref's original framerate 
    if resample_ref and duplicates:
        deletions = gen_deletions(duplicates)
        result    = core.std.DeleteFrames(result, frames=deletions)
    
    # update fps prop if it was set manually
    if resample_ref or resample_clip:
        result    = core.std.AssumeFPS(result, fpsnum=ref_num, fpsden=ref_den)
    # copy framerates per frame from ref to fix framerates that are set trough a pattern, which would break after shuffeling frames around
    else:
        result    = core.std.CopyFrameProps(result, ref, props=["_DurationNum", "_DurationDen"])
    
    # trim to ref length and return
    if result.num_frames > ref_orig_length:
        return core.std.Trim(result, length=ref_orig_length)
    return result
