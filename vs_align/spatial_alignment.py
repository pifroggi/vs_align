# Original Rife Frame Interpolation by hzwer
# https://github.com/megvii-research/ECCV2022-RIFE
# https://github.com/hzwer/Practical-RIFE

# Modifications to use Rife for Image Alignment by pifroggi
# or tepete on the "Enhance Everything!" Discord Server
# https://github.com/pifroggi/vs_align

# Additional helpful github issues
# https://github.com/megvii-research/ECCV2022-RIFE/issues/278
# https://github.com/megvii-research/ECCV2022-RIFE/issues/344

import vapoursynth as vs
import os
import torch
import numpy as np
from .enums import SpatialPrecision, Device
from .rife.IFNet_HDv3_v4_14_align import IFNet

core = vs.core


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame):
    frame_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    for p in range(3):
        np.copyto(np.asarray(frame[p]), frame_np[:, :, p])


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    frame_np = np.dstack([np.asarray(frame[p]) for p in range(3)])
    return torch.from_numpy(frame_np).to(device).permute(2, 0, 1).unsqueeze(0)


def mask_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    frame_np = np.array(frame[0], copy=True)
    return torch.from_numpy(frame_np).to(device).unsqueeze(0).unsqueeze(0)


def calculate_padding(height, width, padding):
    pad_height = (padding - height % padding) % padding
    pad_width = (padding - width % padding) % padding
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    return top_pad, bottom_pad, left_pad, right_pad


def spatial(clip, ref, mask=None, precision=3, iterations=1, lq_input=False, device="cuda"):

    # checks
    if isinstance(device, Device):
        device = device.value
    if isinstance(precision, SpatialPrecision):
        precision = precision.value
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("Clip is not a vapoursynth clip.")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("Ref is not a vapoursynth clip.")
    if mask and not isinstance(mask, vs.VideoNode):
        raise TypeError("Mask is not a vapoursynth clip.")
    if clip.format.color_family != vs.RGB:
        raise ValueError("Clip must be in RGB format.")
    if ref.format.color_family  != vs.RGB:
        raise ValueError("Ref must be in RGB format.")

    # parameters
    its    = iterations
    blur   =  2 if lq_input else 0
    smooth = 11 if lq_input else 0
    fp16   = device == "cuda" and torch.cuda.get_device_capability()[0] >= 7
    
    # scales
    s1 = [16,  8,   4,   2   ]
    s2 = [8,   4,   2,   1   ]
    s3 = [4,   2,   1,   0.5 ]
    s4 = [2,   1,   0.5, 0.25]
    
    # padding for scales
    tp1, bp1, lp1, rp1 = calculate_padding(ref.height, ref.width, 64)
    tp2, bp2, lp2, rp2 = calculate_padding(ref.height, ref.width, 32)
    tp3, bp3, lp3, rp3 = calculate_padding(ref.height, ref.width, 16)
    tp4, bp4, lp4, rp4 = calculate_padding(ref.height, ref.width,  8)
    
    # load model
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_folder, "rife", "flownet_v4.14.pkl")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = IFNet().to(device)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    if fp16:
        model.half()
    
    # convert inputs to half if fp16/full if not fp16
    format_id      = vs.RGBH  if fp16 else vs.RGBS 
    format_id_mask = vs.GRAYH if fp16 else vs.GRAYS
    clip_orig_format = clip.format.id
    if clip.format.id != format_id:
        clip = core.resize.Point(clip, format = format_id)
    if ref.format.id  != format_id:
        ref  = core.resize.Point(ref,  format = format_id)
    if mask:
        if mask.format.color_family != vs.GRAY:
            mask = core.std.ShufflePlanes(mask, 0, vs.GRAY)
        if mask.format.id    != format_id_mask:
            mask = core.resize.Point(mask, format = format_id_mask)

    # cache tensor of last mask frame
    if mask and mask.num_frames < clip.num_frames:
        fmask_last = mask_to_tensor(mask.get_frame(mask.num_frames - 1), device)



    def _align(n, f):
        with torch.no_grad():
            fclip = frame_to_tensor(f[0], device).clamp_(0, 1)
            fref = frame_to_tensor(f[1], device).clamp_(0, 1)
            if mask:
                if n < mask.num_frames:
                    fmask = mask_to_tensor(f[2], device)
                else:
                    fmask = fmask_last # use cached last frame if mask clip is too short or just one frame
            
            with torch.amp.autocast(device, enabled=fp16):
                if   precision == 1:
                    fclip = model(fclip, fref, fmask if mask else None, s1, lp1, rp1, tp1, bp1, blur=blur, smooth=smooth, its=its, compensate=True,                       device=device, fp16=fp16)
                elif precision == 2:
                    fclip = model(fclip, fref, fmask if mask else None, s1, lp1, rp1, tp1, bp1, blur=9,    smooth=31*its, its=its, compensate=True if its > 1 else False, device=device, fp16=fp16)
                    fclip = model(fclip, fref, fmask if mask else None, s2, lp2, rp2, tp2, bp2, blur=blur, smooth=smooth, its=1,   compensate=True,                       device=device, fp16=fp16)
                elif precision == 3:
                    fclip = model(fclip, fref, fmask if mask else None, s1, lp1, rp1, tp1, bp1, blur=9,    smooth=31*its, its=its, compensate=True if its > 1 else False, device=device, fp16=fp16)
                    fclip = model(fclip, fref, fmask if mask else None, s2, lp2, rp2, tp2, bp2, blur=9,    smooth=15*its, its=its, compensate=True if its > 1 else False, device=device, fp16=fp16)
                    fclip = model(fclip, fref, fmask if mask else None, s3, lp3, rp3, tp3, bp3, blur=blur, smooth=smooth, its=1,   compensate=True,                       device=device, fp16=fp16)
                elif precision == 4:
                    fclip = model(fclip, fref, fmask if mask else None, s1, lp1, rp1, tp1, bp1, blur=9,    smooth=31*its, its=its, compensate=True if its > 1 else False, device=device, fp16=fp16)
                    fclip = model(fclip, fref, fmask if mask else None, s2, lp2, rp2, tp2, bp2, blur=9,    smooth=15*its, its=its, compensate=True if its > 1 else False, device=device, fp16=fp16)
                    fclip = model(fclip, fref, fmask if mask else None, s3, lp3, rp3, tp3, bp3, blur=2,    smooth=7,      its=1,   compensate=False,                      device=device, fp16=fp16)
                    fclip = model(fclip, fref, fmask if mask else None, s4, lp4, rp4, tp4, bp4, blur=blur, smooth=smooth, its=1,   compensate=True,                       device=device, fp16=fp16)
                else:
                    raise ValueError("Precision must be 1, 2, 3, or 4.")
                
            fout = f[1].copy()
            tensor_to_frame(fclip, fout)
        return fout

    clip = core.std.ModifyFrame(clip=ref, clips=[clip, ref, mask] if mask else [clip, ref], selector=_align)
    if clip.format.id != clip_orig_format:
        return core.resize.Point(clip, format = clip_orig_format)
    return clip
