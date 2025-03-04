# Original Rife Frame Interpolation by hzwer
# https://github.com/megvii-research/ECCV2022-RIFE
# https://github.com/hzwer/Practical-RIFE

# Original XFeat Accelerated Features for Lightweight Image Matching
# https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
# https://github.com/verlab/accelerated_features

# Modifications by pifroggi
# or tepete on the "Enhance Everything!" Discord Server
# https://github.com/pifroggi/vs_align


import vapoursynth as vs
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
from .rife.IFNet_HDv3_v4_14_align import IFNet
from .xfeat.xfeat_align import XFeat, gen_grid
from .enums import SpatialPrecision, Device

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


def calculate_padding(height, width, modulus):
    pad_height = (modulus - height % modulus) % modulus
    pad_width  = (modulus - width % modulus) % modulus
    return pad_height, pad_width


def spatial(clip, ref, mask=None, precision=3, wide_search=False, lq_input=False, device="cuda"):

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
    blur   =  2 if lq_input else 0
    smooth = 11 if lq_input else 0
    fp16   = device == "cuda" and torch.cuda.get_device_capability()[0] >= 7
    
    # scales
    s1 = (16,  8,   4,   2   ) # needs mod64 pad
    s2 = (8,   4,   2,   1   ) # needs mod32 pad
    s3 = (4,   2,   1,   0.5 ) # needs mod16 pad
    s4 = (2,   1,   0.5, 0.25) # needs mod8  pad
    
    # padding for scales
    p1 = calculate_padding(ref.height, ref.width, 64)
    p2 = calculate_padding(ref.height, ref.width, 32)
    p3 = calculate_padding(ref.height, ref.width, 16)
    p4 = calculate_padding(ref.height, ref.width,  8)
    
    # initialize rife_align model
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_folder, "rife", "flownet_v4.14.pkl")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    rife_align = IFNet().to(device)
    rife_align.load_state_dict(state_dict, strict=False)
    rife_align.eval()
    if fp16:
        rife_align.half()
    
    # initialize xfeat model
    if wide_search:
        model_path = os.path.join(current_folder, "xfeat", "xfeat.pt")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        descriptor = XFeat(top_k=3000, weights=state_dict, height=480, width=704, device=device)
        matcher    = XFeat()
        if fp16:
            descriptor.half()
            matcher.half()
    
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


    def _align(n, f, p1, p2, p3, p4):
        with torch.inference_mode():
            fmask    = None
            fclip    = frame_to_tensor(f[0], device).clamp_(0, 1)
            fref     = frame_to_tensor(f[1], device).clamp_(0, 1)
            if mask:
                if n < mask.num_frames:
                    fmask = mask_to_tensor(f[2], device)
                else:
                    fmask = fmask_last # use cached last frame if mask clip is too short or just one frame

            # homography alignment with xfeat and cv2
            if wide_search:
                # detect points
                fref = F.pad(fref, (4, 4, 4, 4), mode="replicate") # pad to reduce border artifacts
                _, _, fref_h,  fref_w  = fref.shape  # update fref_h and fref_w due to padding
                _, _, fclip_h, fclip_w = fclip.shape # update fref_h and fref_w due to padding
                fclip_points  = descriptor.detectAndCompute(fclip)[0] # compute points
                fref_points   = descriptor.detectAndCompute(fref)[0]  # compute points
                kpts1, descs1 = fclip_points["keypoints"], fclip_points["descriptors"]
                kpts2, descs2 = fref_points["keypoints"],  fref_points["descriptors"]
                points_found  = len(kpts1) > 100 and len(kpts2) > 100
                
                if points_found: # only proceed if there are enough points
                    # match points between the two images
                    idx0, idx1  = matcher.match(descs1, descs2, 0.82)
                    match_found = len(idx0) > 50
                    
                    if match_found: # only proceed if there are enough matched points
                        # find homography from matched points with cv2
                        pts1 = kpts1[idx0].cpu().numpy()
                        pts2 = kpts2[idx1].cpu().numpy()
                        homography, _ = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 10.0, maxIters=1000, confidence=0.995) # find homography on cpu
                        homography_t  = torch.tensor(homography, dtype=torch.float32, device=device) # send back to gpu
                        homography_t  = gen_grid(homography_t, fclip_h, fclip_w, fref_h, fref_w, device) # generate grid for grid sample
                        if fp16:
                            homography_t = homography_t.half()
                        fclip = F.grid_sample(fclip, homography_t, mode="bicubic", padding_mode="border", align_corners=True) # warp
                        
                        # mask area outside the warped image
                        corners_src = np.float32([[0, 0],[fclip_w-1, 0],[fclip_w-1, fclip_h-1],[0, fclip_h-1]]).reshape(-1, 1, 2) # corners of source
                        corners_dst = cv2.perspectiveTransform(corners_src, homography) # transform corners with homography
                        homography_mask = np.ones((fref_h, fref_w), dtype=np.float32)   # create full white mask
                        cv2.fillConvexPoly(homography_mask, np.int32(corners_dst), 0.0) # fill inside corners with black
                        
                        # get bounding box
                        corners = corners_dst.reshape(-1, 2)
                        min_xy  = np.floor(corners.min(axis=0)).astype(int)
                        max_xy  = np.ceil(corners.max(axis=0)).astype(int)
                        min_x, min_y = np.maximum(min_xy, 0)
                        max_x, max_y = np.minimum(max_xy, [fref_w, fref_h])
                        
                        # avoid cropping to small value
                        if max_x - min_x < 16:
                            min_x, max_x = 0, 16
                        if max_y - min_y < 16:
                            min_y, max_y = 0, 16
                        
                        # convert to tensor and combine homography_mask with fmask if it exists
                        if mask:
                            homography_mask = torch.from_numpy(homography_mask)[None, None].to(device)
                            if fmask.shape[2] != fref_h or fmask.shape[3] != fref_w:
                                fmask = F.interpolate(fmask, size=(fref_h-8, fref_w-8), mode="nearest") # resize but compensate for pad from earlier
                                fmask = F.pad(fmask, (4, 4, 4, 4), mode="replicate")
                            fmask = fmask + homography_mask
                        else:
                            fmask = torch.from_numpy(homography_mask)[None, None].to(device)
                        
                        # crop to bounding box
                        fclip = fclip[:, :, min_y:max_y, min_x:max_x]
                        fref  =  fref[:, :, min_y:max_y, min_x:max_x]
                        if fmask is not None:
                            fmask = fmask[:, :, min_y:max_y, min_x:max_x]
                        
                # update padding for scales due to bounding box crop and border pad
                _, _, fref_h_new, fref_w_new = fref.shape
                if precision == 1 or precision == 2:
                    p1 = calculate_padding(fref_h_new, fref_w_new, 64)
                if precision > 1:
                    p2 = calculate_padding(fref_h_new, fref_w_new, 32)
                if precision > 2:
                    p3 = calculate_padding(fref_h_new, fref_w_new, 16)
                if precision > 3:
                    p4 = calculate_padding(fref_h_new, fref_w_new,  8)
            
            # flow based alignment with rife
            with torch.amp.autocast(device, enabled=fp16):
                if   precision == 1:
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s1, p1, blur=blur, smooth=smooth*3 if smooth>0 else 11, compensate=True, device=device, fp16=fp16)
                elif precision == 2:
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s1, p1, blur=9,    smooth=91,     compensate=False, device=device, fp16=fp16)
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s2, p2, blur=blur, smooth=smooth, compensate=True,  device=device, fp16=fp16)
                elif precision == 3:
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s2, p2, blur=9,    smooth=15,     compensate=False, device=device, fp16=fp16)
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s3, p3, blur=blur, smooth=smooth, compensate=True,  device=device, fp16=fp16)
                elif precision == 4:
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s2, p2, blur=9,    smooth=15,     compensate=False, device=device, fp16=fp16)
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s3, p3, blur=2,    smooth=7,      compensate=False, device=device, fp16=fp16)
                    fclip = rife_align(fclip, fref, fmask if fmask is not None else None, s4, p4, blur=blur, smooth=smooth, compensate=True,  device=device, fp16=fp16)
                else:
                    raise ValueError("Precision must be 1, 2, 3, or 4.")
                
            if wide_search:
                if points_found and match_found:
                    fclip = F.pad(fclip, (min_x, fref_w-max_x, min_y, fref_h-max_y), mode="replicate") # pad what was removed from boundry box crop
                fclip = fclip[:, :, 4:-4, 4:-4] # crop padding used to reduce border artifacts

            fout = f[1].copy()
            tensor_to_frame(fclip, fout)
        return fout

    clip = core.std.ModifyFrame(clip=ref, clips=[clip, ref, mask] if mask else [clip, ref], selector=partial(_align, p1=p1, p2=p2, p3=p3, p4=p4))
    if clip.format.id != clip_orig_format:
        return core.resize.Point(clip, format = clip_orig_format)
    return clip
