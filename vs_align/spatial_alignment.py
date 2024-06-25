
# Original Rife Frame Interpolation by hzwer
# https://github.com/megvii-research/ECCV2022-RIFE
# https://github.com/hzwer/Practical-RIFE

# Modifications to use Rife for Image Alignment by pifroggi
# or tepete on the "Enhance Everything!" Discord Server

# Additional helpful github issues
# https://github.com/megvii-research/ECCV2022-RIFE/issues/278
# https://github.com/megvii-research/ECCV2022-RIFE/issues/


import vapoursynth as vs
import os
import torch
import numpy as np
import torch.nn.functional as F
from .enums import SpatialPrecision, Device
from .rife.IFNet_HDv3_v4_14_align import IFNet

core = vs.core

def tensor_to_frame(img: torch.Tensor, frame: vs.VideoFrame):
    img_np = img.permute(1, 2, 0).cpu().numpy()
    for p in range(3):
        np.copyto(np.asarray(frame[p]), img_np[:, :, p])

def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    frame_np = np.dstack([np.asarray(frame[p]) for p in range(frame.format.num_planes)])
    return torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device)

def calculate_padding(height, width, padding):
    pad_height = (padding - height % padding) % padding
    pad_width = (padding - width % padding) % padding
    return pad_height, pad_width

def spatial(clip, ref, precision=3, iterations=1, blur_strength=0, ensemble=True, device="cuda"):
    if isinstance(device, Device):
        device = device.value
    device = torch.device(device)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_folder, 'rife', 'flownet_v4.15.pkl')
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model = IFNet().to(device)
    model.load_state_dict(new_state_dict)
    model.eval()

    precision_value_map = {
        1: (2,    64), #50%
        2: (1,    32), #100%
        3: (0.5,  16), #200%
        4: (0.25,  8), #400%
        5: (0.125, 4)  #800%
    }

    if isinstance(precision, SpatialPrecision):
        precision = precision.value
    if precision not in precision_value_map:
        raise ValueError("Precision must be 1, 2, 3, 4, or 5.")

    multiplier, padding = precision_value_map[precision]

    def align(n, f):
        fclip = frame_to_tensor(f[0], device)
        fref = frame_to_tensor(f[1], device)
        _, _, h, w = fref.shape

        #resize frame to reference frame
        if fclip.shape != fref.shape:
            fclip = torch.nn.functional.interpolate(fclip, size=(h, w), mode='bicubic', align_corners=False)

        #calculate and apply padding based on precision
        pad_h, pad_w = calculate_padding(h, w, padding)
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad
        fref_padded = torch.nn.functional.pad(fref, (left_pad, right_pad, top_pad, bottom_pad), mode='replicate')
        fclip_padded = torch.nn.functional.pad(fclip, (left_pad, right_pad, top_pad, bottom_pad), mode='replicate')
        fref_padded.clamp_(0, 1)
        fclip_padded.clamp_(0, 1)

        #align
        with torch.no_grad():
            aligned_img0, _ = model(fclip_padded, fref_padded, multiplier=multiplier, num_iterations=iterations, blur_strength=blur_strength, ensemble=ensemble, device=device)

        #crop
        output_img_cropped = aligned_img0.squeeze(0)[:, top_pad:top_pad+h, left_pad:left_pad+w]

        fout = f[1].copy()
        tensor_to_frame(output_img_cropped, fout)

        return fout

    if clip.format.id != vs.RGBS or clip.format.sample_type != vs.FLOAT:
        raise ValueError("clip must be in RGBS format.")
    if ref.format.id != vs.RGBS or ref.format.sample_type != vs.FLOAT:
        raise ValueError("ref must be in RGBS format.")

    return core.std.ModifyFrame(clip=ref, clips=[clip, ref], selector=align)
