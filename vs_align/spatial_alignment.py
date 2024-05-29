
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
import cv2
from enum import Enum

from .rife.IFNet_HDv3_v4_14_align import IFNet

core = vs.core

def array_to_frame(img: np.ndarray, frame: vs.VideoFrame):
    for p in range(3):
        pls = frame[p]
        frame_arr = np.asarray(pls)
        np.copyto(frame_arr, img[:, :, p])

def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    return np.dstack([np.asarray(frame[p]) for p in range(frame.format.num_planes)])

class PrecisionMode(Enum):
    FIFTY_PERCENT = 2000
    ONE_HUNDRED_PERCENT = 1000
    TWO_HUNDRED_PERCENT = 500
    FOUR_HUNDRED_PERCENT = 250
    EIGHT_HUNDRED_PERCENT = 125

def calculate_padding(height, width, precision):
    if precision == PrecisionMode.EIGHT_HUNDRED_PERCENT:
        pad_value = 4
    elif precision == PrecisionMode.FOUR_HUNDRED_PERCENT:
        pad_value = 8
    elif precision == PrecisionMode.TWO_HUNDRED_PERCENT:
        pad_value = 16
    elif precision == PrecisionMode.ONE_HUNDRED_PERCENT:
        pad_value = 32
    else:
        pad_value = 64
    
    pad_height = (pad_value - height % pad_value) % pad_value
    pad_width = (pad_value - width % pad_value) % pad_value
    return pad_height, pad_width

def spatial(clip, ref, precision="100", iterations=1, blur_strength=0, ensemble=True, device="cuda"):
    device = torch.device(device)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_folder, 'rife', 'flownet_v4.14.pkl')
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model = IFNet().to(device)
    model.load_state_dict(new_state_dict)
    model.eval()

    #convert precision to PrecisionMode enum
    precision_value = int(precision)
    precision_enum_map = {
        50: PrecisionMode.FIFTY_PERCENT,
        100: PrecisionMode.ONE_HUNDRED_PERCENT,
        200: PrecisionMode.TWO_HUNDRED_PERCENT,
        400: PrecisionMode.FOUR_HUNDRED_PERCENT,
        800: PrecisionMode.EIGHT_HUNDRED_PERCENT
    }

    enum_precision = precision_enum_map.get(precision_value, PrecisionMode.ONE_HUNDRED_PERCENT)
    multiplier = enum_precision.value / 1000

    def align(n, f):
    
        frame1 = frame_to_array(f[0])
        frame2 = frame_to_array(f[1])
        h, w, _ = frame1.shape

        #resize and shift target image to match source dimensions
        frame2_resized = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_LANCZOS4)
        frame2_resized = np.roll(frame2_resized, -1, axis=1)
        frame2_resized[:, -1] = frame2_resized[:, -2]

        #calculate and apply padding based on precision
        pad_h, pad_w = calculate_padding(h, w, precision)
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad
        frame2_padded = np.pad(frame2_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='edge')
        frame1_padded = np.pad(frame1, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='edge')

        #convert to tensors, concatenate, and align
        img0 = torch.from_numpy(frame1_padded).permute(2, 0, 1).unsqueeze(0).to(device)
        img1 = torch.from_numpy(frame2_padded).permute(2, 0, 1).unsqueeze(0).to(device)
        x = torch.cat((img0, img1), 1)

        with torch.no_grad():
            aligned_img0, _ = model(x, multiplier=multiplier, num_iterations=iterations, blur_strength=blur_strength, ensemble=ensemble)

        #prepare and return the aligned output
        output_img = aligned_img0.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_img_cropped = output_img[top_pad:top_pad+h, left_pad:left_pad+w]

        fout = f[0].copy()
        array_to_frame(output_img_cropped, fout)

        return fout

    if clip.format.id != vs.RGBS or clip.format.sample_type != vs.FLOAT:
        raise ValueError("clip must be in RGBS format.")
    if ref.format.id != vs.RGBS or ref.format.sample_type != vs.FLOAT:
        raise ValueError("ref must be in RGBS format.")

    return core.std.ModifyFrame(clip=clip, clips=[clip, ref], selector=align)
