import torch
import torch.nn as nn

backwarp_tenGrid = {}

GRID_SAMPLE_FP32_PIXEL_THRESHOLD = 512 * 512


def _grid_sample_dtype(tenInput):
    return torch.float32 if tenInput.dtype == torch.float16 and tenInput.shape[2] * tenInput.shape[3] > GRID_SAMPLE_FP32_PIXEL_THRESHOLD else tenInput.dtype


def grid_sample(tenInput, tenGrid, mode='bicubic', padding_mode='border', align_corners=True):
    output_dtype = tenInput.dtype
    work_dtype   = _grid_sample_dtype(tenInput)
    tenOutput    = torch.nn.functional.grid_sample(input=tenInput.to(work_dtype), grid=tenGrid.to(work_dtype), mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return tenOutput.to(output_dtype)


def warp(tenInput, tenFlow, device, alpha=False):
    work_dtype = _grid_sample_dtype(tenInput)
    k = (str(tenFlow.device), str(tenFlow.size()), str(work_dtype))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], dtype=work_dtype, device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], dtype=work_dtype, device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = tenFlow.to(work_dtype)
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    if alpha:
        tenOutput = grid_sample(tenInput[:, :3],  g, mode='bicubic', padding_mode='border', align_corners=True)
        tenAlpha  = grid_sample(tenInput[:, 3:4], g, mode='nearest', padding_mode='zeros',  align_corners=True)
        tenOutput = torch.cat((tenOutput, tenAlpha), 1)
    else:
        tenOutput = grid_sample(tenInput, g, mode='bicubic', padding_mode='border', align_corners=True)
    #tenOutput = tenOutput.clamp(min=0.0, max=1.0) #changes output, without seems better
    return tenOutput
