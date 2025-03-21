# Original XFeat Accelerated Features for Lightweight Image Matching
# https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
# https://github.com/verlab/accelerated_features

# Modifications for homography alignment by pifroggi
# or tepete on the "Enhance Everything!" Discord Server
# https://github.com/pifroggi/vs_align


import numpy as np
import torch
import torch.nn.functional as F
from .xfeat_arch import *


def gen_grid(homography, src_h, src_w, out_h, out_w, device):
    # generates the grid from homograhpy to warp with grid_sample
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, out_h - 1, out_h, device=device),
        torch.linspace(0, out_w - 1, out_w, device=device),
        indexing="ij"  # make sure first dimension is y and second is x
    )
    grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1) # stack grids
    grid_flat = grid.view(-1, 3).T # flatten grid
    homography_inv = torch.inverse(homography) # inverse homography
    src_coords = homography_inv @ grid_flat # apply homography to grid
    src_coords = src_coords / src_coords[2:3, :] # mormalize
    src_coords = src_coords[:2, :].view(2, out_h, out_w).permute(1, 2, 0) #reshape to (out_h, out_w, 2)
    src_coords[..., 0] = 2.0 * src_coords[..., 0] / (src_w - 1) - 1 # normalize to [-1, 1] for grid_sample
    src_coords[..., 1] = 2.0 * src_coords[..., 1] / (src_h - 1) - 1 # normalize to [-1, 1] for grid_sample
    return src_coords.unsqueeze(0)


class XFeat(nn.Module):
    # XFeat inference module
    def __init__(self, weights=None, top_k=4096, detection_threshold=0.05, height=480, width=704, device="cuda"):
        super().__init__()
        self.dev = torch.device(device)
        self.net = XFeatModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        self.height = height
        self.width = width
        if weights is not None:
            self.net.load_state_dict(weights)
        self.interpolator = InterpolateSparse2d("bicubic")


    @torch.inference_mode()
    def detectAndCompute(self, x, top_k=None, detection_threshold=None, height=None, width=None):
        # computes sparse keypoints & descriptors
        if top_k is None: top_k = self.top_k
        if detection_threshold is None: detection_threshold = self.detection_threshold
        if height is None: height = self.height
        if width  is None: width  = self.width
        x, rh1, rw1 = self.preprocess_tensor(x, height, width)
        B, _, _H1, _W1 = x.shape
        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        # convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)

        # compute reliability scores
        _nearest = InterpolateSparse2d("nearest")
        _bilinear = InterpolateSparse2d("bilinear")
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        # select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :top_k]
        mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]
        
        feats = self.interpolator(M1, mkpts, H = _H1, W = _W1) # interpolate descriptors at kpts positions
        feats = F.normalize(feats, dim=-1) # L2-Normalize
        mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1) # correct kpt scale
        valid = scores > 0
        return [  
                   {"keypoints": mkpts[b][valid[b]],
                    "scores": scores[b][valid[b]],
                    "descriptors": feats[b][valid[b]]} for b in range(B) 
               ]


    def preprocess_tensor(self, x, target_height=480, target_width=704):
        # resize to common resolution (must be divisible by 32)
        H, W = x.shape[-2:]
        if H != target_height or W != target_width:
            rh, rw = H / target_height, W / target_width
            x = F.interpolate(x, (target_height, target_width), mode="bilinear", align_corners=False)
        else:
            rh, rw = 1.0, 1.0
        return x, rh, rw


    def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap


    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        # pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos


    @torch.inference_mode()
    def match(self, feats1, feats2, min_cossim = 0.82):
        # matches two sets of points
        cossim   = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)
        idx0   = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]
        return idx0, idx1


class InterpolateSparse2d(nn.Module):
    # interpolate tensor at given sparse 2D positions
    def __init__(self, mode = "bicubic", align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)
