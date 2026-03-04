# vggt_align_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os

from external.vggt.models.vggt import VGGT

def mean_flat(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class ConvProjector2D(nn.Module):
    """
    minimal projector: projects to 24 maps of vggt
    """
    def __init__(self, in_channels: int, out_channels: int = 24, mid_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        #x: [B,C,H,W] -> [B,24,H,W]  interpolate to out_hw
        y = self.net(x)
        y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y


class VGGTGeometryForcingLoss(nn.Module):
    """
    GeometryForcing loss:
    - VGGT frozen
    - target: 24 aggregator outputs stacked to [BT,24,H,W]
    - pred: for each selected Qwen layer, projector(hidden) -> [BT,24,H,W]
    - loss: map-wise cosine alignment + optional scale recon
    """
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        vggt_img_size: int = 518,
        vggt_target_hw: tuple[int, int] = (1374, 2048),
        use_scale_recon: bool = False,
        scale_lambda: float = 1.0,
        projector_mid_channels: int = 128,
        qwen_layer_indices: list[int] | None = None,
        timestep_max_for_3d: int | None = None,
    ):
        super().__init__()
        self.vggt_target_hw = vggt_target_hw
        self.use_scale_recon = use_scale_recon
        self.scale_lambda = scale_lambda
        self.projector_mid_channels = projector_mid_channels
        self.qwen_layer_indices = qwen_layer_indices or []
        self.timestep_max_for_3d = timestep_max_for_3d

        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt.eval()
        for p in self.vggt.parameters():
            p.requires_grad_(False)
        self.vggt.to(device=device, dtype=torch.float32)

        self.projectors = nn.ModuleDict()

        if self.use_scale_recon:
            self.scale_head = nn.Sequential(
                nn.Conv2d(24, projector_mid_channels, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(projector_mid_channels, 24, kernel_size=1),
            )
        else:
            self.scale_head = None

        self.qwen_dtype = dtype

    @torch.no_grad()
    def _vggt_preprocess(self, images_btchw: torch.Tensor) -> torch.Tensor:
        """
        images_btchw: [B,T,3,H,W] in [0,1]
        """
        b, t, c, h, w = images_btchw.shape
        x = rearrange(images_btchw, "b t c h w -> (b t) c h w")
        x = F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False)
        x = torch.clamp(x, 0.0, 1.0)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)
        return x

    @torch.no_grad()
    def compute_vggt_target(self, gt_images_btc: torch.Tensor) -> torch.Tensor:
        vggt_dev = next(self.vggt.parameters()).device
        imgs = self._vggt_preprocess(gt_images_btc).to(vggt_dev, dtype=torch.float32)
        aggregated_list, patch_start_idx = self.vggt.shortcut_forward(imgs)

        b, t, _, h_in, w_in = imgs.shape
        patch_size = 14  # VGGT default
        hp = h_in // patch_size  # 518//14 = 37
        wp = w_in // patch_size  # 37

        tgt_h, tgt_w = self.vggt_target_hw
        maps = []

        for tok in aggregated_list:
            if tok.dim() != 4:
                raise RuntimeError(f"Expected tok [B,S,P,D], got {tok.shape}")

            tok = rearrange(tok, "b s p d -> (b s) p d")
            tok = tok[:, patch_start_idx:, :]

            tok = rearrange(tok, "bt (h w) d -> bt d h w", h=hp, w=wp)
            tok = tok.mean(dim=1, keepdim=True)

            tok = F.interpolate(tok, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
            maps.append(tok)

        target = torch.cat(maps, dim=1)
        return target

    def _get_or_make_projector(self, layer_idx: int, in_channels: int) -> nn.Module:
        key = str(layer_idx)
        if key in self.projectors:
            return self.projectors[key]
        proj = ConvProjector2D(in_channels=in_channels, out_channels=24, mid_channels=self.projector_mid_channels)
        self.projectors[key] = proj.to(device=next(self.parameters()).device, dtype=self.qwen_dtype)
        return self.projectors[key]

    def _tokens_to_grid(self, hid: torch.Tensor) -> torch.Tensor:
        """
        hid: [B, N, D]
        returns: [B, D, H, W]  (H*W = largest square wera using last H*W tokens)
        """
        b, n, d = hid.shape
        side = int((n ** 0.5))
        hw = side * side
        if hw < 1:
            raise RuntimeError(f"Cannot infer square grid from token length n={n}")

        img_tokens = hid[:, -hw:, :]
        grid = rearrange(img_tokens, "b (h w) d -> b d h w", h=side, w=side)
        return grid

    def forward(
        self,
        qwen_hidden_by_layer: dict[int, torch.Tensor],
        gt_images_btc: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        qwen_hidden_by_layer: {layer_idx: hidden}
          hidden expected as 4D feature map: [BT,C,H,W] or 5D [B,T,C,H,W]
        gt_images_btc: [B,T,3,H,W] (edited GT)
        timesteps optional: [BT] or [B] or scalar
        """
        if self.timestep_max_for_3d is not None and timesteps is not None:

            tmax = int(self.timestep_max_for_3d)
            try:
                t_val = timesteps.detach().view(-1)
                if torch.all(t_val > tmax):
                    return torch.zeros([], device=gt_images_btc.device, dtype=self.qwen_dtype)
            except Exception:
                pass

        target = self.compute_vggt_target(gt_images_btc)
        target = target.to(device=gt_images_btc.device, dtype=self.qwen_dtype)

        total = 0.0
        count = 0

        for layer_idx, hid in qwen_hidden_by_layer.items():
            if hid.dim() == 5:
                hid = rearrange(hid, "b t c h w -> (b t) c h w")

            if hid.dim() == 3:
                hid = self._tokens_to_grid(hid)

            if os.environ.get("RANK", "0") == "0":
                print("GRID HID:", hid.shape)

            if hid.dim() != 4:
                raise RuntimeError(f"Qwen hidden for layer {layer_idx} must be 3D/4D/5D, got {hid.shape}")

            proj = self._get_or_make_projector(layer_idx, in_channels=hid.shape[1])
            pred = proj(hid, out_hw=self.vggt_target_hw)

            pred_flat = rearrange(pred, "b c h w -> b c (h w)")
            tgt_flat  = rearrange(target, "b c h w -> b c (h w)")

            pred_n = F.normalize(pred_flat, p=2, dim=-1)
            tgt_n = F.normalize(tgt_flat, p=2, dim=-1)

            cos = (pred_n * tgt_n).sum(dim=-1)
            loss_map = 1.0 - cos
            loss = loss_map.sum(dim=-1)
            total = total + loss.mean()
            count += 1

            if self.use_scale_recon and self.scale_head is not None:
                pred_n_img = rearrange(pred_n, "b c (h w) -> b c h w", h=self.vggt_target_hw[0])
                recon = self.scale_head(pred_n_img)
                total = total + self.scale_lambda * F.mse_loss(recon, target)

        if count == 0:
            return torch.zeros([], device=gt_images_btc.device, dtype=self.qwen_dtype)
        return total / count