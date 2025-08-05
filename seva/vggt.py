import os
import sys

vggt_repo = "/playpen-nas-ssd4/nofrahm/Multi-View-Gen/vggt"
vggt_repo_path = os.path.abspath(vggt_repo)
if not os.path.isdir(vggt_repo_path):
    raise RuntimeError(f"vggt path not found: {vggt_repo_path}")
if vggt_repo_path not in sys.path: sys.path.insert(0, vggt_repo_path)


import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from torch.autograd.forward_ad import dual_level, make_dual

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_latent_tensors
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTObjective:
    def __init__(
        self,
        input_image_folder, # Path to folder with input images
        ae, # SEVA autoencoder (with .decode)
        rel_gt: torch.Tensor, # (N_in,4,4) GT relative w2c from gen->input_i
        decoding_t: int = 1,
        # warmup: int = 2,
        warmup: int = 0,
        every: int = 2,
        step_size: float = 0.05,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        # print(self.dtype)
        # self.dtype = torch.bfloat16

        # Autoencoder
        self.ae = ae
        for p in self.ae.parameters():  p.requires_grad_(False)

        # VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        for p in self.model.parameters(): p.requires_grad_(False)

        # Input images
        image_names = []
        for file in os.listdir(input_image_folder):
            if file.endswith(".png") or file.endswith(".jpg"):
                image_names.append(os.path.join(input_image_folder, file))
        images = load_and_preprocess_images(image_names).to(device)

        self.decoding_t = decoding_t
        self.warmup = warmup
        self.every = every
        self.step_size = step_size
        self.images = images
        self.relative_transform_gt = rel_gt[0].to(device)

    @staticmethod
    def _relative_transform(E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
        """Given two 4x4 w2c poses, return the relative w2c_j @ inv(w2c_i)."""
        return E2 @ torch.linalg.inv(E1)

    @staticmethod
    def _rotation_error(R_pred: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
        """Squared geodesic rotation error between two 3×3 rotations."""
        M = R_pred.transpose(-1, -2) @ R_tgt
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        return torch.acos(((tr - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)).pow(2)

    @staticmethod
    def _translation_error(E_pred: torch.Tensor, E_tgt: torch.Tensor) -> torch.Tensor:
        """Squared translation error on camera centers: C = -R^T t."""
        R_pred, t_pred = E_pred[:3, :3], E_pred[:3, 3:4]
        R_tgt, t_tgt     = E_tgt[:3, :3], E_tgt[:3, 3:4]
        C_p = -(R_pred.transpose(-1, -2) @ t_pred)
        C_t = -(R_tgt.transpose(-1, -2) @ t_tgt)
        return (C_p - C_t).pow(2).sum()
    
    @staticmethod
    def _to_homogeneous(E):
        bottom = torch.tensor([[0, 0, 0, 1.]], device=E.device, dtype=E.dtype)
        return torch.cat([E, bottom], dim=0) # (4,4)
    
    def preprocess_images(self, decoded_latent):
        processed_decoded_latent = preprocess_latent_tensors(decoded_latent)
        generated_images = processed_decoded_latent.to(self.device, dtype=self.dtype)
        input_images = self.images.to(self.device, dtype=self.dtype)

        all_images = torch.cat([input_images, generated_images], dim=0)

        return all_images

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, step: int, lr = 1e-2, **kwargs) -> torch.Tensor:
        # Skip early or infrequent steps
        if step < self.warmup or ((step + 1) % self.every) != 0:
            return x

        # Make latents a leaf
        x = x.detach().requires_grad_(True)

        # This runs your entire decode(x, t) without storing intermediates,
        # then re-runs it on backward to get grads w.r.t. x.
        with torch.enable_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            # Auto encoder
            decoded = checkpoint(self.ae.decode, x, self.decoding_t, use_reentrant=False)
            images = self.preprocess_images(decoded)[None]

            print("images shape:", images.shape)

            # VGGT forward
            # aggregated_tokens_list, _ = checkpoint(self.model.aggregator, images)
            aggregated_tokens_list, _ = self.model.aggregator(images)

            # Compute loss
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            extrinsic = extrinsic[0]
            print(f"Step {step}: extrinsic shape: {extrinsic.shape}")
            
            # Compute total relative‐transform loss
            total_loss = 0.0
            for i in range(extrinsic.shape[0] - 1):
                E_input, E_gen = self._to_homogeneous(extrinsic[i]), self._to_homogeneous(extrinsic[-1])
                E_rel_pred = self._relative_transform(E_input, E_gen)
                rot_err   = self._rotation_error(E_rel_pred[:3, :3], self.relative_transform_gt[:3, :3])
                trans_err = self._translation_error(E_rel_pred, self.relative_transform_gt)
                total_loss = total_loss + rot_err + trans_err
                break

            loss = total_loss / len(self.relative_transform_gt)
            print("\ngradients enabled")
            print("Tokens:", aggregated_tokens_list[0].requires_grad)
            print("pose_enc:", pose_enc.requires_grad)
            print("extrinsic:", extrinsic.requires_grad)
            print("Loss:", loss.requires_grad)

        # Backprop -> latent update
        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad # gradient descent step
        x.grad.zero_() # clear grad for the next iteration

        breakpoint()

        return x


# NOTE: if needed it is available here
# def preprocess_latent_tensors(
#         imgs: Union[torch.Tensor, List[torch.Tensor]],
#         *,                       # keyword-only for clarity
#         mode: str = "crop",      # "crop" (default) or "pad"
#         target_size: int = 518,  # longest dimension ≤ 518 after resize
#         patch: int = 14,         # make H and W multiples of 14
# ) -> torch.Tensor:
#     """
#     Pre-process decoded latent RGB tensors for VGGT.

#     Args
#     ----
#     imgs : Tensor or list[Tensor]
#         • Single tensor  (3,H,W)  or  (N,3,H,W)  
#         • or list of tensors, each  (3,H,W)
#         Values may be in [-1,1] (as from VAE decode) or [0,1].

#     mode : {"crop","pad"}
#         Identical semantics to the file-based helper:
#         - "crop": isotropically resize so **width == 518 px**,
#                   then centre-crop height if it exceeds 518 px.
#         - "pad" : isotropically resize so **long side == 518 px**,
#                   then pad the short side with white (1.0) to square.

#     target_size : int
#         Maximum size after resize (default 518 px).

#     patch : int
#         Final H and W are forced to be multiples of this value (14).

#     Returns
#     -------
#     torch.Tensor  with shape (N, 3, H', W')
#     """

#     if isinstance(imgs, torch.Tensor):
#         imgs = [imgs] if imgs.dim() == 3 else list(imgs)  # (3,H,W) or (N,3,H,W)
#     if len(imgs) == 0:
#         raise ValueError("At least one tensor is required")

#     if mode not in {"crop", "pad"}:
#         raise ValueError('mode must be "crop" or "pad"')

#     processed = []
#     for im in imgs:
#         if im.min() < 0: # latent decode gives [-1,1]
#             im = (im.clamp(-1, 1) + 1) / 2  # -> [0,1]

#         # ensure float32 (AMP will cast later)
#         im = im.float()

#         C, H, W = im.shape
#         long_side = max(H, W)

#         # Resize to target size while maintaining aspect ratio
#         if mode == "crop":
#             new_W = target_size
#             scale = new_W / W
#             new_H = round(H * scale)
#         else: # "pad"
#             if long_side > target_size:
#                 scale = target_size / long_side
#                 new_H = round(H * scale)
#                 new_W = round(W * scale)
#             else:
#                 new_H, new_W = H, W

#         # keep result divisible by 14 px
#         new_H = ((new_H + patch - 1) // patch) * patch
#         new_W = ((new_W + patch - 1) // patch) * patch

#         if (new_H, new_W) != (H, W):
#             im = F.interpolate(
#                 im.unsqueeze(0), (new_H, new_W),
#                 mode="bilinear", align_corners=False, antialias=True
#             ).squeeze(0)

#         # Crop or pad to square ----------
#         if mode == "crop":
#             if new_H > target_size:
#                 top = (new_H - target_size) // 2
#                 im = im[:, top:top+target_size, :]
#         else:  # "pad"
#             pad_H = target_size - im.shape[1]
#             pad_W = target_size - im.shape[2]
#             if pad_H > 0 or pad_W > 0:
#                 pad_top, pad_bottom = pad_H // 2, pad_H - pad_H // 2
#                 pad_left, pad_right = pad_W // 2, pad_W - pad_W // 2
#                 # constant-value padding with 1.0 (white)
#                 im = F.pad(
#                     im, (pad_left, pad_right, pad_top, pad_bottom),
#                     mode="constant", value=1.0
#                 )

#         # Final divisibility by patch
#         Hf, Wf = im.shape[-2:]
#         pad_H = (patch - Hf % patch) % patch
#         pad_W = (patch - Wf % patch) % patch
#         if pad_H or pad_W:
#             im = F.pad(im, (0, pad_W, 0, pad_H), mode="constant", value=1.0)

#         processed.append(im)

#     images = torch.stack(processed, 0)
#     return images
