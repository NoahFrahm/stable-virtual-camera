import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, preprocess_latent_tensors
import os



import torch
import torch.nn.functional as F
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class VGGTObjective:
    def __init__(
        self,
        input_image_folder, # Path to folder with input images
        ae, # SEVA autoencoder (with .decode)
        decoding_t: int = 1,
        warmup: int = 2,
        every: int = 2,
        step_size: float = 0.05,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Autoencoder
        self.ae = ae.to(self.device).eval()

        # VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)

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

    def _compute_all_relative(self, w2c: torch.Tensor):
        """Compute E_rel = E2 @ E1^{-1} for each consecutive pair."""
        rels = []
        for i in range(w2c.shape[0] - 1):
            E1, E2 = w2c[i], w2c[i + 1]
            rels.append(self._relative_transform(E1, E2))
        return rels

    @staticmethod
    def _relative_transform(E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
        """Given two 4×4 w2c poses, return the relative w2c_j @ inv(w2c_i)."""
        return E2 @ torch.linalg.inv(E1)

    @staticmethod
    def _rotation_error(R_pred: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
        """Squared geodesic rotation error between two 3×3 rotations."""
        M = R_pred.transpose(-1, -2) @ R_tgt
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        return torch.acos(((tr - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)).pow(2)

    @staticmethod
    def _translation_error(E_pred: torch.Tensor, E_tgt: torch.Tensor) -> torch.Tensor:
        """Squared translation error on camera centers: C = –R^T t."""
        R_pred, t_pred = E_pred[:3, :3], E_pred[:3, 3:4]
        R_tgt, t_tgt     = E_tgt[:3, :3], E_tgt[:3, 3:4]
        C_p = -(R_pred.transpose(-1, -2) @ t_pred)
        C_t = -(R_tgt.transpose(-1, -2) @ t_tgt)
        return (C_p - C_t).pow(2).sum()
    
    def preprocess_images(self, decoded_latent):
        # Preprocess decoded latent
        processed_decoded_latent = preprocess_latent_tensors(decoded_latent)

        # Combine processed latents and input images
        all_images = torch.cat([self.images, processed_decoded_latent], dim=0)

        return all_images        

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, step: int, **kwargs) -> torch.Tensor:
        # Skip early or infrequent steps
        if step < self.warmup or ((step + 1) % self.every) != 0:
            return x

        # Make latents a leaf
        x = x.detach().requires_grad_(True)

        # Decode latent into image
        decoded_latent = self.ae.decode(x, self.decoding_t).clamp(-1, 1)

        # preprocess decoded latent
        images = self.preprocess_images(decoded_latent)
        
        # VGGT forward
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, _ = self.model.aggregator(images)
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])


        # Compute total relative‐transform loss
        total_loss = torch.tensor(0.0, device=self.device)
        for i, E_rel_tgt in enumerate(self.rel_targets):
            E_rel_pred = self._relative_transform(extrinsic[i], extrinsic[i + 1])
            rot_err   = self._rotation_error(E_rel_pred[:3, :3], E_rel_tgt[:3, :3])
            trans_err = self._translation_error(E_rel_pred, E_rel_tgt)
            total_loss = total_loss + rot_err + trans_err

        loss = total_loss / len(self.rel_targets)

        # Backprop → latent update
        grad, = torch.autograd.grad(loss, x)
        norm = grad.norm().detach().clamp_min(1e-8)
        with torch.no_grad():
            x = (x - self.step_size * grad / norm).detach()
        return x
    

