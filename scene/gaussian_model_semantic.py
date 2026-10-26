from pathlib import Path
import torch
from scene.gaussian_model import GaussianModel


class GaussianModelSemantic(GaussianModel):
    def load_ply(self, path):
        super().load_ply(path)
        sidecar = Path(path).with_name("language_features_3d.pt")
        if not sidecar.is_file():
            sidecar = Path(path).parents[2] / "language_features_3d.pt"
        if not sidecar.is_file():
            raise FileNotFoundError(f"Language features missing: {sidecar}")
        self._language_feature = torch.load(sidecar, map_location="cuda")["language_features"].cuda()
        if len(self._language_feature) != len(self.get_xyz):
            raise ValueError("Language feature count differs from Gaussian count")

    @property
    def get_language_feature(self):
        return self._language_feature
