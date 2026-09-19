"""Small optional projection head; training loops stay explicit and inspectable."""
import json
import numpy as np
from .torch_geometry import torch, parameters_from_latent, tensor_matrix
from torch import nn
from .geometry import Geometry, geometry_from_spec
from .embedding import Embedding, _pca, _strings, _write_npz


class TrainableProjector(nn.Module):
    """Frozen input embeddings -> MLP -> geometry parameters.

    forward returns parameter coordinates for losses. Use the same head for the
    query. Fit input normalization on training rows only; fit display after
    optimization and before exporting held-out embeddings.
    """
    def __init__(self, input_dim, geometry, hidden_dim=64):
        super().__init__()
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be a Geometry")
        if not isinstance(input_dim, int) or not isinstance(hidden_dim, int) or min(input_dim, hidden_dim) < 1:
            raise ValueError("input_dim and hidden_dim must be positive integers")
        self.geometry, self.input_dim, self.hidden_dim = geometry, input_dim, hidden_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, geometry.intrinsic_dim))
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        self.display_mean_ = self.display_components_ = None

    def fit_normalization(self, train_x):
        tensor_matrix(train_x, self.input_dim)
        with torch.no_grad():
            self.input_mean.copy_(train_x.mean(0))
            self.input_scale.copy_(train_x.std(0, unbiased=False).clamp_min(1e-6))
        self.display_mean_ = self.display_components_ = None
        return self

    def forward(self, x):
        tensor_matrix(x, self.input_dim)
        return parameters_from_latent(self.geometry, self.net((x-self.input_mean)/self.input_scale))

    def fit_display(self, train_x):
        with torch.no_grad():
            p = self(train_x).detach().cpu().numpy()
        z = self.geometry.from_parameters(p)
        self.display_mean_, self.display_components_, _ = _pca(z, 3)
        return self

    def transform(self, x, texts=None):
        if self.display_mean_ is None:
            raise RuntimeError("Call fit_display on training rows after optimization")
        with torch.no_grad():
            p = self(x).detach().cpu().numpy()
        source = x.detach().cpu().numpy().copy()
        return Embedding(self.geometry, self.geometry.from_parameters(p), p, source,
                         _strings(texts, len(p)), self.display_mean_.copy(), self.display_components_.copy(),
                         mapping="trained MLP + parametrization")

    def save(self, path):
        if self.display_mean_ is None:
            raise RuntimeError("Fit the display transform before saving")
        arrays = {f"state_{k}": v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        metadata = {"format": "manifold-studio-trained-projector", "version": 1,
                    "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
                    "geometry": self.geometry.spec(), "dtype": str(self.input_mean.dtype)}
        _write_npz(path, **arrays, display_mean=self.display_mean_, display_components=self.display_components_,
                   metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        """Restore weights without pickle; returns a CPU model in evaluation mode."""
        with np.load(path, allow_pickle=False) as b:
            m = json.loads(str(b["metadata"]))
            if m.get("format") != "manifold-studio-trained-projector" or m.get("version") != 1:
                raise ValueError("Unsupported trained projector")
            if m.get("dtype") not in ("torch.float32", "torch.float64"):
                raise ValueError("Unsupported model dtype")
            # Loading must not consume the caller's training RNG sequence.
            with torch.random.fork_rng(devices=[]):
                obj = cls(m["input_dim"], geometry_from_spec(m["geometry"]), m["hidden_dim"])
            obj = obj.double() if m["dtype"] == "torch.float64" else obj.float()
            state = {k: torch.from_numpy(b[f"state_{k}"].copy()) for k in obj.state_dict()}
            if any(not torch.isfinite(v).all() for v in state.values()) or (state["input_scale"] <= 0).any():
                raise ValueError("Invalid model weights or input scale")
            obj.load_state_dict(state)
            obj.display_mean_, obj.display_components_ = b["display_mean"].copy(), b["display_components"].copy()
            a = obj.geometry.ambient_dim
            if obj.display_mean_.shape != (a,) or obj.display_components_.shape != (3, a) or not np.isfinite(obj.display_mean_).all() or not np.isfinite(obj.display_components_).all():
                raise ValueError("Invalid display transform")
        return obj.eval()
