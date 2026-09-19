"""Selectable PyTorch losses adapted from Vinit's research; see docs/LOSSES.md.

Import this module explicitly; the NumPy-only core does not import PyTorch.
All losses consume manifold *parameters*, not 3D display coordinates.
"""
import math
from dataclasses import dataclass
from .torch_geometry import torch, tensor_matrix, coordinates, distances
from torch import nn
from torch.nn import functional as F


LOSS_CATALOG = {
    "euclidean_triplet": "Squared ambient triplet; adapts manifold-embedding-nlp/train.py",
    "geodesic_triplet": "Intrinsic triplet; requires a supported intrinsic geometry",
    "ambient_triplet": "Unsquared ambient triplet; explicit option for Möbius",
    "distance_alignment": "Normalized pairwise distance alignment; SFM curvature-alignment adaptation",
    "query_energy_separation": "High-frequency query-field energy fraction; experimental SFM adaptation",
    "energy_concentration": "Normalized spectral energy entropy; experimental regularizer",
    "smtl": "Triplet + weighted separation + entropy + alignment; experimental repaired SMTL",
}


def available_losses():
    return dict(LOSS_CATALOG)


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _indices(t, n, device):
    if t is None:
        raise ValueError("Provide triplets as a nonempty T x 3 integer tensor")
    t = torch.as_tensor(t, device=device)
    if t.ndim != 2 or t.shape[1] != 3 or not len(t) or t.dtype not in (torch.int32, torch.int64):
        raise ValueError("triplets must be a nonempty T x 3 integer tensor")
    if (t < 0).any() or (t >= n).any() or (t[:, 0] == t[:, 1]).any() or (t[:, 0] == t[:, 2]).any() or (t[:, 1] == t[:, 2]).any():
        raise ValueError("Triplet indices must be in range and distinct within each triplet")
    return t.long()


@dataclass
class LossResult:
    total: torch.Tensor
    components: dict

    def detached(self):
        return {**{k: float(v.detach()) for k, v in self.components.items()}, "total": float(self.total.detach())}


def _cosine_distances(source):
    source = tensor_matrix(source, name="source")
    if (torch.linalg.vector_norm(source, dim=1) == 0).any():
        raise ValueError("Source cosine distances require nonzero vectors")
    x = F.normalize(source.detach(), dim=1)
    return (1-x @ x.T).clamp(0, 2)


def fixed_spectral_basis(source):
    """Full detached basis of a dense RBF normalized Laplacian, low to high λ.

    Build once on a training document/batch (3..512 rows), retain its row order.
    Uses source cosine distances and their median as bandwidth. No gradients
    through the graph, eigenvectors or bandwidth. Dense cost is O(N^3).
    """
    source = tensor_matrix(source, name="source")
    n = len(source)
    if not 3 <= n <= 512:
        raise ValueError("Spectral batches must contain 3..512 rows")
    with torch.no_grad():
        d = _cosine_distances(source)
        upper = d[torch.triu_indices(n, n, 1, device=d.device).unbind()]
        if upper.max() <= 1e-7:
            raise ValueError("A constant source batch does not define a useful spectral graph")
        sigma = upper[upper > 1e-7].median().clamp_min(1e-6)
        w = torch.exp(-0.5*(d/sigma).square())
        w.fill_diagonal_(0)
        inv = w.sum(1).clamp_min(torch.finfo(w.dtype).tiny).rsqrt()
        lap = torch.eye(n, dtype=w.dtype, device=w.device)-inv[:, None]*w*inv[None, :]
        _, basis = torch.linalg.eigh((lap+lap.T)/2)
    return basis.detach()


class GeometryLoss(nn.Module):
    def __init__(self, name, geometry, *, metric=None, margin=0.3,
                 alpha=1.0, beta=0.5, gamma=0.3, temperature=1.0, noise_fraction=1/3):
        super().__init__()
        if name not in LOSS_CATALOG:
            raise ValueError(f"Unknown loss {name!r}; choose from {', '.join(LOSS_CATALOG)}")
        metric = ("ambient" if name in ("euclidean_triplet", "ambient_triplet") else "intrinsic") if metric is None else metric
        if metric not in ("intrinsic", "ambient"):
            raise ValueError("metric must be intrinsic or ambient")
        if name == "geodesic_triplet" and metric != "intrinsic":
            raise ValueError("geodesic_triplet requires intrinsic distance; use ambient_triplet")
        if name in ("euclidean_triplet", "ambient_triplet") and metric != "ambient":
            raise ValueError(f"{name} requires metric='ambient'")
        if metric == "intrinsic" and not geometry.supports_intrinsic:
            raise NotImplementedError("Möbius needs an explicit ambient loss/metric")
        for key, value in dict(margin=margin, alpha=alpha, beta=beta, gamma=gamma).items():
            _nonnegative(value, key)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not 0 < noise_fraction < 1:
            raise ValueError("noise_fraction must lie strictly between zero and one")
        self.name, self.geometry, self.metric = name, geometry, metric
        self.margin, self.alpha, self.beta, self.gamma = margin, alpha, beta, gamma
        self.temperature, self.noise_fraction = temperature, noise_fraction

    def forward(self, parameters, *, triplets=None, source=None, query_parameters=None, basis=None):
        p = tensor_matrix(parameters, self.geometry.intrinsic_dim, "parameters")
        n = len(p)
        if n > 512:
            raise ValueError("Loss batches are limited to 512 rows; use smaller training batches")
        parts = {}
        needs_triplet = self.name.endswith("triplet") or self.name == "smtl"
        needs_alignment = self.name == "distance_alignment" or (self.name == "smtl" and self.gamma > 0)
        needs_spectral = self.name in ("query_energy_separation", "energy_concentration") or (self.name == "smtl" and (self.alpha > 0 or self.beta > 0))
        d = distances(self.geometry, p, metric=self.metric) if needs_triplet or needs_alignment else None
        if needs_triplet:
            t = _indices(triplets, n, p.device)
            dp, dn = d[t[:, 0], t[:, 1]], d[t[:, 0], t[:, 2]]
            if self.name == "euclidean_triplet":
                dp, dn = dp.square(), dn.square()
            parts["triplet"] = F.relu(dp-dn+self.margin).mean()
        if needs_alignment:
            if source is None or len(source) != n:
                raise ValueError("distance_alignment requires source vectors in the same row order")
            tensor_matrix(source, name="source")
            if source.device != p.device or source.dtype != p.dtype:
                raise ValueError("source and parameters must share dtype and device")
            if n < 3:
                raise ValueError("distance_alignment needs at least three rows")
            i, j = torch.triu_indices(n, n, 1, device=p.device)
            a, b = d[i, j], _cosine_distances(source)[i, j]
            if b.max() <= 1e-7:
                raise ValueError("distance_alignment requires nonconstant source cosine distances")
            # Preserve zeros and avoid min-max degeneracy; source is a fixed teacher.
            parts["distance_alignment"] = ((a/a.max().clamp_min(1e-7))-(b/b.max())).square().mean()
        if needs_spectral:
            if basis is None or query_parameters is None:
                raise ValueError("Spectral losses need a fixed basis and one query_parameters row")
            tensor_matrix(basis, n, "basis")
            if basis.shape != (n, n) or n < 3:
                raise ValueError("basis must be a full N x N orthonormal eigenbasis for N >= 3")
            q = tensor_matrix(query_parameters, self.geometry.intrinsic_dim, "query_parameters")
            if len(q) != 1:
                raise ValueError("Pass exactly one query per document/batch")
            if any(x.device != p.device or x.dtype != p.dtype for x in (basis, q)):
                raise ValueError("parameters, query and basis must share dtype and device")
            u = basis.detach()
            eye = torch.eye(n, dtype=p.dtype, device=p.device)
            if not torch.allclose(u.T @ u, eye, atol=2e-4, rtol=2e-4):
                raise ValueError("basis columns must be orthonormal, ordered low to high eigenvalue")
            # Same learned head for documents and query; manifold distance excitation.
            g = torch.softmax(-distances(self.geometry, p, q, metric=self.metric)[:, 0]/self.temperature, dim=0)
            field = g[:, None]*coordinates(self.geometry, p)
            energies = (u.T @ field).square().sum(1)
            total = energies.sum()
            if total.detach() <= torch.finfo(p.dtype).tiny:
                raise ValueError("Zero query-field energy; spectral objective is undefined")
            probabilities = energies/total
            count = max(1, int(n*self.noise_fraction))
            parts["query_energy_separation"] = probabilities[-count:].sum()
            parts["energy_concentration"] = -(probabilities*probabilities.clamp_min(torch.finfo(p.dtype).tiny).log()).sum()/math.log(n)
        if self.name == "smtl":
            total = parts["triplet"]
            for key, weight in (("query_energy_separation", self.alpha), ("energy_concentration", self.beta), ("distance_alignment", self.gamma)):
                if weight > 0:
                    total = total + weight*parts[key]
        elif self.name.endswith("triplet"):
            total = parts["triplet"]
        else:
            total = parts[self.name]
        return LossResult(total, parts)


def make_loss(name, geometry, **kwargs):
    return GeometryLoss(name, geometry, **kwargs)
