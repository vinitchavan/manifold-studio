"""Optional differentiable counterparts of the NumPy geometry parametrizations."""
import math
try:
    import torch
except ImportError as exc:
    raise ImportError('Install manifold-studio with the [train] extra for PyTorch support') from exc
from .geometry import Plane, Circle, Sphere, Cylinder, Torus, Mobius, Product


def tensor_matrix(x, width=None, name="input"):
    if not isinstance(x, torch.Tensor) or x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a float32 or float64 torch tensor")
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    if width is not None and x.shape[1] != width:
        raise ValueError(f"{name} needs {width} columns")
    return x


def parameters_from_latent(geometry, latent):
    p = tensor_matrix(latent, geometry.intrinsic_dim)
    if isinstance(geometry, Product):
        return torch.cat([parameters_from_latent(f, p[:, ps]) for f, ps, _ in geometry.blocks()], 1)
    if isinstance(geometry, Sphere):
        return torch.stack((p[:, 0], (math.pi/2-1e-7)*p[:, 1].tanh()), 1)
    if isinstance(geometry, Mobius):
        return torch.stack((p[:, 0], geometry.width*p[:, 1].tanh()), 1)
    return p


def coordinates(geometry, parameters):
    p = tensor_matrix(parameters, geometry.intrinsic_dim, "parameters")
    if isinstance(geometry, Product):
        return torch.cat([coordinates(f, p[:, ps]) for f, ps, _ in geometry.blocks()], 1)
    if isinstance(geometry, Plane):
        return p
    u = p[:, 0]
    if isinstance(geometry, Circle):
        return geometry.radius*torch.stack((u.cos(), u.sin()), 1)
    v = p[:, 1]
    if isinstance(geometry, Sphere):
        return geometry.radius*torch.stack((v.cos()*u.cos(), v.cos()*u.sin(), v.sin()), 1)
    if isinstance(geometry, Cylinder):
        return torch.stack((geometry.radius*u.cos(), geometry.radius*u.sin(), v), 1)
    if isinstance(geometry, Torus):
        return torch.stack((geometry.radius1*u.cos(), geometry.radius1*u.sin(),
                            geometry.radius2*v.cos(), geometry.radius2*v.sin()), 1)
    if isinstance(geometry, Mobius):
        if (v.abs() > geometry.width + 1e-7).any():
            raise ValueError("Möbius width coordinate is outside the strip")
        r = geometry.radius + v*(u/2).cos()
        return torch.stack((r*u.cos(), r*u.sin(), v*(u/2).sin()), 1)
    raise TypeError("Unsupported geometry")


def _arc(a, b):
    return torch.remainder(a[:, None]-b[None, :]+math.pi, 2*math.pi)-math.pi


def distances(geometry, p, q=None, metric="intrinsic"):
    """N x M distances in parameter coordinates. Zero subgradient at coincidence.

    Intrinsic shortest-path distances are nonsmooth at cut loci (e.g. antipodes).
    Möbius requires explicit ambient mode, including inside a product.
    """
    tensor_matrix(p, geometry.intrinsic_dim, "parameters")
    q = p if q is None else tensor_matrix(q, geometry.intrinsic_dim, "query parameters")
    if q.dtype != p.dtype or q.device != p.device:
        raise ValueError("Point tensors must have the same dtype and device")
    if metric == "ambient":
        x, y = coordinates(geometry, p), coordinates(geometry, q)
        return torch.linalg.vector_norm(x[:, None]-y[None, :], dim=-1)
    if metric != "intrinsic":
        raise ValueError("metric must be intrinsic or ambient")
    if not geometry.supports_intrinsic:
        raise NotImplementedError("Möbius intrinsic distance is not implemented; select metric='ambient'")
    if isinstance(geometry, Product):
        return torch.linalg.vector_norm(torch.stack([
            distances(f, p[:, ps], q[:, ps]) for f, ps, _ in geometry.blocks()], -1), dim=-1)
    if isinstance(geometry, Plane):
        return torch.linalg.vector_norm(p[:, None]-q[None, :], dim=-1)
    if isinstance(geometry, Circle):
        return geometry.radius*_arc(p[:, 0], q[:, 0]).abs()
    if isinstance(geometry, Sphere):
        x, y = coordinates(geometry, p)/geometry.radius, coordinates(geometry, q)/geometry.radius
        cross = torch.linalg.cross(x[:, None], y[None, :], dim=-1)
        return geometry.radius*torch.atan2(torch.linalg.vector_norm(cross, dim=-1),
                                           (x @ y.T).clamp(-1, 1))
    du = geometry.radius1*_arc(p[:, 0], q[:, 0]) if isinstance(geometry, Torus) else geometry.radius*_arc(p[:, 0], q[:, 0])
    dv = geometry.radius2*_arc(p[:, 1], q[:, 1]) if isinstance(geometry, Torus) else p[:, 1, None]-q[None, :, 1]
    return torch.linalg.vector_norm(torch.stack((du, dv), -1), dim=-1)
