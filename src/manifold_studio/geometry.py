"""Explicit low-dimensional geometries and Cartesian products.

Torus is the flat product of two circles in R4, not the induced metric of
a donut in R3. Möbius intrinsic distance is deliberately unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def matrix(x, width=None, name="input"):
    raw = np.asarray(x)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    a = np.asarray(raw, dtype=float)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty, finite 2D array")
    if width is not None and a.shape[1] != width:
        raise ValueError(f"{name} needs {width} columns; got {a.shape[1]}")
    return a


def positive(v, name):
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _arc(a, b):
    return np.abs((a[:, None] - b[None, :] + np.pi) % (2*np.pi) - np.pi)


class Geometry:
    intrinsic_dim: int
    ambient_dim: int
    supports_intrinsic = True

    def __mul__(self, other):
        if not isinstance(other, Geometry):
            return NotImplemented
        left = self.factors if isinstance(self, Product) else (self,)
        right = other.factors if isinstance(other, Product) else (other,)
        return Product(*(left + right))

    def parameters_from_latent(self, latent):
        return matrix(latent, self.intrinsic_dim).copy()

    def from_parameters(self, parameters):
        raise NotImplementedError

    def jacobian(self, parameters):
        """N x ambient_dim x intrinsic_dim derivative of the parametrization."""
        raise NotImplementedError

    def tangent_project(self, parameters, vectors):
        p = matrix(parameters, self.intrinsic_dim, "parameters")
        v = matrix(vectors, self.ambient_dim, "vectors")
        if len(p) != len(v):
            raise ValueError("One vector is required per base point")
        j = self.jacobian(p)
        # Pseudoinverse also handles coordinate-chart rank deficiencies.
        return np.einsum("nai,nib,nb->na", j, np.linalg.pinv(j), v)

    def distance(self, p, q=None, metric="intrinsic"):
        p = matrix(p, self.intrinsic_dim, "parameters")
        q = p if q is None else matrix(q, self.intrinsic_dim, "parameters")
        if metric == "ambient":
            x, y = self.from_parameters(p), self.from_parameters(q)
            return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        if metric != "intrinsic":
            raise ValueError("metric must be 'intrinsic' or 'ambient'")
        if not self.supports_intrinsic:
            raise NotImplementedError("Möbius intrinsic distance is not implemented; select metric='ambient' explicitly")
        return self._intrinsic(p, q)

    def spec(self):
        raise NotImplementedError


@dataclass(frozen=True)
class Plane(Geometry):
    intrinsic_dim = 2
    ambient_dim = 2

    def from_parameters(self, parameters):
        return matrix(parameters, 2).copy()

    def jacobian(self, parameters):
        return np.tile(np.eye(2), (len(matrix(parameters, 2)), 1, 1))

    def _intrinsic(self, p, q):
        return np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)

    def spec(self):
        return {"type": "plane"}


@dataclass(frozen=True)
class Circle(Geometry):
    radius: float = 1.0
    intrinsic_dim = 1
    ambient_dim = 2

    def __post_init__(self):
        positive(self.radius, "radius")

    def from_parameters(self, parameters):
        u = matrix(parameters, 1)[:, 0]
        return self.radius * np.column_stack([np.cos(u), np.sin(u)])

    def jacobian(self, parameters):
        u = matrix(parameters, 1)[:, 0]
        return self.radius * np.stack([-np.sin(u), np.cos(u)], axis=1)[:, :, None]

    def _intrinsic(self, p, q):
        return self.radius * _arc(p[:, 0], q[:, 0])

    def spec(self):
        return {"type": "circle", "radius": self.radius}


@dataclass(frozen=True)
class Sphere(Geometry):
    radius: float = 1.0
    intrinsic_dim = 2
    ambient_dim = 3

    def __post_init__(self):
        positive(self.radius, "radius")

    def parameters_from_latent(self, latent):
        p = matrix(latent, 2).copy()
        p[:, 1] = (np.pi/2 - 1e-7) * np.tanh(p[:, 1])
        return p

    def from_parameters(self, parameters):
        u, v = matrix(parameters, 2).T
        return self.radius * np.column_stack([np.cos(v)*np.cos(u), np.cos(v)*np.sin(u), np.sin(v)])

    def jacobian(self, parameters):
        u, v = matrix(parameters, 2).T
        du = np.column_stack([-np.cos(v)*np.sin(u), np.cos(v)*np.cos(u), np.zeros_like(u)])
        dv = np.column_stack([-np.sin(v)*np.cos(u), -np.sin(v)*np.sin(u), np.cos(v)])
        return self.radius * np.stack([du, dv], axis=-1)

    def tangent_project(self, parameters, vectors):
        x = self.from_parameters(parameters) / self.radius
        v = matrix(vectors, 3)
        if len(x) != len(v):
            raise ValueError("One vector is required per base point")
        return v - np.sum(v*x, axis=1, keepdims=True)*x

    def _intrinsic(self, p, q):
        x, y = self.from_parameters(p)/self.radius, self.from_parameters(q)/self.radius
        cross = np.linalg.norm(np.cross(x[:, None, :], y[None, :, :]), axis=-1)
        return self.radius * np.arctan2(cross, np.clip(x@y.T, -1, 1))

    def log_map(self, p, q):
        p, q = matrix(p, 2), matrix(q, 2)
        if p.shape != q.shape:
            raise ValueError("log_map expects paired points")
        x, y = self.from_parameters(p)/self.radius, self.from_parameters(q)/self.radius
        dot = np.clip(np.sum(x*y, axis=1), -1, 1)
        v = y - dot[:, None]*x
        sine = np.linalg.norm(v, axis=1)
        if np.any((sine < 1e-10) & (dot < 0)):
            raise ValueError("Antipodal sphere points have no unique shortest-path direction")
        angle = np.arctan2(sine, dot)
        return self.radius*v*(angle/np.maximum(sine, 1e-15))[:, None]

    def spec(self):
        return {"type": "sphere", "radius": self.radius}


@dataclass(frozen=True)
class Cylinder(Geometry):
    radius: float = 1.0
    intrinsic_dim = 2
    ambient_dim = 3

    def __post_init__(self):
        positive(self.radius, "radius")

    def from_parameters(self, parameters):
        u, v = matrix(parameters, 2).T
        return np.column_stack([self.radius*np.cos(u), self.radius*np.sin(u), v])

    def jacobian(self, parameters):
        u = matrix(parameters, 2)[:, 0]
        j = np.zeros((len(u), 3, 2))
        j[:, 0, 0], j[:, 1, 0], j[:, 2, 1] = -self.radius*np.sin(u), self.radius*np.cos(u), 1
        return j

    def _intrinsic(self, p, q):
        return np.hypot(self.radius*_arc(p[:, 0], q[:, 0]), p[:, 1, None]-q[None, :, 1])

    def spec(self):
        return {"type": "cylinder", "radius": self.radius}


@dataclass(frozen=True)
class Torus(Geometry):
    """Flat Riemannian product S1(r1) x S1(r2), represented in R4."""
    radius1: float = 1.0
    radius2: float = 1.0
    intrinsic_dim = 2
    ambient_dim = 4

    def __post_init__(self):
        positive(self.radius1, "radius1")
        positive(self.radius2, "radius2")

    def from_parameters(self, parameters):
        p = matrix(parameters, 2)
        return np.column_stack([Circle(self.radius1).from_parameters(p[:, :1]), Circle(self.radius2).from_parameters(p[:, 1:])])

    def jacobian(self, parameters):
        p = matrix(parameters, 2)
        j = np.zeros((len(p), 4, 2))
        j[:, :2, :1] = Circle(self.radius1).jacobian(p[:, :1])
        j[:, 2:, 1:] = Circle(self.radius2).jacobian(p[:, 1:])
        return j

    def _intrinsic(self, p, q):
        return np.hypot(self.radius1*_arc(p[:, 0], q[:, 0]), self.radius2*_arc(p[:, 1], q[:, 1]))

    def spec(self):
        return {"type": "torus", "radius1": self.radius1, "radius2": self.radius2}


@dataclass(frozen=True)
class Mobius(Geometry):
    radius: float = 1.0
    width: float = 0.35
    intrinsic_dim = 2
    ambient_dim = 3
    supports_intrinsic = False

    def __post_init__(self):
        positive(self.radius, "radius")
        positive(self.width, "width")
        if self.width >= self.radius:
            raise ValueError("width must be smaller than radius")

    def parameters_from_latent(self, latent):
        p = matrix(latent, 2).copy()
        p[:, 1] = self.width*np.tanh(p[:, 1])
        # No independent angle wrapping: (u+2pi, v) is equivalent to (u,-v).
        return p

    def _parameters(self, parameters):
        p = matrix(parameters, 2)
        if np.any(np.abs(p[:, 1]) > self.width + 1e-12):
            raise ValueError("Möbius width coordinate is outside the strip")
        return p

    def from_parameters(self, parameters):
        u, v = self._parameters(parameters).T
        r = self.radius + v*np.cos(u/2)
        return np.column_stack([r*np.cos(u), r*np.sin(u), v*np.sin(u/2)])

    def jacobian(self, parameters):
        u, v = self._parameters(parameters).T
        r = self.radius + v*np.cos(u/2)
        dr = -0.5*v*np.sin(u/2)
        du = np.column_stack([dr*np.cos(u)-r*np.sin(u), dr*np.sin(u)+r*np.cos(u), 0.5*v*np.cos(u/2)])
        dv = np.column_stack([np.cos(u/2)*np.cos(u), np.cos(u/2)*np.sin(u), np.sin(u/2)])
        return np.stack([du, dv], axis=-1)

    def spec(self):
        return {"type": "mobius", "radius": self.radius, "width": self.width}


@dataclass(frozen=True, init=False)
class Product(Geometry):
    factors: tuple

    def __init__(self, *factors):
        flat = []
        for f in factors:
            if not isinstance(f, Geometry):
                raise TypeError("All factors must be Geometry objects")
            flat.extend(f.factors if isinstance(f, Product) else [f])
        if len(flat) < 2:
            raise ValueError("Product requires at least two factors")
        object.__setattr__(self, "factors", tuple(flat))

    @property
    def intrinsic_dim(self):
        return sum(f.intrinsic_dim for f in self.factors)

    @property
    def ambient_dim(self):
        return sum(f.ambient_dim for f in self.factors)

    @property
    def supports_intrinsic(self):
        return all(f.supports_intrinsic for f in self.factors)

    def blocks(self):
        a = p = 0
        for f in self.factors:
            yield f, slice(p, p+f.intrinsic_dim), slice(a, a+f.ambient_dim)
            p += f.intrinsic_dim
            a += f.ambient_dim

    def parameters_from_latent(self, latent):
        x = matrix(latent, self.intrinsic_dim)
        return np.column_stack([f.parameters_from_latent(x[:, ps]) for f, ps, _ in self.blocks()])

    def from_parameters(self, parameters):
        p = matrix(parameters, self.intrinsic_dim)
        return np.column_stack([f.from_parameters(p[:, ps]) for f, ps, _ in self.blocks()])

    def jacobian(self, parameters):
        p = matrix(parameters, self.intrinsic_dim)
        j = np.zeros((len(p), self.ambient_dim, self.intrinsic_dim))
        for f, ps, xs in self.blocks():
            j[:, xs, ps] = f.jacobian(p[:, ps])
        return j

    def tangent_project(self, parameters, vectors):
        p = matrix(parameters, self.intrinsic_dim)
        v = matrix(vectors, self.ambient_dim)
        if len(p) != len(v):
            raise ValueError("One vector is required per base point")
        return np.column_stack([f.tangent_project(p[:, ps], v[:, xs]) for f, ps, xs in self.blocks()])

    def _intrinsic(self, p, q):
        return np.sqrt(sum(f.distance(p[:, ps], q[:, ps])**2 for f, ps, _ in self.blocks()))

    def spec(self):
        return {"type": "product", "factors": [f.spec() for f in self.factors]}


def geometry_from_spec(spec):
    s = dict(spec)
    kind = s.pop("type")
    if kind == "product":
        return Product(*(geometry_from_spec(f) for f in s["factors"]))
    types = {"plane": Plane, "circle": Circle, "sphere": Sphere, "cylinder": Cylinder, "torus": Torus, "mobius": Mobius}
    if kind not in types:
        raise ValueError(f"Unsupported geometry: {kind}")
    return types[kind](**s)
