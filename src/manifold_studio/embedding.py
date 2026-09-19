"""Fitted maps and portable, non-pickle embedding bundles."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from .geometry import Geometry, Product, geometry_from_spec, matrix


def _pca(x, dim):
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x-mean, full_matrices=False)
    components = np.zeros((dim, x.shape[1]))
    n = min(dim, len(vt))
    components[:n] = vt[:n]
    # Fix arbitrary eigenvector signs for more consistent replays.
    for row in components[:n]:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return mean, components, s


def _strings(values, n):
    if values is None:
        return np.asarray([str(i) for i in range(n)])
    a = np.asarray(values, dtype=str)
    if a.ndim != 1 or len(a) != n:
        raise ValueError("texts must have one entry per embedding row")
    return a


def _write_npz(path, **arrays):
    # An explicit file handle preserves the exact caller-specified path.
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


@dataclass
class Embedding:
    geometry: Geometry
    coordinates: np.ndarray
    parameters: np.ndarray
    source: np.ndarray
    texts: np.ndarray
    display_mean: np.ndarray
    display_components: np.ndarray

    def __post_init__(self):
        self.coordinates = matrix(self.coordinates, self.geometry.ambient_dim, "coordinates")
        self.parameters = matrix(self.parameters, self.geometry.intrinsic_dim, "parameters")
        self.source = matrix(self.source, name="source")
        n = len(self.coordinates)
        if len(self.parameters) != n or len(self.source) != n:
            raise ValueError("Bundle row counts do not match")
        self.texts = _strings(self.texts, n)
        self.display_mean = np.asarray(self.display_mean, dtype=float)
        self.display_components = matrix(self.display_components, self.geometry.ambient_dim)
        if self.display_mean.shape != (self.geometry.ambient_dim,) or not np.isfinite(self.display_mean).all() or self.display_components.shape[0] != 3:
            raise ValueError("Invalid display transform")
        if not np.allclose(self.coordinates, self.geometry.from_parameters(self.parameters)):
            raise ValueError("Coordinates do not match the stored geometry and parameters")

    @property
    def shape(self):
        return self.coordinates.shape

    def distances(self, metric="intrinsic"):
        if len(self.coordinates) > 5000:
            raise ValueError("Dense distances limited to 5,000 rows; sample the dataset first")
        return self.geometry.distance(self.parameters, metric=metric)

    def tangent_project(self, vectors):
        return self.geometry.tangent_project(self.parameters, vectors)

    def toward(self, target_index):
        """Tangent-projected ambient displacement; NOT a generic log map."""
        if not 0 <= target_index < len(self.coordinates):
            raise IndexError("target_index is out of range")
        return self.tangent_project(self.coordinates[target_index] - self.coordinates)

    def display(self):
        """3D PCA display using the training/reference display basis."""
        return (self.coordinates-self.display_mean) @ self.display_components.T

    def plot_3d(self, **kwargs):
        from .visualization import plot_3d
        return plot_3d(self, **kwargs)

    def save(self, path):
        metadata = {"format": "manifold-studio-embedding", "version": 1,
                    "geometry": self.geometry.spec(), "mapping": "fitted PCA + parametrization",
                    "display": "reference-fitted PCA; distances are distorted"}
        _write_npz(path, coordinates=self.coordinates, parameters=self.parameters,
                   source=self.source, texts=self.texts, display_mean=self.display_mean,
                   display_components=self.display_components, metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as b:
            m = json.loads(str(b["metadata"]))
            if m.get("format") != "manifold-studio-embedding" or m.get("version") != 1:
                raise ValueError("Unsupported embedding bundle")
            return cls(geometry_from_spec(m["geometry"]), *(b[k].copy() for k in
                       ["coordinates", "parameters", "source", "texts", "display_mean", "display_components"]))

    def export_csv(self, path):
        """Export full ambient coordinates, not just the 3D display."""
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "text"] + [f"coordinate_{i}" for i in range(self.shape[1])])
            for i, (text, row) in enumerate(zip(self.texts, self.coordinates)):
                writer.writerow([i, text, *row])


class ManifoldProjector:
    """Fit once on reference/train vectors; transform queries with the same map.

    A descriptive geometry mapping, not semantic training or topology discovery.
    Uses all input columns in PCA and allocates consecutive PCs to factors.
    """
    def __init__(self, geometry):
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be a Geometry object")
        self.geometry = geometry

    def fit(self, x):
        x = matrix(x)
        dim = self.geometry.intrinsic_dim
        if min(len(x)-1, x.shape[1]) < dim:
            raise ValueError(f"This geometry needs at least {dim+1} reference rows and {dim} input dimensions")
        mean, components, s = _pca(x, dim)
        tolerance = max(x.shape)*np.finfo(float).eps*s[0] if s[0] else 1e-12
        if np.sum(s > tolerance) < dim:
            raise ValueError("Reference data rank is too small for this geometry; use fewer factors or more varied data")
        self.mean_, self.components_ = mean, components
        self.scale_ = s[:dim]/np.sqrt(len(x)-1)
        self.n_features_in_ = x.shape[1]
        self.training_rows_ = len(x)
        self.explained_variance_ratio_ = float(np.sum(s[:dim]**2)/np.sum(s**2))
        parameters = self.geometry.parameters_from_latent(self.latent(x))
        z = self.geometry.from_parameters(parameters)
        self.display_mean_, self.display_components_, _ = _pca(z, 3)
        return self

    def latent(self, x):
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit on a reference/training collection first")
        x = matrix(x, self.n_features_in_)
        return ((x-self.mean_) @ self.components_.T)/self.scale_

    def transform(self, x, texts=None):
        x = matrix(x)
        p = self.geometry.parameters_from_latent(self.latent(x))
        return Embedding(self.geometry, self.geometry.from_parameters(p), p, x.copy(),
                         _strings(texts, len(x)), self.display_mean_.copy(), self.display_components_.copy())

    def fit_transform(self, x, texts=None):
        return self.fit(x).transform(x, texts=texts)

    def save(self, path):
        if not hasattr(self, "components_"):
            raise RuntimeError("Cannot save an unfitted projector")
        metadata = {"format": "manifold-studio-projector", "version": 1,
                    "geometry": self.geometry.spec(), "training_rows": self.training_rows_,
                    "explained_variance_ratio": self.explained_variance_ratio_}
        _write_npz(path, mean=self.mean_, components=self.components_, scale=self.scale_,
                   display_mean=self.display_mean_, display_components=self.display_components_,
                   metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as b:
            m = json.loads(str(b["metadata"]))
            if m.get("format") != "manifold-studio-projector" or m.get("version") != 1:
                raise ValueError("Unsupported projector bundle")
            obj = cls(geometry_from_spec(m["geometry"]))
            for name in ["mean", "components", "scale", "display_mean", "display_components"]:
                setattr(obj, name+"_", np.asarray(b[name], dtype=float).copy())
        obj.n_features_in_ = len(obj.mean_)
        d = obj.geometry.intrinsic_dim
        a = obj.geometry.ambient_dim
        expected = [(obj.mean_, (obj.n_features_in_,)), (obj.components_, (d, obj.n_features_in_)),
                    (obj.scale_, (d,)), (obj.display_mean_, (a,)), (obj.display_components_, (3, a))]
        if any(v.shape != sh or not np.isfinite(v).all() for v, sh in expected) or np.any(obj.scale_ <= 0):
            raise ValueError("Invalid fitted projector state")
        obj.training_rows_ = int(m["training_rows"])
        obj.explained_variance_ratio_ = float(m["explained_variance_ratio"])
        return obj
