"""Manifold Studio: compose geometry, inspect vectors, preserve provenance."""
from .geometry import Plane, Circle, Sphere, Cylinder, Torus, Mobius, Product
from .embedding import ManifoldProjector, Embedding
from .evaluation import neighbor_preservation, evaluate_knn
from .encoders import encode_sentences

__version__ = "0.1.0"
__all__ = ["Plane", "Circle", "Sphere", "Cylinder", "Torus", "Mobius", "Product",
           "ManifoldProjector", "Embedding", "neighbor_preservation", "evaluate_knn", "encode_sentences"]
