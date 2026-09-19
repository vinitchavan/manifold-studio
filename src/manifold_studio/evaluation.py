"""Small-dataset diagnostics. Accuracy requires labels and an explicit split."""
import numpy as np
from .geometry import matrix
from .embedding import ManifoldProjector


def _euclidean(x, y):
    return np.linalg.norm(x[:, None, :]-y[None, :, :], axis=-1)


def neighbor_preservation(source, embedding, k=5, metric="intrinsic"):
    source = matrix(source)
    n = len(source)
    if n != len(embedding.coordinates) or not 1 <= k < n:
        raise ValueError("Need matching rows and 1 <= k < number of rows")
    if n > 5000:
        raise ValueError("Dense diagnostics limited to 5,000 rows")
    a = _euclidean(source, source)
    b = embedding.distances(metric=metric)
    np.fill_diagonal(a, np.inf)
    np.fill_diagonal(b, np.inf)
    ia, ib = np.argsort(a, axis=1, kind="stable")[:, :k], np.argsort(b, axis=1, kind="stable")[:, :k]
    overlap = [len(set(x) & set(y))/k for x, y in zip(ia, ib)]
    return {"neighbor_overlap": float(np.mean(overlap)), "k": k, "n": n,
            "source_metric": "euclidean", "target_metric": metric,
            "interpretation": "Neighborhood preservation, not semantic accuracy"}


def _scores(distances, train_y, test_y, k):
    classes = np.unique(np.concatenate([train_y, test_y]))
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
    predicted = []
    for row in nearest:
        labels, counts = np.unique(train_y[row], return_counts=True)
        predicted.append(labels[np.argmax(counts)])
    predicted = np.asarray(predicted)
    f1 = []
    for c in classes:
        tp = np.sum((predicted == c) & (test_y == c))
        fp = np.sum((predicted == c) & (test_y != c))
        fn = np.sum((predicted != c) & (test_y == c))
        f1.append(2*tp/max(2*tp+fp+fn, 1))
    return {"accuracy": float(np.mean(predicted == test_y)), "macro_f1": float(np.mean(f1))}


def evaluate_knn(train_x, train_y, test_x, test_y, geometry, k=3, metric="intrinsic"):
    """Fit only on train_x; compare source, matched PCA, and manifold kNN.

    Caller owns split provenance, duplicate grouping and upstream leakage.
    This is one holdout probe, not cross-validation or geometry selection.
    """
    train_x, test_x = matrix(train_x), matrix(test_x)
    train_y, test_y = np.asarray(train_y, dtype=str), np.asarray(test_y, dtype=str)
    if train_y.shape != (len(train_x),) or test_y.shape != (len(test_x),):
        raise ValueError("One label is required per row")
    if not 1 <= k <= len(train_x):
        raise ValueError("k must be between 1 and the training size")
    if len(train_x)*len(test_x) > 25_000_000:
        raise ValueError("Dense evaluation is limited to 25 million pairs")
    model = ManifoldProjector(geometry).fit(train_x)
    train, test = model.transform(train_x), model.transform(test_x)
    d = geometry.distance(test.parameters, train.parameters, metric=metric)
    return {"source_euclidean": _scores(_euclidean(test_x, train_x), train_y, test_y, k),
            "pca_standardized_euclidean": _scores(_euclidean(model.latent(test_x), model.latent(train_x)), train_y, test_y, k),
            "manifold": _scores(d, train_y, test_y, k),
            "metric": metric, "k": k, "train_rows": len(train_x), "test_rows": len(test_x),
            "intrinsic_dim": geometry.intrinsic_dim, "ambient_dim": geometry.ambient_dim}
