"""Generate a reproducible numerical demo or map an existing NPY matrix."""
import argparse
import json
from pathlib import Path
import numpy as np

from . import Sphere, Circle, Torus, Plane, Cylinder, Mobius, ManifoldProjector, neighbor_preservation


def parse_geometry(value):
    constructors = {"sphere": Sphere, "circle": Circle, "torus": Torus, "plane": Plane,
                    "cylinder": Cylinder, "mobius": Mobius}
    parts = value.lower().replace("×", "*").split("*")
    try:
        factors = [constructors[p.strip()]() for p in parts]
    except KeyError as e:
        raise argparse.ArgumentTypeError("Use sphere, circle, torus, plane, cylinder or mobius joined by *") from e
    geometry = factors[0]
    for f in factors[1:]:
        geometry = geometry*f
    return geometry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "map"])
    parser.add_argument("--input", help="Numeric .npy matrix, rows are items")
    parser.add_argument("--texts", help="JSON list with one string per input row")
    parser.add_argument("--geometry", type=parse_geometry, default=Sphere()*Torus()*Plane())
    parser.add_argument("--out", default="outputs/demo")
    parser.add_argument("--metric", choices=["intrinsic", "ambient"], default="intrinsic")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "demo":
        rng = np.random.default_rng(42)
        labels = np.repeat(np.arange(3), 40)
        x = rng.normal(size=(120, 24)) + rng.normal(size=(3, 24))[labels]*1.5
        texts = [f"Synthetic group {c}, point {i}" for i, c in enumerate(labels)]
    else:
        if not args.input:
            parser.error("map requires --input")
        x = np.load(args.input, allow_pickle=False)
        labels = None
        texts = json.loads(Path(args.texts).read_text(encoding="utf-8")) if args.texts else None
    if args.metric == "intrinsic" and not args.geometry.supports_intrinsic:
        parser.error("This geometry contains Möbius; use --metric ambient explicitly")
    if not args.no_html:
        try:
            import plotly  # noqa: F401
        except ImportError:
            parser.error('Install ".[viz]" or pass --no-html')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mapper = ManifoldProjector(args.geometry)
    result = mapper.fit_transform(x, texts=texts)
    mapper.save(out/"projector.npz")
    result.save(out/"embeddings.npz")
    result.export_csv(out/"coordinates.csv")
    report = neighbor_preservation(x, result, k=min(5, len(x)-1), metric=args.metric)
    report.update({"input_kind": "synthetic numerical demo" if args.command == "demo" else "user-supplied embeddings",
                   "geometry": args.geometry.spec(), "source_shape": list(x.shape),
                   "coordinate_shape": list(result.shape), "pca_variance_retained": mapper.explained_variance_ratio_,
                   "embedding_bundle_bytes": (out/"embeddings.npz").stat().st_size,
                   "projector_bytes": (out/"projector.npz").stat().st_size})
    (out/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_html:
        result.plot_3d(labels=labels, target_index=0).write_html(out/"explorer.html", include_plotlyjs=True)
    print(json.dumps({"output": str(out.resolve()), "shape": list(result.shape), "metrics": report}, indent=2))


if __name__ == "__main__":
    main()
