# Manifold Studio

**Compose geometries. Inspect embeddings. Keep the full representation.**

An initial Python research package by Vinit K. Chavan. Create Cartesian products
such as `Sphere() * Torus() * Plane()`, map your vectors with a fitted transform,
inspect 3D product/factor views and tangent directions, and export numerical data.

The repository includes a local FastAPI web playground. The core package adds
optional PyTorch projection-head training and selectable research losses to the
NumPy geometry and interactive plotting foundation. The package has not been
published on PyPI.

## Run the web playground

```bash
python -m pip install -e ".[viz,train]"
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Compose up to three geometries, load vectors or
use the demo, choose mapping/training, inspect 3D surfaces and tangent arrows,
view loss curves, and download your embeddings. Sentence input additionally
requires `.[text]` and downloads MiniLM on first use.

Source folders: [FastAPI backend](backend/README.md) and [frontend UI](frontend/README.md).
This is a local workbench; it is not deployed or configured for public multi-user hosting.

## Choose a loss and train

Seven selectable objectives include the original squared Euclidean triplet,
geodesic/ambient triplets, distance alignment, spectral query-energy separation,
energy concentration, and an experimental repaired SMTL combination.

```bash
python -m pip install -e ".[train,viz]"
python examples/train_projection.py --loss smtl --geometry 'sphere*torus'
```

```python
from manifold_studio import Sphere, Torus
from manifold_studio.losses import available_losses, make_loss
objective = make_loss("smtl", Sphere() * Torus(), alpha=1, beta=0.5, gamma=0.3)
print(available_losses())
```

See [loss choices, formulas, training, and historical audit](docs/LOSSES.md) and
[the training notebook](notebooks/02_selectable_losses.ipynb). Spectral components
use fixed detached graph supervision; these are experimental adaptations, not
reproductions of old benchmark results. Train only on training data and compare
on a held-out set before claiming an improvement.

## Install from this repository

Python 3.10 or newer:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[viz]"
```

The core requires NumPy. Plotly is optional. For the optional sentence encoder:

```bash
python -m pip install -e ".[viz,text]"
```

Do not run `pip install manifold-studio` expecting this code: it has not been
published to a package index. Install from the extracted repository directory.

## Compose two or three geometries

```python
import numpy as np
from manifold_studio import Sphere, Circle, Torus, Plane, Mobius, ManifoldProjector

geometry = Sphere() * Torus() * Plane()
# Also: Sphere() * Circle(), Torus() * Mobius(), Sphere() * Sphere()

X = np.load("your_embeddings.npy", allow_pickle=False)  # items × features
mapper = ManifoldProjector(geometry)
result = mapper.fit_transform(X)

print(geometry.intrinsic_dim)  # 6
print(result.coordinates.shape)  # (number of items, 9)
print(result.display().shape)    # (number of items, 3)

fig = result.plot_3d(target_index=0)
fig.show()
fig.write_html("explorer.html", include_plotlyjs=True)  # self-contained/offline

result.save("embeddings.npz")
result.export_csv("coordinates.csv")
mapper.save("projector.npz")
```

This is a **Cartesian product**, not the vector cross product. Products preserve
factor coordinates. They do not magically fit all information into three
dimensions. The interactive dropdown switches between the product's 3D PCA
display and separate factor surfaces. Hover identifies points, drag rotates,
scroll zooms, and buttons toggle labels. Choose an arrow target in Python with
`target_index`; changing targets by clicking is planned for the application.

## Immediate demo, no model download

```bash
manifold-studio demo --geometry 'sphere*torus*plane' --out outputs/demo
```

Open `outputs/demo/explorer.html`. This uses **synthetic numerical points**, not
language embeddings, and reports neighborhood preservation rather than accuracy.
Outputs include full embeddings, a fitted projector, coordinates CSV, metrics,
and a standalone Plotly visualization.

Map your own matrix and optionally annotate rows:

```bash
manifold-studio map --input embeddings.npy --texts sentences.json \
  --geometry 'sphere*circle' --out outputs/my_run
```

`sentences.json` is a list with one text string per row. For products containing
Möbius, explicitly select ambient distance:

```bash
manifold-studio demo --geometry 'torus*mobius' --metric ambient
```

Use `--no-html` without the Plotly dependency. Module invocation also works:
`python -m manifold_studio.cli demo --no-html`.

## Encode sentences

```python
from manifold_studio import encode_sentences, Sphere, Circle, ManifoldProjector

texts = [
    "I deposited money at the bank.",
    "We sat beside the river bank.",
    "The bridge is safe.",
    "The bridge is not safe.",
    "Maya believes the bridge is not safe.",
    "Maya does not believe the bridge is safe.",
    "The company announced higher earnings.",
    "The football team won the final.",
]
X = encode_sentences(texts)  # first use downloads the model
mapper = ManifoldProjector(Sphere() * Circle())
result = mapper.fit_transform(X, texts=texts)
result.plot_3d(show_labels=True, target_index=2).show()
```

These are sentence representations. Contextual word/span extraction and its
tokenizer alignment are planned; this release does not assign sentence vectors
to individual words. Pass `revision=` to `encode_sentences` to pin model weights
and record that revision in your experiment notes. The optional adapter was not
end-to-end model-tested in the initial build environment; core tests need no LLM.

## Geometry conventions

| Geometry | Intrinsic dimension | Ambient coordinates | Intrinsic distance |
|---|---:|---:|---|
| Plane | 2 | 2 | Euclidean |
| Circle | 1 | 2 | Shortest circular arc |
| Sphere | 2 | 3 | Great-circle distance |
| Cylinder | 2 | 3 | Circle arc × linear height |
| Torus | 2 | 4 | Flat Riemannian product of two circles |
| Möbius strip | 2 | 3 | Not implemented; ambient chord distance is explicit |
| Product | Sum | Sum | Square root of summed squared factor distances |

`Torus()` is mathematically `S¹ × S¹` in R4. Its donut rendering in R3 is
**non-isometric**: visual lengths on the donut are not the flat-torus metric.
The Möbius parametrization respects `(u + 2π, v) ~ (u, -v)`. No globally
consistent normal field or intrinsic Möbius shortest-path solver is claimed.

```python
geometry = Sphere(radius=2) * Circle(radius=0.5)
# Radius changes the corresponding factor's metric scale.
```

## Tangent vectors

```python
directions = result.toward(0)
# Tangent-projected ambient displacements, one at each base point.
# They are not generic geodesic/log-map directions.

random_vectors = np.random.default_rng(0).normal(size=result.shape)
tangents = result.tangent_project(random_vectors)
```

The sphere also exposes a true `log_map` for paired angular coordinates; exactly
antipodal points are rejected because the shortest direction is ambiguous.
General parallel transport, geodesic animation, and learned semantic edit
operators are future work. Plot arrows are scaled and limited to 150 nonzero
directions for readability. Full-product arrows are images under the PCA display
map, not tangent vectors to a fictitious 3D product surface.

## Fit once, transform consistently

The mapper fits PCA on all input columns, standardizes retained PC scores, and
allocates consecutive latent coordinates to the geometry factors. Spherical
latitude and Möbius width use smooth bounded maps. This is an explicit descriptive
mapping—not a discovery that language naturally lies on the selected surface,
and not end-to-end semantic training.

Fit on a reference or training collection. New inputs use the saved transform:

```python
from manifold_studio import ManifoldProjector, Embedding

mapper = ManifoldProjector.load("projector.npz")
new_result = mapper.transform(new_embeddings, texts=new_sentences)
old_result = Embedding.load("embeddings.npz")
```

Mapping an input alone and in a batch gives the same coordinates. The training
display PCA also remains fixed, so changes between views are not caused by
refitting the display. Coordinate comparisons require the same fitted mapper.
Degenerate/low-rank collections are rejected instead of inventing dimensions.
For intrinsic dimension `d`, fitting requires at least `d+1` varied examples,
at least `d` input columns, and sufficient rank.

## Evaluate rather than assume

```python
from manifold_studio import neighbor_preservation, evaluate_knn

print(neighbor_preservation(X, result, k=5, metric="intrinsic"))

report = evaluate_knn(
    X_train, y_train, X_test, y_test,
    geometry=Sphere() * Circle(), k=3,
)
print(report)
```

The labeled probe fits its mapper only on training rows. It compares full source
Euclidean kNN, dimension-matched standardized PCA kNN, and manifold kNN, reporting
accuracy and macro-F1. Caller-provided train/test groups must be independent;
upstream embedding leakage cannot be detected by this package. Do not repeatedly
choose geometry on the test set. This initial probe is not a multi-seed benchmark.

Unlabeled data gets neighborhood overlap, not an invented accuracy score. Dense
pairwise diagnostics are capped at 5,000 rows; holdout evaluation is capped at
25 million pairs. Sparse/ANN execution is on the roadmap.

## Export format

`embeddings.npz` stores source vectors, full manifold coordinates, intrinsic
parameters, row-aligned text, the fixed display PCA, and versioned geometry
metadata. `projector.npz` stores the fitted input transform for future queries.
Both use numeric/string arrays and are loaded with `allow_pickle=False`.
`coordinates.csv` contains ambient coordinates, not just the visible three axes.

The bundle includes the source matrix for inspection and is **not a compressed
memory codec**. File sizes are not token savings. Downstream systems must use the
same transform and an appropriate distance; Euclidean ANN on exported coordinates
does not automatically reproduce intrinsic manifold rankings.

## Repository layout

| Path | Responsibility |
|---|---|
| `src/manifold_studio/geometry.py` | Geometries, products, distances, Jacobians, tangents |
| `src/manifold_studio/embedding.py` | Fitted transforms and portable bundles |
| `src/manifold_studio/visualization.py` | Interactive full-product/factor views |
| `src/manifold_studio/evaluation.py` | Neighborhood and held-out classification probes |
| `src/manifold_studio/encoders.py` | Optional sentence-transformer adapter |
| `src/manifold_studio/cli.py` | Numerical demo and matrix-import command |
| `examples/` | Runnable numerical and text examples |
| `notebooks/` | Package-first walkthrough |
| `tests/` | Mathematical invariants and integration tests |
| `docs/` | Research provenance and development roadmap |

## Test

```bash
python -m unittest discover -s tests -v
```

Tests cover product dimensions and metrics, finite-difference Jacobians, sphere
tangents including poles, Möbius seam transitions, projector batch invariance,
export/reload/replay, validation, evaluation, and optional Plotly serialization.

## Repository and installation

The project repository is [vinitchavan/manifold-studio](https://github.com/vinitchavan/manifold-studio).
The repository is public. The package has not been published to PyPI; install
from a source checkout:

```bash
git clone https://github.com/vinitchavan/manifold-studio.git
cd manifold-studio
python -m pip install -e '.[viz]'
```

## Research roots and next steps

This package develops the product direction motivated by Vinit's
[manifold embedding research](https://github.com/vinitchavan/manifold-embedding-nlp)
and [Spectral Field Memory](https://github.com/vinitchavan/spectral-field-memory).
It uses a new small numerical implementation; old notebook performance numbers
are not claimed as results of this package. See `docs/PROVENANCE.md`.

Next: a playground UI, contextual token inspection, trained heads, validated
intrinsic mesh paths, spectral-memory diagnostics, and experimental transport
operators. The first release does not include these future components.
