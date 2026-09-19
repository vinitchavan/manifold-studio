# Initial validation — 19 September 2026

Environment: Python 3.12; NumPy runtime; Plotly 6.9.0 for optional charts.

## Passed

- 22 unittest cases, including analytic versus finite-difference Jacobians,
  tangent orthogonality/idempotence, Möbius seam and tangent-coordinate
  transitions, product dimensions, explicit product metrics and sphere log maps.
- Fitted-map consistency for an input encoded alone versus in a batch.
- Full embedding and fitted-projector export/load, plus future-query replay.
- Invalid input, rank deficiency, mismatched labels/features and tampered
  coordinate validation.
- Train-only fitting in the held-out kNN probe path, with baseline output.
- Plotly figure serialization and product/factor dropdown trace configuration.
- Wheel build and installation into a separate target directory.
- Installed-package CLI smoke run from outside the source checkout for
  `Torus() * Mobius()` using explicit ambient distance.
- `Sphere() * Torus() * Plane()` CLI demo, with data/transform/CSV/metric exports
  and self-contained HTML.

## Not verified

- End-to-end sentence-transformer download/encoding: the optional neural-model
  dependency and weights were not installed in this build environment.
- Browser-rendered visual inspection and browser click automation: Playwright
  was available, but its Chromium binary was absent and the attempted download
  timed out. HTML generation passed; a visual rendering check remains necessary.
- The proposed GitHub Actions Python 3.10/3.12 matrix has not run on GitHub.
- Cross-platform installation, large-data scalability and semantic benchmark
  superiority. No such claims are made for this release.

The demo uses synthetic numerical points. Its neighborhood score is a measured
property of that example, not evidence of improved language understanding.

## v0.2 loss integration validation — 19 September 2026

Python 3.12, PyTorch 2.14.0+cpu, NumPy 2.3.5, Plotly 6.9.0.

- All **32 tests passed** with both train and visualization extras installed.
- Torch/NumPy coordinate and metric parity, periodic seams and Möbius metric rejection.
- Original squared-triplet arithmetic and inactive-triplet zero gradients.
- Finite nonzero gradients for all seven selectable losses.
- Double-precision finite-difference autograd check for composite SMTL, including query gradients.
- Basis sign invariance; no gradients through source teacher/basis; bounded component values.
- Finite gradients at duplicate points, invalid supervision and zero-weight behavior.
- Optimization smoke test, non-pickle model restoration, embedding mapping metadata,
  and batch-independent held-out inference after restoration.
- The new tutorial's code cells executed, exported files and generated its Plotly HTML.
- An isolated editable install with optional visualization succeeded.

Example results (synthetic only, not paper benchmarks):

- `--loss smtl --geometry 'sphere*torus' --steps 20 --seed 7`:
  objective **0.188665 -> 0.058781**.
- Notebook's independent synthetic setup (48 train / 15 test, seed 7): source
  1-NN accuracy **1.000**, learned manifold 1-NN **0.867**, while the logged
  objective falls **0.379483 -> 0.025845**. The last logged value is before the
  final update. This is a concrete example of why a lower SMTL objective does
  not establish better semantic embeddings or even better task accuracy.

No real NLP benchmark, CUDA run, browser visual inspection, or superiority claim
was added. GitHub CI status should be checked separately from these local results.
