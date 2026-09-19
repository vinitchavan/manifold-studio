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
