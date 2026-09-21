# Contributing to Manifold Studio

Contributions from researchers, students, Python developers and frontend engineers
are welcome. Documentation fixes and negative experimental results matter as much
as new features. Read the [project motivation and theory](README.md) and
[roadmap](docs/ROADMAP.md) before proposing a substantial extension.

## A practical workflow

1. Check [existing issues](https://github.com/vinitchavan/manifold-studio/issues).
   Open an issue for a substantial method or API change so its scope and assumptions
   can be discussed. Small fixes can go directly to a pull request.
2. Fork the repository and create a focused branch from `main`.
3. Install the extras needed for your change, implement it, and update examples
   or documentation when behavior changes.
4. Run the relevant tests and explain the result in your pull request. State
   anything you could not verify instead of implying it passed.
5. Describe the problem, your changes, source attribution and material limitations.
   Keep unrelated changes in separate pull requests.

## Development setup

From the repository root, in a Python 3.10+ virtual environment:

```bash
python -m pip install -e ".[viz,train]"
python -m pip install -r backend/requirements.txt
python -m unittest discover -s tests -v
python -m unittest discover -s backend/tests -v
node --check frontend/app.js
```

For NumPy-only changes, `python -m pip install -e .` is sufficient; optional tests
may skip when their dependencies are absent. A change to training or plotting
must be checked with its corresponding extra installed. Node is only needed for
the JavaScript syntax check, not to serve the frontend.

To inspect UI changes, run `python -m uvicorn backend.main:app --port 8000` and
open http://127.0.0.1:8000. Check loading, success and error states, a narrow
viewport, keyboard controls and a representative plot/export workflow.

## Mathematical and research contributions

- **Geometry:** specify intrinsic/ambient dimensions, parametrization, Jacobian,
  supported metrics and serialization. Test membership, derivatives, tangents,
  chart singularities and boundary/seam behavior. Unsupported operations should
  raise an explicit error rather than substitute a different metric.
- **Loss functions:** provide the formula, inputs, supervision, units and weight
  conventions. Show finite/useful gradients and independent numerical checks
  where appropriate. Explain graph detachment, approximation or non-differentiable
  operations. Extend the loss documentation and selection interfaces together.
- **Experiments:** identify dataset/model versions, seeds, split construction,
  preprocessing and triplet/pseudo-label mining. Fit only on training data; choose
  hyperparameters on validation data. Compare against matched baselines and
  report uncertainty, failure cases and resource requirements where relevant.
- **Claims:** distinguish adaptation, implementation and demonstrated research
  novelty. Lower loss, improved silhouette or an attractive plot alone does not
  establish semantic improvement. Do not reuse historical scores as new results.

See [loss provenance](docs/LOSSES.md), [validation](docs/VALIDATION.md) and
[research sources](docs/PROVENANCE.md). These documents are part of the scientific
record of the package and should stay aligned with code changes.

## Useful places to start

- Clarify an example, formula or error message.
- Improve frontend accessibility and mobile plot controls.
- Add a small independent geometry/gradient regression test.
- Add reproducible held-out evaluation with clearly documented baselines.
- Investigate a roadmap topic and contribute a minimal, measurable experiment.

## Data, attribution and collaboration

Keep demos deterministic and identify synthetic data. Do not commit private
embeddings, credentials, model caches or client notebooks. Only contribute code
and data you are entitled to share; cite adapted methods and respect upstream
licenses. Package contributions are intended for the repository's MIT license;
external datasets and model weights retain their own terms.

Be constructive when reviewing research and code. Explain disagreements with
formulas, evidence or a reproducible example, and give credit to contributors and
original sources. There is no requirement to claim a positive result to contribute.
