# Contributing

Install from the repository with `python -m pip install -e ".[viz]"`.
Run `python -m unittest discover -s tests -v` before proposing a change.

New geometries should provide explicit intrinsic/ambient dimensions, a
parametrization, its Jacobian, serialization and supported distance operations.
Add membership, derivative, tangent and boundary/seam tests. Unsupported
operations must raise rather than silently substitute another metric.

Keep demos deterministic and identify synthetic data. Do not commit private
embeddings, model caches, API keys or client notebooks. New evaluation protocols
must state their split and supervision rules. Include meaningful mathematical
or integration tests, rather than tests that only duplicate implementation steps.
