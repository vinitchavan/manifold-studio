# Package first, application second

## Implemented in 0.1

- Plane, circle, sphere, cylinder, flat torus, Möbius and flattened products.
- Two or three factors via ordinary Python multiplication; more are supported
  when the reference matrix has sufficient independent dimensions.
- Fitted PCA-based parametrization with batch-invariant future transforms.
- Tangent projections and sphere log maps.
- Full-product 3D PCA plus factor surfaces, text hover, and direction cones.
- Portable embedding/projector bundles and CSV.
- Neighborhood preservation and explicit train/test kNN comparisons.
- Optional sentence encoding adapter and synthetic offline demo.

## Next package milestones

1. Contextual-token spans, encoder provenance and model caching.
2. Reusable higher-dimensional display slices and persistent factor weights.
3. Surface path solvers with convergence tests, then parallel transport.
4. Sparse/ANN neighbor computation for larger datasets.
5. Trained projection heads with clean split-before-fit supervision.
6. SFM plugin with actual byte accounting, paired retrieval metrics and cached graphs.
7. Experimental semantic edit transport and composition-held-out tests.

## Application milestone

Build a geometry playground over this package: drag factors into a product,
paste/upload inputs, select arrow targets with a click, compare metric panels,
adjust radii, and export the same portable bundles. The UI must call this core
rather than reimplement geometry in an incompatible code path.

## Research acceptance criteria

Geometric validity is not semantic usefulness. Claim improvements only after
matched-dimension, matched-supervision comparisons, independent splits, repeated
seeds and paired task evaluation. A geometric rendering alone does not establish
the topology of language.
