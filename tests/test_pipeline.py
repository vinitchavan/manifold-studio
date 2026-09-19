import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from manifold_studio import Sphere, Torus, Plane, Mobius, ManifoldProjector, Embedding, neighbor_preservation, evaluate_knn


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(4).normal(size=(60, 16))

    def test_frozen_single_query_transform(self):
        model = ManifoldProjector(Sphere()*Torus()*Plane()).fit(self.x)
        batch = model.transform(self.x[:4])
        for i in range(4):
            single = model.transform(self.x[i:i+1])
            np.testing.assert_allclose(single.coordinates, batch.coordinates[i:i+1], atol=1e-12)
            np.testing.assert_allclose(single.display(), batch.display()[i:i+1], atol=1e-12)
        self.assertFalse(np.allclose(batch.coordinates[0], batch.coordinates[1]))

    def test_uses_later_input_dimensions(self):
        x = self.x.copy(); x[:, :3] = 0
        out = ManifoldProjector(Sphere()*Torus()).fit_transform(x)
        self.assertGreater(np.std(out.coordinates), 0.1)

    def test_save_load_and_future_query(self):
        model = ManifoldProjector(Sphere()*Mobius()*Torus()).fit(self.x)
        result = model.transform(self.x, texts=[f"word {i}" for i in range(60)])
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/"mapper.npz", Path(d)/"embeddings.npz"
            model.save(a); result.save(b)
            restored, bundle = ManifoldProjector.load(a), Embedding.load(b)
            np.testing.assert_allclose(restored.transform(self.x[1:2]).coordinates, result.coordinates[1:2])
            np.testing.assert_allclose(bundle.parameters, result.parameters)
            np.testing.assert_equal(bundle.texts, result.texts)
            self.assertEqual(bundle.geometry.spec(), result.geometry.spec())
            with np.load(b, allow_pickle=False) as z:
                self.assertTrue(all(z[k].dtype.kind != "O" for k in z.files))

    def test_invalid_inputs(self):
        for x in [np.ones((10, 3)), [[1, float("nan")]], [["a", "b"]], [1, 2, 3], np.ones((3, 2), dtype=complex)]:
            with self.assertRaises(ValueError):
                ManifoldProjector(Sphere()).fit(x)
        with self.assertRaises(ValueError):
            ManifoldProjector(Sphere()*Torus()).fit(self.x[:3])
        with self.assertRaises(RuntimeError):
            ManifoldProjector(Sphere()).transform(self.x)

    def test_bad_labels_and_feature_dimensions(self):
        model = ManifoldProjector(Sphere()).fit(self.x)
        with self.assertRaises(ValueError): model.transform(self.x, texts=["one"])
        with self.assertRaises(ValueError): model.transform(np.ones((2, 7)))

    def test_metadata_and_shape(self):
        out = ManifoldProjector(Sphere()*Torus()*Plane()).fit_transform(self.x)
        self.assertEqual(out.shape, (60, 9))
        self.assertEqual(out.display().shape, (60, 3))
        self.assertEqual(out.parameters.shape, (60, 6))

    def test_neighbors_and_tangent_target(self):
        out = ManifoldProjector(Sphere()).fit_transform(self.x)
        report = neighbor_preservation(self.x, out)
        self.assertTrue(0 <= report["neighbor_overlap"] <= 1)
        np.testing.assert_allclose(out.toward(2)[2], 0, atol=1e-12)
        with self.assertRaises(ValueError): neighbor_preservation(self.x, out, k=60)

    def test_heldout_probe(self):
        train = self.x[:40].copy(); test = self.x[40:].copy()
        train_y = np.asarray(["a"]*20+["b"]*20)
        test_y = np.asarray(["a"]*10+["b"]*10)
        train[20:, 0] += 20; test[10:, 0] += 20
        report = evaluate_knn(train, train_y, test, test_y, Sphere(), k=3)
        self.assertEqual(report["train_rows"], 40)
        self.assertGreater(report["source_euclidean"]["accuracy"], 0.9)
        self.assertIn("pca_standardized_euclidean", report)

    def test_tampered_coordinates_rejected(self):
        out = ManifoldProjector(Sphere()).fit_transform(self.x)
        with self.assertRaises(ValueError):
            Embedding(out.geometry, out.coordinates*2, out.parameters, out.source, out.texts, out.display_mean, out.display_components)

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "optional Plotly dependency not installed")
    def test_viewer_serialization_and_dropdown(self):
        out = ManifoldProjector(Sphere()*Torus()*Mobius()).fit_transform(self.x)
        fig = out.plot_3d(target_index=0)
        self.assertEqual(len(fig.layout.updatemenus[0].buttons), 4)
        self.assertEqual(len(fig.data), 15)
        for button in fig.layout.updatemenus[0].buttons:
            self.assertEqual(len(button.args[0]["visible"]), len(fig.data))
        html = fig.to_html(include_plotlyjs=True)
        self.assertIn("non-isometric", html)
        self.assertGreater(len(html), 1_000_000)


if __name__ == "__main__":
    unittest.main()
