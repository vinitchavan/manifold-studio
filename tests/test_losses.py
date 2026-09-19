"""Mathematical and autograd regression tests for the optional training extra."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from manifold_studio import Plane, Circle, Sphere, Cylinder, Torus, Mobius, Embedding
    from manifold_studio.losses import make_loss, available_losses, fixed_spectral_basis
    from manifold_studio.torch_geometry import coordinates, distances, parameters_from_latent
    from manifold_studio.training import TrainableProjector


@unittest.skipUnless(HAS_TORCH, 'optional [train] extra is not installed')
class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_torch_numpy_geometry_parity(self):
        for g in [Plane(), Circle(2), Sphere(2), Cylinder(2), Torus(2, 3), Mobius(), Sphere()*Torus()*Plane()]:
            p = parameters_from_latent(g, torch.randn(6, g.intrinsic_dim, dtype=torch.double))
            np.testing.assert_allclose(coordinates(g, p).numpy(), g.from_parameters(p.numpy()), atol=1e-12)
            metric = 'intrinsic' if g.supports_intrinsic else 'ambient'
            np.testing.assert_allclose(distances(g, p, metric=metric).numpy(), g.distance(p.numpy(), metric=metric), atol=1e-12)

    def test_periodicity_and_mobius_seam(self):
        p = torch.tensor([[0.2, 0.1]], dtype=torch.double)
        t = p + torch.tensor([[2*np.pi, -2*np.pi]])
        self.assertLess(float(distances(Torus(), p, t)), 1e-6)
        q = p.clone(); q[:, 0] += 2*np.pi; q[:, 1] *= -1
        self.assertLess(float(distances(Mobius(), p, q, metric='ambient')), 1e-12)

    def test_original_squared_triplet_formula(self):
        p = torch.tensor([[0., 0.], [2., 0.], [1., 0.]])
        result = make_loss('euclidean_triplet', Plane(), margin=1)(p, triplets=[[0, 1, 2]])
        self.assertAlmostEqual(float(result.total), 4.)

    def test_separated_triplet_is_zero(self):
        p = torch.tensor([[0., 0.], [0.1, 0.], [3., 0.]], requires_grad=True)
        loss = make_loss('geodesic_triplet', Plane())(p, triplets=[[0, 1, 2]]).total
        self.assertEqual(float(loss.detach()), 0)
        loss.backward(); self.assertTrue(torch.isfinite(p.grad).all())

    def test_all_losses_have_finite_useful_gradients(self):
        source = torch.randn(6, 5, dtype=torch.double)
        basis = fixed_spectral_basis(source)
        for name in available_losses():
            p = torch.randn(6, 2, dtype=torch.double, requires_grad=True)
            q = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
            out = make_loss(name, Sphere(), margin=5)(p, source=source, basis=basis,
                query_parameters=q, triplets=[[0, 1, 2], [3, 4, 5]])
            out.total.backward()
            self.assertTrue(torch.isfinite(out.total), name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 1e-8, name)

    def test_finite_difference_composite_gradient(self):
        source = torch.randn(5, 4, dtype=torch.double)
        basis = fixed_spectral_basis(source)
        p = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
        q = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
        objective = make_loss('smtl', Sphere(), margin=5)
        def fn(p, q):
            return objective(p, source=source, basis=basis, query_parameters=q, triplets=[[0, 1, 2]]).total
        self.assertTrue(torch.autograd.gradcheck(fn, (p, q), atol=1e-4))

    def test_spectral_bounds_sign_invariance_and_detachment(self):
        source = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
        basis = fixed_spectral_basis(source).requires_grad_()
        p = torch.randn(7, 2, dtype=torch.double, requires_grad=True)
        q = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
        objective = make_loss('smtl', Sphere(), margin=5)
        kwargs = dict(source=source, triplets=[[0, 1, 2]], query_parameters=q)
        a = objective(p, basis=basis, **kwargs)
        signs = torch.tensor([1, -1, 1, -1, 1, -1, 1], dtype=torch.double)
        b = objective(p, basis=basis*signs, **kwargs)
        torch.testing.assert_close(a.total, b.total)
        for name in ['query_energy_separation', 'energy_concentration', 'distance_alignment']:
            self.assertGreaterEqual(float(a.components[name].detach()), 0)
            self.assertLessEqual(float(a.components[name].detach()), 1+1e-9)
        a.total.backward()
        self.assertIsNone(basis.grad)
        self.assertIsNone(source.grad)
        self.assertGreater(float(q.grad.abs().sum()), 1e-8)

    def test_invalid_inputs_and_metric_contract(self):
        with self.assertRaises(NotImplementedError): make_loss('geodesic_triplet', Mobius())
        with self.assertRaises(ValueError): make_loss('geodesic_triplet', Sphere(), metric='ambient')
        with self.assertRaises(ValueError): make_loss('smtl', Sphere(), alpha=-1)
        with self.assertRaises(ValueError): make_loss('smtl', Sphere(), temperature=0)
        with self.assertRaises(ValueError): make_loss('unknown', Sphere())
        with self.assertRaises(ValueError): fixed_spectral_basis(torch.ones(5, 3))
        with self.assertRaises(ValueError): make_loss('energy_concentration', Sphere())(torch.ones(5, 2))
        with self.assertRaises(ValueError): make_loss('geodesic_triplet', Sphere())(torch.ones(5, 2), triplets=[])
        with self.assertRaises(ValueError): make_loss('geodesic_triplet', Sphere())(torch.ones(5, 2), triplets=[[0, 0, 2]])
        # Zero weights disable optional supervision requirements.
        result = make_loss('smtl', Sphere(), alpha=0, beta=0, gamma=0)(torch.randn(3, 2), triplets=[[0, 1, 2]])
        self.assertTrue(torch.isfinite(result.total))

    def test_duplicate_points_do_not_produce_nan_gradients(self):
        for g in [Sphere(), Torus(), Cylinder(), Sphere()*Torus()]:
            p = torch.zeros(3, g.intrinsic_dim, dtype=torch.double, requires_grad=True)
            out = make_loss('geodesic_triplet', g)(p, triplets=[[0, 1, 2]])
            out.total.backward()
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_training_and_portable_replay(self):
        g = Sphere()*Torus()
        x = torch.randn(16, 5, dtype=torch.double)
        model = TrainableProjector(5, g, hidden_dim=12).double().fit_normalization(x)
        objective = make_loss('geodesic_triplet', g, margin=1)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        trips = [[0, 1, 8], [2, 3, 9], [4, 5, 10]]
        initial = float(objective(model(x), triplets=trips).total.detach())
        for _ in range(30):
            opt.zero_grad(); result = objective(model(x), triplets=trips)
            result.total.backward(); opt.step()
        final = float(objective(model(x), triplets=trips).total.detach())
        self.assertLess(final, initial)
        model.fit_display(x)
        unseen = torch.randn(4, 5, dtype=torch.double)
        a = model.transform(unseen)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp)/'model.npz')
            restored = TrainableProjector.load(Path(tmp)/'model.npz')
            b = restored.transform(unseen)
            np.testing.assert_allclose(a.coordinates, b.coordinates)
            np.testing.assert_allclose(b.coordinates[:1], restored.transform(unseen[:1]).coordinates)
            a.save(Path(tmp)/'embedding.npz')
            self.assertEqual(Embedding.load(Path(tmp)/'embedding.npz').mapping, 'trained MLP + parametrization')


if __name__ == '__main__':
    unittest.main()
