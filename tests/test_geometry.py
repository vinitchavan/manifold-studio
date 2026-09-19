import unittest
import numpy as np
from manifold_studio import Plane, Circle, Sphere, Cylinder, Torus, Mobius, Product
from manifold_studio.geometry import geometry_from_spec


class GeometryTests(unittest.TestCase):
    def test_dimensions_and_associativity(self):
        a = (Sphere()*Torus())*Plane()
        b = Sphere()*(Torus()*Plane())
        self.assertEqual(a.spec(), b.spec())
        self.assertEqual((a.intrinsic_dim, a.ambient_dim), (6, 9))

    def test_sphere_membership_and_pole_tangent(self):
        g = Sphere(2)
        p = np.array([[0, np.pi/2], [1, 0.3]])
        x = g.from_parameters(p)
        v = g.tangent_project(p, np.ones((2, 3)))
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 2)
        np.testing.assert_allclose(np.sum(v*x, axis=1), 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(v[0]), 0)

    def test_circle_arc_and_cylinder(self):
        np.testing.assert_allclose(Circle(2).distance([[0]], [[np.pi]]), [[2*np.pi]])
        np.testing.assert_allclose(Cylinder().distance([[0, 0]], [[np.pi, 4]]), [[np.hypot(np.pi, 4)]])

    def test_flat_torus_metric(self):
        g = Torus(2, 3)
        np.testing.assert_allclose(g.distance([[0, 0]], [[np.pi, np.pi]]), [[np.pi*np.sqrt(13)]])
        self.assertEqual(g.from_parameters([[0, 0]]).shape, (1, 4))

    def test_mobius_seam_and_tangent_transition(self):
        g = Mobius()
        a, b = np.array([[0., 0.2]]), np.array([[2*np.pi, -0.2]])
        np.testing.assert_allclose(g.from_parameters(a), g.from_parameters(b), atol=1e-12)
        ja, jb = g.jacobian(a), g.jacobian(b)
        np.testing.assert_allclose(ja[:, :, 0], jb[:, :, 0], atol=1e-12)
        np.testing.assert_allclose(ja[:, :, 1], -jb[:, :, 1], atol=1e-12)

    def test_mobius_no_fake_geodesic(self):
        g = Sphere()*Mobius()
        with self.assertRaises(NotImplementedError):
            g.distance(np.zeros((2, 4)))
        np.testing.assert_allclose(g.distance(np.zeros((2, 4)), metric="ambient"), 0)

    def test_product_distance_pythagoras(self):
        g = Circle(2)*Plane()
        d = g.distance([[0, 0, 0]], [[np.pi/2, 3, 4]])
        np.testing.assert_allclose(d, [[np.sqrt(np.pi**2+25)]])

    def test_jacobians_finite_difference(self):
        for g in [Plane(), Circle(), Sphere(), Cylinder(), Torus(), Mobius(), Sphere()*Mobius()*Circle()]:
            p = g.parameters_from_latent(np.full((3, g.intrinsic_dim), 0.3))
            j = g.jacobian(p)
            for i in range(g.intrinsic_dim):
                step = np.zeros_like(p); step[:, i] = 1e-6
                numerical = (g.from_parameters(p+step)-g.from_parameters(p-step))/2e-6
                np.testing.assert_allclose(j[:, :, i], numerical, atol=1e-8)

    def test_tangent_projection_is_orthogonal_and_idempotent(self):
        rng = np.random.default_rng(3)
        for g in [Sphere(), Cylinder(), Torus(), Mobius(), Sphere()*Torus()*Mobius()]:
            p = g.parameters_from_latent(rng.normal(size=(10, g.intrinsic_dim)))
            v = rng.normal(size=(10, g.ambient_dim))
            t = g.tangent_project(p, v)
            np.testing.assert_allclose(g.tangent_project(p, t), t, atol=1e-12)
            np.testing.assert_allclose(np.einsum("nai,na->ni", g.jacobian(p), v-t), 0, atol=1e-12)

    def test_log_map_and_antipodes(self):
        g = Sphere(2)
        np.testing.assert_allclose(g.log_map([[0, 0]], [[np.pi/2, 0]]), [[0, np.pi, 0]], atol=1e-12)
        with self.assertRaises(ValueError):
            g.log_map([[0, 0]], [[np.pi, 0]])

    def test_serializable_geometry(self):
        for g in [Plane(), Circle(2), Sphere(3), Cylinder(2), Torus(2, 3), Mobius(), Sphere()*Mobius()*Circle()]:
            self.assertEqual(geometry_from_spec(g.spec()).spec(), g.spec())

    def test_invalid_geometry(self):
        for make in [lambda: Sphere(-1), lambda: Torus(0), lambda: Mobius(width=2), lambda: Product(Sphere())]:
            with self.assertRaises(ValueError): make()


if __name__ == "__main__":
    unittest.main()
