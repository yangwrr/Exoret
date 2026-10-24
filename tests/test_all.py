import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from exoret import model as M
from exoret.sampler import ensemble_sample, autocorr_time
from exoret.data import Spectrum, validate, load_csv
from exoret.synthetic import make_synthetic

S = M.System(1.15, 0.7); X = M.ToyH2O()

class T(unittest.TestCase):
    def test_scale_height_and_slope(self):
        g = M.G * 0.7 * M.M_JUP / (1.35 * M.R_JUP) ** 2; H = M.K_B * 1200 / (M.MU * M.M_H * g)
        r = lambda w, x: M.radius(np.array([w]), [1.35, 1200, x, 1.0], S, X)[0]
        # pure-Rayleigh limit: dR/dln(lam) = -4H
        a, b = r(0.50, -12), r(0.55, -12)
        self.assertAlmostEqual((b - a) / np.log(0.55 / 0.50) / H, -4, delta=0.05)
    def test_depth_matches_geometry(self):
        d = M.depth_ppm(np.array([1.0]), np.array([1e-4]), [1.35, 1200, -12, 1.0], S, X)[0]
        R = M.radius(np.array([1.0]), [1.35, 1200, -12, 1.0], S, X)[0]   # R0 is radius at 10 bar, so use R(lam)
        self.assertAlmostEqual(d / 1e6, (R / (1.15 * M.R_SUN)) ** 2, delta=1e-4 * d / 1e6)
    def test_abundance_monotonic_and_cloud_flattens(self):
        w = np.array([1.38]); f = lambda x, c: M.radius(w, [1.35, 1200, x, c], S, X)[0]
        self.assertLess(f(-6, 1.0), f(-3, 1.0))
        self.assertAlmostEqual(f(-3, -5.5), f(-6, -5.5), delta=1.0)  # cloud deck above gas photosphere
    def test_sampler_gaussian(self):
        C = np.array([[1., .8], [.8, 2.]]); Ci = np.linalg.inv(C); mu = np.array([1., -2.])
        lp = lambda x: -0.5 * (x - mu) @ Ci @ (x - mu)
        p0 = np.random.default_rng(0).normal(size=(16, 2)) + mu
        ch, _, acc = ensemble_sample(lp, p0, 4000, seed=3); s = ch[500:].reshape(-1, 2)
        self.assertTrue(np.allclose(s.mean(0), mu, atol=0.1)); self.assertTrue(np.allclose(np.cov(s.T), C, atol=0.25))
        self.assertTrue(0.2 < acc < 0.8); self.assertTrue(np.all(autocorr_time(ch[500:]) > 1))
    def test_validation_errors(self):
        n = np.arange(1, 9.)
        with self.assertRaises(ValueError): validate(Spectrum(n, n * 0 + .1, n, n * 0, "x", "real"))   # err=0
        with self.assertRaises(ValueError): validate(Spectrum(n[::-1].copy(), n * 0 + .1, n, n, "x", "real"))
        with self.assertRaises(ValueError): validate(Spectrum(n, n * 0 + .1, n * np.nan, n, "x", "real"))
    def test_loader(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("wavelength_um,bandwidth_um,depth,depth_err\n" + "\n".join(f"{1+.1*i},0.1,1.5,0.01" for i in range(8)))
        s = load_csv(f.name, depth_unit="percent"); self.assertAlmostEqual(s.depth[0], 15000); self.assertEqual(s.kind, "real")
        with open(f.name, "w") as g: g.write("a,b\n1,2\n")
        with self.assertRaises(ValueError): load_csv(f.name)
    def test_synthetic_reproducible(self):
        a, _ = make_synthetic([1.35, 1200, -3.3, 1.0], S, X); b, _ = make_synthetic([1.35, 1200, -3.3, 1.0], S, X)
        self.assertTrue(np.array_equal(a.depth, b.depth)); self.assertEqual(a.kind, "synthetic")

if __name__ == "__main__": unittest.main()
