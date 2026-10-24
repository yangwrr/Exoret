import numpy as np
from scipy.optimize import differential_evolution
from .model import depth_ppm, R_JUP, R_SUN, PARAM_NAMES
from .sampler import ensemble_sample, autocorr_time

class Retrieval:
    def __init__(self, spec, sysm, xsec):
        self.s, self.sys, self.xs = spec, sysm, xsec
        r = sysm.rs_rsun * R_SUN * np.sqrt(np.mean(spec.depth) * 1e-6) / R_JUP
        self.bounds = np.array([[0.7 * r, 1.3 * r], [300, 2500], [-8, -0.5], [-6, 1]])
        self.n = len(spec.wl)

    def model(self, th):
        return depth_ppm(self.s.wl, self.s.hw, th, self.sys, self.xs)

    def chi2(self, th):
        return np.sum(((self.s.depth - self.model(th)) / self.s.err) ** 2)

    def logp(self, th):
        if np.any(th < self.bounds[:, 0]) or np.any(th > self.bounds[:, 1]): return -np.inf
        c = self.chi2(th)
        return -0.5 * c if np.isfinite(c) else -np.inf

    def flat(self):
        w = 1 / self.s.err ** 2; m = np.sum(w * self.s.depth) / w.sum()
        return m, np.sum(((self.s.depth - m) / self.s.err) ** 2)

    def run(self, nwalkers=32, nburn=2000, nsteps=6000, seed=1):
        def cost(t):
            lp = self.logp(t); return -lp if np.isfinite(lp) else 1e30
        de = differential_evolution(cost, self.bounds, seed=seed, maxiter=300, popsize=20, tol=1e-8, polish=False)
        self.map = de.x; w = self.bounds[:, 1] - self.bounds[:, 0]; rng = np.random.default_rng(seed)
        p0 = np.clip(self.map + 1e-2 * w * rng.standard_normal((nwalkers, 4)),
                     self.bounds[:, 0] + 1e-6 * w, self.bounds[:, 1] - 1e-6 * w)
        ch, lp, acc = ensemble_sample(self.logp, p0, nburn + nsteps, seed=seed)
        self.chain_full, self.lp_full, self.acc = ch, lp, acc
        self.chain, self.lp = ch[nburn:], lp[nburn:]
        self.tau = autocorr_time(self.chain)
        self.flat_samples = self.chain.reshape(-1, 4)
        i = np.unravel_index(np.argmax(self.lp), self.lp.shape); self.map = self.chain[i]
        return self

    def summary(self, truth=None):
        import pandas as pd
        q = np.percentile(self.flat_samples, [16, 50, 84], axis=0)
        d = {"parameter": PARAM_NAMES, "median": q[1], "err_minus": q[1] - q[0], "err_plus": q[2] - q[1], "MAP": self.map}
        if truth is not None: d["truth"] = truth
        c2 = self.chi2(self.map); m, c2f = self.flat(); k = 4
        stats = {"chi2_full": c2, "dof_full": self.n - k, "chi2_flat": c2f, "dof_flat": self.n - 1,
                 "BIC_full": c2 + k * np.log(self.n), "BIC_flat": c2f + np.log(self.n),
                 "acceptance": self.acc, "tau_max": float(self.tau.max()),
                 "n_eff": float(len(self.flat_samples) / self.tau.max())}
        stats["dBIC_flat_minus_full"] = stats["BIC_flat"] - stats["BIC_full"]
        return pd.DataFrame(d), stats
