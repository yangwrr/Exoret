"""Affine-invariant ensemble MCMC (Goodman & Weare 2010), parallel-split form as in Foreman-Mackey+2013."""
import numpy as np

def ensemble_sample(logp, p0, nsteps, a=2.0, seed=0):
    rng = np.random.default_rng(seed); nw, nd = p0.shape
    if nw % 2 or nw < 2 * nd: raise ValueError("need even nwalkers >= 2*ndim")
    x = p0.copy(); lp = np.array([logp(v) for v in x])
    if not np.all(np.isfinite(lp)): raise ValueError("initial walker outside prior")
    chain = np.empty((nsteps, nw, nd)); lps = np.empty((nsteps, nw)); acc = 0; h = nw // 2
    for t in range(nsteps):
        for s in (0, 1):
            idx = np.arange(s * h, (s + 1) * h); oth = np.arange((1 - s) * h, (2 - s) * h)
            z = ((a - 1) * rng.random(h) + 1) ** 2 / a
            c = x[rng.choice(oth, h)]; y = c + z[:, None] * (x[idx] - c)
            lpy = np.array([logp(v) for v in y])
            ok = np.log(rng.random(h)) < (nd - 1) * np.log(z) + lpy - lp[idx]
            x[idx[ok]] = y[ok]; lp[idx[ok]] = lpy[ok]; acc += ok.sum()
        chain[t] = x; lps[t] = lp
    return chain, lps, acc / (nsteps * nw)

def autocorr_time(chain, c=5.0):
    """Integrated autocorrelation time per parameter (Sokal window), walker-averaged ACF."""
    n, nw, nd = chain.shape; taus = []
    for d in range(nd):
        x = chain[:, :, d] - chain[:, :, d].mean(0)
        f = np.fft.rfft(x, n=2 * n, axis=0); acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].mean(1)
        acf /= acf[0]; tau = 2 * np.cumsum(acf) - 1
        m = np.arange(n) < c * tau; w = np.argmin(m) if not m.all() else n - 1
        taus.append(tau[w])
    return np.array(taus)
