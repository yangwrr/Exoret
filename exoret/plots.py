import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .model import PARAM_NAMES

def workflow(path):
    steps = ["Public spectrum\n(NASA Exoplanet Archive)\nor synthetic test", "Validate &\npreprocess", "Forward model\n(H, tau=0.56, H2O,\nRayleigh, cloud)",
             "Bayesian retrieval\n(DE optimiser +\nensemble MCMC)", "Posteriors, best fit,\nBIC vs flat model"]
    fig, ax = plt.subplots(figsize=(12, 2.4)); ax.axis("off"); ax.set_xlim(0, 5.2); ax.set_ylim(0, 1)
    for i, t in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + .05, .15), .9, .7, boxstyle="round,pad=0.02", fc="#dbe9f6", ec="#1f4e79"))
        ax.text(i + .5, .5, t, ha="center", va="center", fontsize=8.5)
        if i < 4: ax.annotate("", xy=(i + 1.07, .5), xytext=(i + .98, .5), arrowprops=dict(arrowstyle="->"))
    fig.suptitle("Figure 1: Retrieval workflow", fontsize=10); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

def spectrum(path, r, spec, truth_clean=None):
    rng = np.random.default_rng(0); idx = rng.choice(len(r.flat_samples), 300, replace=False)
    wl = np.linspace(spec.wl[0] - spec.hw[0], spec.wl[-1] + spec.hw[-1], 200); hw = np.full(200, (wl[1] - wl[0]) / 2)
    from .model import depth_ppm
    mods = np.array([depth_ppm(wl, hw, r.flat_samples[i], r.sys, r.xs) for i in idx])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(wl, *np.percentile(mods, [2.5, 97.5], 0), color="C0", alpha=.2, label="Posterior 95%")
    ax.fill_between(wl, *np.percentile(mods, [16, 84], 0), color="C0", alpha=.4, label="Posterior 68%")
    ax.plot(wl, depth_ppm(wl, hw, r.map, r.sys, r.xs), "C0", label="MAP model")
    ax.axhline(r.flat()[0], color="gray", ls="--", label="Flat (featureless) model")
    ax.errorbar(spec.wl, spec.depth, spec.err, spec.hw, "k.", capsize=2, label=f"Data ({spec.kind})")
    ax.set_xlabel("Wavelength [µm]"); ax.set_ylabel("Transit depth [ppm]"); ax.legend(fontsize=8)
    ax.set_title(f"Transmission spectrum: {spec.name}"); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

def corner(path, r, truth=None, bins=40):
    s = r.flat_samples; n = s.shape[1]; fig, axs = plt.subplots(n, n, figsize=(9, 9))
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            if j > i: ax.axis("off"); continue
            if i == j:
                ax.hist(s[:, i], bins, color="C0", histtype="stepfilled", alpha=.6)
                lo, md, hi = np.percentile(s[:, i], [16, 50, 84]); ax.axvline(md, c="k", lw=1)
                ax.set_title(f"{md:.3g} +{hi-md:.2g} -{md-lo:.2g}", fontsize=8); ax.set_yticks([])
                if truth is not None: ax.axvline(truth[i], c="r", ls="--")
            else:
                ax.hist2d(s[:, j], s[:, i], bins=bins, cmap="Blues")
                if truth is not None: ax.axvline(truth[j], c="r", ls="--", lw=.8); ax.axhline(truth[i], c="r", ls="--", lw=.8)
            if i < n - 1: ax.set_xticklabels([])
            else: ax.set_xlabel(PARAM_NAMES[j], fontsize=8)
            if j > 0 and i != j: ax.set_yticklabels([])
            elif j == 0 and i > 0: ax.set_ylabel(PARAM_NAMES[i], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)

def trace(path, r):
    fig, axs = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    for d, ax in enumerate(axs):
        ax.plot(r.chain_full[:, :, d], lw=.3, alpha=.5); ax.set_ylabel(PARAM_NAMES[d], fontsize=8)
        ax.axvline(len(r.chain_full) - len(r.chain), c="r")
    axs[-1].set_xlabel("Step (red line = end of burn-in)"); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
