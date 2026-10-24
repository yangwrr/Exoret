"""SIMULATED data for validation (injection-recovery). Not an observation."""
import numpy as np
from .data import Spectrum, validate
from .model import depth_ppm

def make_synthetic(theta, sysm, xsec, err_ppm=50.0, seed=42, wl_lo=0.9, wl_hi=1.7, nbin=36):
    edges = np.linspace(wl_lo, wl_hi, nbin + 1); wl = 0.5 * (edges[1:] + edges[:-1]); hw = 0.5 * np.diff(edges)
    clean = depth_ppm(wl, hw, theta, sysm, xsec); rng = np.random.default_rng(seed)
    err = np.full(nbin, err_ppm)
    return validate(Spectrum(wl, hw, clean + err * rng.standard_normal(nbin), err, "synthetic", "synthetic")), clean
