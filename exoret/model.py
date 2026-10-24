"""Forward model: isothermal, constant-gravity, H2-dominated transmission spectrum.

R_p(lam) = R0 + max(z_gas(lam), z_cloud),   delta(lam) = (R_p(lam)/R_s)^2
z_gas   = H ln( n0 sigma(lam) sqrt(2 pi R0 H) / tau_eq )   (Lecavelier des Etangs+2008; de Wit & Seager 2013)
z_cloud = H ln(P0 / Pc)                                    (grey opaque cloud deck at pressure Pc)
H = kT/(mu m_H g),  n0 = P0/(kT),  sigma = X_H2O sigma_H2O + X_H2 sigma_Rayleigh
"""
from dataclasses import dataclass
import numpy as np

K_B, M_H, G = 1.380649e-23, 1.6735575e-27, 6.67430e-11
R_JUP, M_JUP, R_SUN = 7.1492e7, 1.89813e27, 6.957e8
TAU_EQ, P0_BAR, MU, X_H2 = 0.56, 10.0, 2.3, 0.85
SIG_RAY0, LAM_RAY0 = 5.31e-31, 0.35  # m^2 at 350 nm (Lecavelier des Etangs+2008)
PARAM_NAMES = ["R0 [R_Jup]", "T [K]", "log10 X_H2O", "log10 Pc [bar]"]

@dataclass
class System:
    rs_rsun: float   # stellar radius
    mp_mjup: float   # planet mass

def rayleigh_h2(wl_um):
    return SIG_RAY0 * (np.asarray(wl_um) / LAM_RAY0) ** -4

class ToyH2O:
    """ILLUSTRATIVE band-shaped H2O cross-section (NOT line-by-line, NOT from a line list).
    Band centres are real H2O band positions; amplitudes are order-of-magnitude only.
    Replace with TableXsec built from ExoMol/HITRAN for scientific results."""
    label = "toy"
    bands = [(0.94, 0.02, 2e-26), (1.13, 0.03, 5e-26), (1.38, 0.04, 2e-25), (1.87, 0.06, 3e-25)]  # m^2
    cont = 1e-28
    def __call__(self, wl_um):
        wl = np.asarray(wl_um, float); s = np.full_like(wl, self.cont)
        for c, w, a in self.bands:
            s = s + a * np.exp(-0.5 * ((wl - c) / w) ** 2)
        return s

class TableXsec:
    """Cross-section table (columns: wavelength_um, sigma_cm2), log-interpolated. Fixed temperature."""
    label = "table"
    def __init__(self, path):
        a = np.loadtxt(path, delimiter=",", comments="#")
        if a.ndim != 2 or a.shape[1] < 2 or np.any(a[:, 1] <= 0):
            raise ValueError("cross-section file needs 2 columns, sigma>0")
        a = a[np.argsort(a[:, 0])]; self.wl, self.ls = a[:, 0], np.log(a[:, 1] * 1e-4)
    def __call__(self, wl_um):
        return np.exp(np.interp(wl_um, self.wl, self.ls))

def radius(wl_um, theta, sysm, xsec):
    r0, T, lx, lpc = theta
    r0 = r0 * R_JUP
    g = G * sysm.mp_mjup * M_JUP / r0 ** 2
    H = K_B * T / (MU * M_H * g)
    sig = 10 ** lx * xsec(wl_um) + X_H2 * rayleigh_h2(wl_um)
    n0 = P0_BAR * 1e5 / (K_B * T)
    z = H * np.log(n0 * sig * np.sqrt(2 * np.pi * r0 * H) / TAU_EQ)
    zc = H * np.log(P0_BAR / 10 ** lpc)
    return r0 + np.maximum(z, zc)

def depth_ppm(wl, hw, theta, sysm, xsec, nsub=5):
    """Binned transit depth [ppm]; averages model over nsub points in each [wl-hw, wl+hw] bin."""
    f = (np.arange(nsub) + 0.5) / nsub * 2 - 1
    grid = wl[:, None] + hw[:, None] * f[None, :]
    r = radius(grid, theta, sysm, xsec)
    return 1e6 * np.mean((r / (sysm.rs_rsun * R_SUN)) ** 2, axis=1)
