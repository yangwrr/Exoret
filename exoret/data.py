from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class Spectrum:
    wl: np.ndarray; hw: np.ndarray; depth: np.ndarray; err: np.ndarray
    name: str; kind: str   # kind: "real" or "synthetic"

def validate(s: Spectrum) -> Spectrum:
    for k in ("wl", "hw", "depth", "err"):
        a = getattr(s, k)
        if a.ndim != 1 or len(a) < 6 or not np.all(np.isfinite(a)):
            raise ValueError(f"{k}: need 1-D finite array with >=6 points")
    if not (len(s.wl) == len(s.hw) == len(s.depth) == len(s.err)): raise ValueError("length mismatch")
    if np.any(s.err <= 0) or np.any(s.hw <= 0) or np.any(s.wl <= 0): raise ValueError("err, hw, wl must be > 0")
    if np.any(np.diff(s.wl) <= 0): raise ValueError("wavelengths must be strictly increasing")
    return s

def load_csv(path, name=None, depth_unit="ppm"):
    """CSV columns: wavelength_um, bandwidth_um (full width), depth, depth_err. depth_unit: ppm|percent|fraction."""
    df = pd.read_csv(path, comment="#")
    need = {"wavelength_um", "bandwidth_um", "depth", "depth_err"}
    if not need <= set(df.columns): raise ValueError(f"missing columns: {need - set(df.columns)}")
    df = df.dropna(subset=list(need)).sort_values("wavelength_um")
    f = {"ppm": 1.0, "percent": 1e4, "fraction": 1e6}[depth_unit]
    return validate(Spectrum(df.wavelength_um.to_numpy(float), df.bandwidth_um.to_numpy(float) / 2,
                             df.depth.to_numpy(float) * f, df.depth_err.abs().to_numpy(float) * f,
                             name or str(path), "real"))
