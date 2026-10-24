"""Download real transmission spectra from the NASA Exoplanet Archive TAP service (needs internet).
Usage: python fetch_data.py "HD 209458 b"   -> data/raw/<planet>_<n>.csv per reference + raw dump.
Column names of the archive's `transitspec` table are matched by keyword; INSPECT the raw dump and the printed
mapping before trusting the output (this script could not be executed offline when written)."""
import sys, io, re, requests, pandas as pd
URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def main(planet):
    q = f"select * from transitspec where plntname='{planet}'"
    r = requests.get(URL, params={"query": q, "format": "csv"}, timeout=60); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty: sys.exit(f"no spectra for {planet}")
    slug = re.sub(r"\W+", "_", planet); df.to_csv(f"data/raw/{slug}_ARCHIVE_RAW.csv", index=False)
    pick = lambda *k: next(c for c in df.columns if all(x in c.lower() for x in k))
    wl, bw, dp = pick("central"), pick("bandwidth"), pick("transdep")
    er = [c for c in df.columns if "transdep" in c.lower() and "err" in c.lower()]
    print("mapping:", wl, bw, dp, er, "\nDepth unit in archive: CHECK documentation (commonly percent).")
    df = df.assign(wavelength_um=df[wl], bandwidth_um=df[bw], depth=df[dp], depth_err=df[er].abs().mean(axis=1))
    refcol = next((c for c in df.columns if "ref" in c.lower() or "bib" in c.lower()), None)
    for i, (k, g) in enumerate(df.groupby(refcol) if refcol else [("all", df)]):
        f = f"data/raw/{slug}_{i}.csv"; g[["wavelength_um", "bandwidth_um", "depth", "depth_err"]].to_csv(f, index=False); print(f, k, len(g))

if __name__ == "__main__": main(sys.argv[1])
