"""Usage:
  python run.py synthetic --case clear|cloudy        # injection-recovery validation (SIMULATED data)
  python run.py real --data data/raw/X.csv --rs 1.15 --mp 0.7 --name X [--depth-unit percent] [--xsec table.csv]
"""
import argparse, json, os, sys, time
import numpy as np
from exoret.model import System, ToyH2O, TableXsec
from exoret.data import load_csv
from exoret.synthetic import make_synthetic
from exoret.retrieval import Retrieval
from exoret import plots

TRUTH = {"clear": [1.35, 1200., -3.3, 1.0], "cloudy": [1.35, 1200., -3.3, -3.0]}
SYN_SYS = System(1.15, 0.7)  # hypothetical hot-Jupiter host/planet chosen by us, not a real system

def main(a=None):
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["synthetic", "real"])
    p.add_argument("--case", default="clear", choices=list(TRUTH)); p.add_argument("--data"); p.add_argument("--name", default="planet")
    p.add_argument("--rs", type=float); p.add_argument("--mp", type=float); p.add_argument("--depth-unit", default="ppm")
    p.add_argument("--xsec"); p.add_argument("--nsteps", type=int, default=6000); p.add_argument("--nburn", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1); a = p.parse_args(a)
    xs = TableXsec(a.xsec) if a.xsec else ToyH2O()
    if a.mode == "synthetic":
        truth = TRUTH[a.case]; sysm = SYN_SYS; spec, clean = make_synthetic(truth, sysm, xs); spec.name = f"synthetic-{a.case}"
    else:
        if not (a.data and a.rs and a.mp): sys.exit("real mode needs --data --rs --mp")
        spec = load_csv(a.data, a.name, a.depth_unit); sysm = System(a.rs, a.mp); truth = clean = None
    out = f"results/{spec.name}"; os.makedirs(out, exist_ok=True)
    t = time.time(); r = Retrieval(spec, sysm, xs).run(nburn=a.nburn, nsteps=a.nsteps, seed=a.seed)
    df, st = r.summary(truth); st["opacity"] = xs.label; st["data_kind"] = spec.kind; st["runtime_s"] = time.time() - t
    df.to_csv(f"{out}/parameters.csv", index=False); json.dump(st, open(f"{out}/stats.json", "w"), indent=1)
    np.savez_compressed(f"{out}/chain.npz", chain=r.chain, lp=r.lp)
    import pandas as pd; pd.DataFrame({"wl": spec.wl, "hw": spec.hw, "depth_ppm": spec.depth, "err_ppm": spec.err, "map_model_ppm": r.model(r.map)}).to_csv(f"{out}/spectrum_fit.csv", index=False)
    plots.workflow(f"{out}/fig1_workflow.png"); plots.spectrum(f"{out}/fig2_spectrum.png", r, spec)
    plots.corner(f"{out}/fig3_corner.png", r, truth); plots.trace(f"{out}/fig4_trace.png", r)
    print(df.to_string(index=False)); print(json.dumps(st, indent=1))

if __name__ == "__main__": main()
