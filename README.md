# Atmospheric Retrieval of Transiting Exoplanets from Transmission Spectra

## Run
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m unittest tests.test_all                      # 7 tests
python run.py synthetic --case clear                   # ~20 s  (SIMULATED data, validation)
python run.py synthetic --case cloudy --nsteps 25000 --nburn 4000
python fetch_data.py "WASP-39 b"                       # needs internet; check column mapping printed
python run.py real --data data/raw/WASP_39_b_0.csv --rs <R_star/R_sun> --mp <M_p/M_Jup> --name WASP-39b --depth-unit percent
```
Outputs in `results/<name>/`: parameters.csv, stats.json, spectrum_fit.csv, chain.npz, fig1-4 PNGs.

## Physics (isothermal, constant-g, H2-dominated)
- Scale height H = kT/(mu m_H g), g = GM_p/R0^2, mu = 2.3.
- Slant optical depth through an exponential atmosphere: tau(z) = sigma n0 sqrt(2 pi R0 H) exp(-z/H); setting tau = 0.56 gives the
  effective altitude z = H ln(n0 sigma sqrt(2 pi R0 H)/0.56) (Lecavelier des Etangs+2008; de Wit & Seager 2013).
- sigma = X_H2O sigma_H2O(lambda) + 0.85 sigma_Rayleigh,H2 (sigma = 5.31e-31 m^2 (lambda/350 nm)^-4).
- Grey cloud deck at Pc: z >= H ln(P0/Pc), P0 = 10 bar. Depth = (R_p(lambda)/R_s)^2, bin-averaged (5 sub-points).
- Free parameters: R0 (radius at 10 bar), T, log10 X_H2O, log10 Pc. Uniform priors (see Retrieval.bounds).
- Inference: Gaussian likelihood; differential evolution for MAP; affine-invariant ensemble MCMC (own implementation, tested on a
  correlated Gaussian); autocorrelation time, acceptance fraction; BIC vs a flat (featureless) model.

## Results obtained so far (all SIMULATED data, hypothetical planet Rs=1.15 Rsun, Mp=0.7 MJup, 50 ppm errors, 36 bins 0.9-1.7 um)
| case | param | truth | median (+/-) |
|---|---|---|---|
| clear | R0 | 1.35 | 1.327 (+0.016/-0.012) |
| clear | T | 1200 | 1246 (+46/-42) |
| clear | log X_H2O | -3.3 | -1.7 (+0.9/-1.1) |
| clear | log Pc | 1.0 (no cloud) | -1.0 (+1.4/-1.5) |
| cloudy | R0 | 1.35 | 1.344 (+0.028/-0.028) |
| cloudy | T | 1200 | 1281 (+170/-150) |
| cloudy | log X_H2O | -3.3 | -3.1 (+1.8/-1.9) |
| cloudy | log Pc | -3.0 | -3.2 (+1.9/-1.8) |

Both fits have chi2/dof ~0.67-0.76; dBIC(flat - full) = 1548 (clear), 234 (cloudy). All truths lie within ~1.6 sigma.
Abundance, reference radius and cloud pressure are strongly degenerate (R0 is defined at fixed P0) - a genuine feature of
transmission retrievals, visible in the corner plot, not a bug.

## Limitations (state these in the report)
1. **No real data was analysed in the build environment (no internet).** Real-data mode and fetch_data.py are written but unrun;
   archive column names/depth units must be verified.
2. **H2O opacity is a toy band model** (real band positions, order-of-magnitude amplitudes), so retrieved abundances are NOT physical
   for real planets. For science results supply ExoMol/HITRAN-derived cross sections via `--xsec` (CSV: wavelength_um, sigma_cm2).
3. Isothermal, single absorber, constant g, no CIA/Na/K/CO2/CH4, no stellar contamination, no offsets between instruments, mu fixed.
4. Synthetic validation uses the same model for generating and fitting (tests sampler/pipeline, not model adequacy).
5. BIC is used rather than nested-sampling evidence.
