# MLQCDTestPlotter

Plotting utilities for comparing two "parent candidate" reconstruction models on
QCD MC samples (Madgraph + Pythia, TuneCP5, 13.6 TeV), binned in HT.

The ROOT files store, per event, two candidate "parents" for each of two models:

| Model            | Mass branch                      | Phi branch                      |
|------------------|----------------------------------|---------------------------------|
| CombSolver       | `CombSolverCandidate_mass[2]`    | `CombSolverCandidate_phi[2]`    |
| Mass-Asymmetry   | `Mass_AsymmetryCandidate_mass[2]`| `Mass_AsymmetryCandidate_phi[2]`|

See `filestructure.md` for the full tree dump.

## What the script plots

`plot_candidate_mass.py` builds a stacked histogram of the per-event **average
candidate mass**

```
m_avg = 0.5 * (m[0] + m[1])
```

for each model, stacked across the four pT-binned MC samples weighted by their
relative cross sections. The same distribution is drawn under four event-level
selections so you can see how a few simple kinematic cuts reshape it:

1. **No cut** — baseline.
2. **Mass similarity** — require the two candidate masses to be compatible:
   `|m0 - m1| / (m0 + m1) < 0.25`.
3. **Back-to-back** — require the two candidates to be roughly opposite in φ:
   `|Δφ(cand0, cand1)| > 2.5` rad.
4. **Both cuts** applied together.

Output is a 4×2 grid (`avg_mass_stacked.pdf`): rows are cut scenarios, columns
are the two models.

## Running

```bash
pip install -r requirements.txt
python plot_candidate_mass.py
```

The four input ROOT files referenced in the `FILES` list at the top of the
script must be in the working directory. Tune `WEIGHTS`, the histogram binning,
and the cut thresholds (`MASS_ASYM_MAX`, `DPHI_MIN`) at the top of the file.

## Files

- `plot_candidate_mass.py` — the plotting script.
- `filestructure.md` — reference dump of the ROOT tree branches.
- `requirements.txt` — Python dependencies.
