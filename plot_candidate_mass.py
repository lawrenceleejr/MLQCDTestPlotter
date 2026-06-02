"""
Stacked histograms of the per-event average candidate mass for two reconstruction
models (CombSolver, Mass-Asymmetry), under a few simple event-level cuts.

For each event the ROOT file stores two candidate "parent" particles per model
(branches are fixed-size arrays of length 2). For every event we compute

    m_avg = (m[0] + m[1]) / 2

and stack the four pT-bin samples by their relative weights. We then repeat
the plot under two loose kinematic cuts on the candidate pair:

  * mass-similarity:  |m0 - m1| / (m0 + m1) < MASS_ASYM_MAX
  * back-to-back:     |Δφ(cand0, cand1)|    > DPHI_MIN

Edit the CONFIG block below to retune.
"""

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

FILES = [
    "800to1000_TuneCP5_13p6TeV_madgraphMLM_output.root",
    "1000to1200_TuneCP5_13p6TeV_madgraphMLM_output.root",
    "1200to1500_TuneCP5_13p6TeV_madgraphMLM_output.root",
    "1500to2000_TuneCP5_13p6TeV_madgraphMLM_output.root",
]

LABELS = [
    r"800–1000 GeV",
    r"1000–1200 GeV",
    r"1200–1500 GeV",
    r"1500–2000 GeV",
]

# Relative weights (e.g. cross-section × lumi / N_events). Only ratios matter.
WEIGHTS = [3033, 883.7, 383.5, 125.2]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

TREE_NAME = "events"

# Each model is a (mass_branch, phi_branch, pretty_name) tuple.
MODELS = [
    ("CombSolverCandidate_mass",    "CombSolverCandidate_phi",    "CombSolver"),
    ("Mass_AsymmetryCandidate_mass", "Mass_AsymmetryCandidate_phi", "Mass-Asymmetry"),
]

# Histogram axis for the average mass
XMIN, XMAX, NBINS = 0, 3000, 60

# Cut thresholds (loose by design)
MASS_ASYM_MAX = 0.25   # |m0-m1| / (m0+m1) below this → "similar masses"
DPHI_MIN      = 2.5    # |Δφ| above this (radians, ≤π) → "back to back"

# Exponential fit on the falling tail: N(m) ∝ exp(-m/τ).
# Fit done as linear regression of log(N) vs m over bin centres in this range.
SLOPE_FIT_RANGE = (500.0, 2500.0)

OUTPUT_FILE = "avg_mass_stacked.pdf"

# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import uproot


def load_event_arrays(model):
    """For each input file, return (m_avg, mass_asym, dphi, weight_per_event)."""
    mass_branch, phi_branch, _ = model
    norm_weights = np.array(WEIGHTS, dtype=float)
    norm_weights /= norm_weights.sum()

    out = []
    for fpath, w in zip(FILES, norm_weights):
        with uproot.open(fpath) as f:
            tree = f[TREE_NAME]
            m   = tree[mass_branch].array(library="np")  # shape (N, 2)
            phi = tree[phi_branch].array(library="np")   # shape (N, 2)

        m0, m1 = m[:, 0], m[:, 1]
        m_avg     = 0.5 * (m0 + m1)
        denom     = m0 + m1
        mass_asym = np.where(denom > 0, np.abs(m0 - m1) / denom, np.inf)

        dphi = phi[:, 0] - phi[:, 1]
        dphi = np.abs(np.arctan2(np.sin(dphi), np.cos(dphi)))  # wrap to [0, π]

        out.append((m_avg, mass_asym, dphi, w))
    return out


CUT_SCENARIOS = [
    ("No cut",                    lambda a, d: np.ones_like(a, dtype=bool)),
    (f"|m0−m1|/(m0+m1) < {MASS_ASYM_MAX}",       lambda a, d: a < MASS_ASYM_MAX),
    (f"|Δφ| > {DPHI_MIN}",        lambda a, d: d > DPHI_MIN),
    (f"both cuts",                lambda a, d: (a < MASS_ASYM_MAX) & (d > DPHI_MIN)),
]


def fit_exponential(centers, total_counts, fit_range):
    """Linear fit of log(N) vs m → N(m) ∝ exp(-m/τ).
    Returns (tau, sigma_tau, x_fit, y_fit) or (None, None, None, None) on failure."""
    lo, hi = fit_range
    mask = (centers >= lo) & (centers <= hi) & (total_counts > 0)
    if mask.sum() < 3:
        return None, None, None, None

    x = centers[mask]
    y = np.log(total_counts[mask])
    # weights from Poisson-ish errors on counts: σ(log N) ≈ 1/√N
    w = np.sqrt(total_counts[mask])
    coeffs, cov = np.polyfit(x, y, deg=1, w=w, cov=True)
    slope, intercept = coeffs
    if slope >= 0:
        return None, None, None, None
    tau = -1.0 / slope
    sigma_slope = float(np.sqrt(cov[0, 0]))
    sigma_tau   = sigma_slope / slope**2  # propagation: τ = -1/slope

    x_fit = np.linspace(lo, hi, 50)
    y_fit = np.exp(intercept + slope * x_fit)
    return tau, sigma_tau, x_fit, y_fit


def stack_one_panel(ax, per_sample, cut_fn, title):
    bin_edges = np.linspace(XMIN, XMAX, NBINS + 1)
    centers   = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hists = []
    for m_avg, mass_asym, dphi, w in per_sample:
        mask   = cut_fn(mass_asym, dphi)
        values = m_avg[mask]
        ew     = np.full(len(values), w)
        counts, _ = np.histogram(values, bins=bin_edges, weights=ew)
        hists.append(counts)

    ax.hist(
        [centers] * len(hists),
        bins=bin_edges,
        weights=hists,
        label=LABELS,
        color=COLORS[:len(hists)],
        stacked=True,
        histtype="stepfilled",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlim(XMIN, XMAX)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    total_counts = np.sum(hists, axis=0)
    int_lo       = SLOPE_FIT_RANGE[0]
    integral     = float(total_counts[centers >= int_lo].sum())

    tau, sigma_tau, x_fit, y_fit = fit_exponential(centers, total_counts, SLOPE_FIT_RANGE)
    if tau is not None:
        ax.plot(x_fit, y_fit, color="k", linestyle="--", linewidth=1.4)
        slope_str = f"τ = {tau:.0f} ± {sigma_tau:.0f} GeV"
    else:
        slope_str = "τ = (fit failed)"

    info = (f"$\\int_{{{int(int_lo)}}}^{{\\infty}}$ = {integral:.3g}\n"
            f"exp fit ({int(SLOPE_FIT_RANGE[0])}–{int(SLOPE_FIT_RANGE[1])} GeV): {slope_str}")
    ax.text(0.97, 0.97, info, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.8,
                                  edgecolor="0.7", boxstyle="round,pad=0.3"))


def main():
    nrows = len(CUT_SCENARIOS)
    ncols = len(MODELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, None]

    per_model = {m[2]: load_event_arrays(m) for m in MODELS}

    for j, model in enumerate(MODELS):
        per_sample = per_model[model[2]]
        for i, (cut_name, cut_fn) in enumerate(CUT_SCENARIOS):
            ax = axes[i, j]
            title = f"{model[2]} — {cut_name}"
            stack_one_panel(ax, per_sample, cut_fn, title)
            if i == nrows - 1:
                ax.set_xlabel(r"$\langle m_{\mathrm{cand}}\rangle$ [GeV]", fontsize=12)
            if j == 0:
                ax.set_ylabel("Events (weighted)", fontsize=12)

    axes[0, 0].legend(fontsize=9, framealpha=0.85, loc="upper right")
    fig.suptitle("Per-event average candidate mass — stacked by pT bin",
                 fontsize=14, y=1.00)
    fig.tight_layout()

    if OUTPUT_FILE:
        fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"Saved → {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
