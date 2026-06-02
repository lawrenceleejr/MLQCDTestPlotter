"""
Stacked histograms of CombSolverCandidate_mass and Mass_AsymmetryCandidate_mass
from four ROOT files, saved as side-by-side subplots (or individually).
Edit the CONFIG block below — nothing else should need changing.
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
WEIGHTS = [
    3033,
    883.7,
    383.5,
    125.2,
]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# Tree name
TREE_NAME = "events"

# ── Per-branch config ─────────────────────────────────────────────────────────
# Each entry: (branch_name, candidate_index, xmin, xmax, nbins, xlabel, title)
# candidate_index: 0, 1, or "both"  (only relevant for fixed-size array branches)

BRANCHES = [
    dict(
        branch      = "CombSolverCandidate_mass",
        cand_index  = 0,
        xmin        = 0,
        xmax        = 3000,
        nbins       = 60,
        xlabel      = r"$m_{\mathrm{cand}}$ [GeV]",
        title       = r"CombSolverCandidate mass",
    ),
    dict(
        branch      = "Mass_AsymmetryCandidate_mass",
        cand_index  = 0,          # change to 1 / "both" if needed
        xmin        = 0,
        xmax        = 3000,
        nbins       = 60,
        xlabel      = r"$m_{\mathrm{asym}}$ [GeV]",
        title       = r"MassAsymmetryCandidate mass",
    ),
]

YLABEL      = "Events (weighted)"
OUTPUT_FILE = "mass_stacked.pdf"   # set to None to show interactively

# ──────────────────────────────────────────────────────────────────────────────
# SCRIPT  (no need to edit below here)
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import uproot

# Normalise weights once
weights_arr = np.array(WEIGHTS, dtype=float)
weights_arr /= weights_arr.sum()


def build_hists(branch_cfg):
    """Return list of per-sample count arrays for one branch config."""
    bin_edges = np.linspace(branch_cfg["xmin"], branch_cfg["xmax"], branch_cfg["nbins"] + 1)
    hists = []
    for fpath, w in zip(FILES, weights_arr):
        with uproot.open(fpath) as f:
            arr = f[TREE_NAME][branch_cfg["branch"]].array(library="np")

        # Handle both scalar-per-event and fixed-size array branches
        if arr.ndim == 1:
            values = arr
            ew = np.full(len(values), w)
        else:
            idx = branch_cfg["cand_index"]
            if idx == "both":
                values = arr.flatten()
                ew = np.full(len(values), w / arr.shape[1])
            else:
                values = arr[:, idx]
                ew = np.full(len(values), w)

        counts, _ = np.histogram(values, bins=bin_edges, weights=ew)
        hists.append(counts)
    return bin_edges, hists


def draw_stack(ax, bin_edges, hists, cfg):
    ax.hist(
        [bin_edges[:-1]] * len(hists),
        bins=bin_edges,
        weights=hists,
        label=LABELS,
        color=COLORS[:len(hists)],
        stacked=True,
        histtype="stepfilled",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel(cfg["xlabel"], fontsize=13)
    ax.set_ylabel(YLABEL, fontsize=13)
    ax.set_title(cfg["title"], fontsize=14)
    ax.set_xlim(cfg["xmin"], cfg["xmax"])
    ax.set_yscale("log")
    ax.legend(fontsize=10, framealpha=0.85)
    ax.tick_params(axis="both", labelsize=11)


# ── Build & plot ──────────────────────────────────────────────────────────────
ncols = len(BRANCHES)
fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 6))
if ncols == 1:
    axes = [axes]

for ax, cfg in zip(axes, BRANCHES):
    bin_edges, hists = build_hists(cfg)
    draw_stack(ax, bin_edges, hists, cfg)

plt.tight_layout()

if OUTPUT_FILE:
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Saved → {OUTPUT_FILE}")
else:
    plt.show()