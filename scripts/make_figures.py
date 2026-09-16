"""Phase 5: generate the paper's figures.

Palette is slots 1-3 of the reference categorical theme (blue / orange /
aqua), which is the documented all-pairs-safe subset. Aqua falls below 3:1
contrast on a white surface, so every series carries a visible direct label;
markers and dash patterns give a second, non-colour channel for print and
colour-vision deficiency.

Figures are vector PDF for LaTeX plus PNG for quick viewing. Expensive
results are cached to data/figure_data.json -- rerun with --force to refresh.

Run: python scripts/make_figures.py [--force]
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbital.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
FIGDIR = ROOT / "writeup" / "figures"
CACHE = ROOT / "data" / "figure_data.json"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SERIES = [
    (BLUE, "o", "-"),
    (ORANGE, "s", "--"),
    (AQUA, "^", ":"),
]

# Measured in Phase 2 on a representative real conjunction.
PC_REF = 4.5072e-6
DRAWS_FOR_10PCT = 22_186_681
CATALOG_SIZE = 337_789
N_TRAIN = 64
SURROGATE_RMSE_LOG10 = 0.029


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,          # recessive grid, behind the marks
        "grid.color": "#e3e2dd",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def compute(force: bool = False) -> dict:
    """Expensive pieces, cached."""
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text())

    from designs import generate_maxpro, random_lhd, uniform_sample
    from montecarlo import pc_analytic, pc_monte_carlo
    from paramspace import DIM_4D, from_unit_cube_4d, unpack_4d
    from surrogate import fit_gp

    rng = np.random.default_rng(11)
    out: dict = {}

    # --- Monte Carlo convergence on one representative encounter ---
    # Chosen to match the Pc of the real conjunction measured in Phase 2;
    # picking an arbitrary box point gives Pc ~ 1e-15, where every budget
    # returns zero hits and there is no convergence to show.
    grid = rng.random((20000, DIM_4D))
    pcs = np.array([pc_analytic(*unpack_4d(r)) for r in from_unit_cube_4d(grid)])
    best = int(np.argmin(np.abs(np.log10(np.maximum(pcs, 1e-300)) - np.log10(PC_REF))))
    x = from_unit_cube_4d(grid[best:best + 1])[0]
    mu, cov, hbr = unpack_4d(x)
    truth = pc_analytic(mu, cov, hbr)
    print(f"  reference encounter: Pc = {truth:.3e}")
    draws, estimates = [], []
    for n in (10 ** k for k in range(3, 9)):
        est = pc_monte_carlo(mu, cov, hbr, int(n), rng)
        draws.append(int(n))
        estimates.append(est.pc)
    out["mc"] = {"truth": truth, "draws": draws, "estimates": estimates}

    # --- Pc dynamic range over the design box ---
    u = rng.random((4000, DIM_4D))
    logs = []
    for row in from_unit_cube_4d(u):
        m, c, h = unpack_4d(row)
        logs.append(np.log10(max(pc_analytic(m, c, h), 1e-300)))
    out["pc_distribution"] = logs

    # --- Surrogate accuracy vs training budget ---
    u_test = rng.random((3000, DIM_4D))
    y_test = np.array([np.log10(max(pc_analytic(*unpack_4d(r)), 1e-300))
                       for r in from_unit_cube_4d(u_test)])
    bench: dict = {}
    for name in ("maxpro", "random_lhd", "uniform"):
        means, spreads, sizes = [], [], []
        for n in (32, 64, 128, 256):
            rmses = []
            for rep in range(5):
                if name == "maxpro":
                    u_tr, _, _ = generate_maxpro(n, DIM_4D)
                elif name == "random_lhd":
                    u_tr = random_lhd(n, DIM_4D, rng)
                else:
                    u_tr = uniform_sample(n, DIM_4D, rng)
                y_tr = np.array([np.log10(max(pc_analytic(*unpack_4d(r)), 1e-300))
                                 for r in from_unit_cube_4d(u_tr)])
                gp = fit_gp(u_tr, y_tr, seed=11 + rep)
                rmses.append(float(np.sqrt(((gp.predict(u_test) - y_test) ** 2).mean())))
            means.append(float(np.mean(rmses)))
            spreads.append(float(np.std(rmses)))
            sizes.append(n)
        bench[name] = {"n": sizes, "rmse": means, "spread": spreads}
        print(f"  {name}: {[f'{m:.4f}' for m in means]}")
    out["surrogate"] = bench

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


def fig_motivation(data: dict) -> None:
    """Why brute force fails: it returns nothing, across most of the space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    truth = data["mc"]["truth"]
    draws = np.array(data["mc"]["draws"], dtype=float)
    est = np.array(data["mc"]["estimates"], dtype=float)
    zero = est <= 0
    rel = np.where(zero, 1.0, np.abs(est - truth) / truth)

    ax1.axhspan(1.0, 30.0, color="#f3f1ec", zorder=0)
    ax1.text(1.6e3, 6.0, "no usable estimate", fontsize=7.5,
             color=INK2, va="center")
    ax1.loglog(draws[~zero], rel[~zero], color=BLUE, lw=1.8, marker="o",
               ms=5, zorder=3)
    ax1.loglog(draws[zero], rel[zero], color=BLUE, lw=0, marker="x", ms=6,
               mew=1.6, zorder=3)
    ax1.axvline(DRAWS_FOR_10PCT, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax1.text(DRAWS_FOR_10PCT * 0.75, 3e-3, "22M draws\nfor 10%", fontsize=7.5,
             color=INK2, ha="right")
    ax1.axhline(0.1, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax1.set_xlabel("Monte Carlo draws")
    ax1.set_ylabel("relative error in $P_c$")
    exp = int(np.floor(np.log10(truth)))
    ax1.set_title(f"(a)  convergence at $P_c \\approx {truth/10**exp:.1f}"
                  f"\\times 10^{{{exp}}}$", loc="left")
    ax1.set_ylim(1e-3, 30)
    despine(ax1)

    logs = np.array(data["pc_distribution"])
    counts, _, _ = ax2.hist(logs, bins=45, color=BLUE, edgecolor="white",
                            linewidth=0.4)
    # Headroom so the budget labels sit above the bars rather than on them.
    ax2.set_ylim(0, counts.max() * 1.42)
    # Staggered vertically: the two reach lines sit only two decades apart,
    # so side-by-side labels would collide.
    for budget, label, height in ((1e8, "$10^8$ draws", 1.20),
                                  (1e6, "$10^6$ draws", 1.05)):
        reach = np.log10(10.0 / budget)          # ~10 expected hits
        ax2.axvline(reach, color=ORANGE, lw=1.2, ls="--", zorder=4)
        ax2.text(reach, counts.max() * height, label, fontsize=7.5,
                 color=ORANGE, ha="center", va="bottom")
    ax2.set_xlim(logs.min() - 0.5, -1.5)
    frac = float(np.mean(logs < np.log10(10.0 / 1e8)))
    ax2.text(0.03, 0.90, f"{frac:.0%} of encounters lie beyond\n"
                         f"reach even at $10^8$ draws",
             transform=ax2.transAxes, fontsize=7.5, color=INK2, va="top",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax2.set_xlabel("$\\log_{10} P_c$")
    ax2.set_ylabel("encounters")
    ax2.set_title("(b)  what brute force can label", loc="left")
    despine(ax2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig1_motivation.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_surrogate(data: dict) -> None:
    """Surrogate accuracy against training budget, by design type."""
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    labels = {"maxpro": "MaxPro (parallel tempering)",
              "random_lhd": "random LHD",
              "uniform": "uniform"}
    for (name, (color, marker, ls)) in zip(("maxpro", "random_lhd", "uniform"), SERIES, strict=False):
        d = data["surrogate"][name]
        n = np.array(d["n"], dtype=float)
        rmse = np.array(d["rmse"])
        spread = np.array(d["spread"])
        ax.errorbar(n, rmse, yerr=spread, color=color, lw=1.8, ls=ls,
                    marker=marker, ms=5.5, capsize=2.5, elinewidth=1.0,
                    label=labels[name], zorder=3)
    # No direct labels here: the three series converge at n=256, so end-of-line
    # labels collide. Identity is carried by marker shape and dash pattern as
    # well as hue, and the paper prints these values as a table.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256])
    ax.set_xticklabels(["32", "64", "128", "256"])
    ax.set_xlabel("training evaluations $n$")
    ax.set_ylabel("RMSE in $\\log_{10} P_c$  (orders of magnitude)")
    ax.set_title("Surrogate accuracy vs training budget", loc="left")
    ax.set_xlim(28, 300)
    ax.legend(loc="lower left")
    despine(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig2_surrogate.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_cost() -> None:
    """The headline: total compute to answer Q conjunctions."""
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    q = np.logspace(0, 6, 200)
    mc = q * DRAWS_FOR_10PCT
    surrogate = np.full_like(q, N_TRAIN * DRAWS_FOR_10PCT)

    ax.loglog(q, mc, color=ORANGE, lw=1.8, ls="--", label="brute-force Monte Carlo")
    ax.loglog(q, surrogate, color=BLUE, lw=1.8, ls="-", label="surrogate (64 training runs)")
    ax.annotate("Monte Carlo", xy=(3e5, mc[-30]), xytext=(-4, 6),
                textcoords="offset points", fontsize=7.5, color=ORANGE, ha="right")
    ax.annotate("surrogate", xy=(3e5, surrogate[0]), xytext=(-4, 6),
                textcoords="offset points", fontsize=7.5, color=BLUE, ha="right")

    ax.plot([N_TRAIN], [N_TRAIN * DRAWS_FOR_10PCT], marker="o", ms=6,
            color=INK, zorder=4)
    ax.annotate("break-even at 64 conjunctions", xy=(N_TRAIN, N_TRAIN * DRAWS_FOR_10PCT),
                xytext=(10, -16), textcoords="offset points", fontsize=7.5, color=INK2)

    speedup = CATALOG_SIZE / N_TRAIN
    ax.axvline(CATALOG_SIZE, color=MUTED, lw=1.0, ls=":")
    ax.annotate(f"catalog screen\n({CATALOG_SIZE:,} pairs)\n$\\approx${speedup:,.0f}$\\times$ cheaper",
                xy=(CATALOG_SIZE, 1e11), xytext=(-8, 0), textcoords="offset points",
                fontsize=7.5, color=INK2, ha="right", va="center")

    ax.set_xlabel("conjunctions evaluated")
    ax.set_ylabel("total Monte Carlo draws")
    ax.set_title("Compute cost at equal accuracy ($\\approx$7% in $P_c$)", loc="left")
    ax.legend(loc="upper left")
    despine(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig3_cost.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_triage() -> None:
    """Triage operating curves: catalog kept vs high-risk events retained."""
    from dataset import FEATURES
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    path = ROOT / "data" / "triage_dataset.csv"
    if not path.exists():
        print("  (skipping fig4: run src/build_dataset.py first)")
        return

    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    x = np.column_stack([raw[f] for f in FEATURES])
    y = (raw["log10_pc"] > -10).astype(int)
    day = raw["day"].astype(int)
    tr, te = ~np.isin(day, (5, 6)), np.isin(day, (5, 6))

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(x[tr], y[tr])
    gb = HistGradientBoostingClassifier(max_iter=300, random_state=7,
                                        class_weight="balanced")
    gb.fit(x[tr], y[tr])

    scores = [
        ("miss-distance cut", -x[te, FEATURES.index("miss_km")]),
        ("logistic regression", lr.decision_function(x[te])),
        ("gradient boosting", gb.decision_function(x[te])),
    ]

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    pos = y[te].astype(bool)
    for (name, s), (color, marker, ls) in zip(scores, SERIES, strict=False):
        order = np.argsort(-s)
        kept = np.arange(1, len(s) + 1) / len(s)
        recall = np.cumsum(pos[order]) / pos.sum()
        ax.plot(recall, kept, color=color, lw=1.8, ls=ls, label=name, zorder=3)
        # Direct label at the full-recall end of each curve.
        idx = int(np.searchsorted(recall, 1.0))
        idx = min(idx, len(kept) - 1)
        ax.plot([recall[idx]], [kept[idx]], marker=marker, ms=6, color=color, zorder=4)

    ax.set_yscale("log")
    ax.set_xlabel("recall of high-risk conjunctions")
    ax.set_ylabel("fraction of catalog kept for full analysis")
    ax.set_title("Triage operating curves (lower is better)", loc="left")
    ax.set_xlim(0.0, 1.02)
    ax.legend(loc="upper left")
    despine(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig4_triage.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    force = "--force" in sys.argv
    FIGDIR.mkdir(parents=True, exist_ok=True)
    style()
    print("computing figure data...")
    data = compute(force=force)
    print("rendering...")
    fig_motivation(data)
    fig_surrogate(data)
    fig_cost()
    fig_triage()
    for f in sorted(FIGDIR.glob("*.pdf")):
        print(f"  {f.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
