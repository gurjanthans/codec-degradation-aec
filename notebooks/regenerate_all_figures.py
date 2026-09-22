"""
Regenerate every figure in the manuscript from the JSON result files.

Usage:
    python regenerate_all_figures.py

Expects the result JSON files in ./results/ (edit RESULTS_DIR if they live elsewhere).
Writes PDF and PNG for all six manuscript figures into ./results/figs/.

Figures:
    fig_bitrate_curve       (Fig 1)  <- bitrate_sweep.json
    fig_regimes             (Fig 2)  <- regimes.json
    fig_bandlimit_comparison(Fig 3)  <- bandlimit_comparison.json, bandlimit_statistics.json
    fig_multicodec          (Fig 4)  <- second_codec_results.json
    fig_bandlimit_acoustic  (Fig 5)  <- acoustic_verification.json
    fig_freqdep_esc50       (Fig 6)  <- experiment3_frequency_dependence.json
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RESULTS_DIR = Path("results")
FIGS = RESULTS_DIR / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "savefig.bbox": "tight"})

CLEAN_BASELINE = 0.786
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3400),
         (3400, 4000), (4000, 8000), (8000, 11025)]


def load(name):
    return json.load(open(RESULTS_DIR / name))


# ---------------------------------------------------------------- Fig 1
def fig_bitrate_curve():
    sweep = load("bitrate_sweep.json")
    modes = [4.75, 5.15, 5.90, 6.70, 7.40, 7.95, 10.20, 12.20]

    def val(arch, k, field):
        d = sweep[arch]
        for cand in [f"{k}", f"{k:.1f}", f"{k:.2f}".rstrip("0").rstrip(".")]:
            if cand in d:
                return d[cand][field]
        return d[list(d)[0]][field]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for arch, colour, marker, ls, label in [
            ("resnet50", "#4477AA", "o", "-",  "ResNet-50"),
            ("cnn",      "#EE6677", "s", "--", "CNN")]:
        if arch not in sweep:
            continue
        means = [val(arch, k, "mean") for k in modes]
        stds  = [val(arch, k, "std")  for k in modes]
        ax.errorbar(modes, means, yerr=stds, marker=marker, capsize=3,
                    color=colour, linestyle=ls, label=label)
    ax.set_xlabel("AMR-NB bitrate (kbit/s)")
    ax.set_ylabel("Macro-F1 (clean-trained)")
    ax.set_title("Clean-trained performance across AMR-NB modes")
    ax.legend(fontsize=8)
    _save(fig, "fig_bitrate_curve")


# ---------------------------------------------------------------- Fig 2
def fig_regimes():
    regimes = load("regimes.json")["resnet50"]
    order  = ["clean_to_clean", "clean_to_4.75", "matched_4.75", "aug_4.75",
              "clean_to_12.2", "matched_12.2", "aug_12.2"]
    labels = ["clean->clean", "clean->4.75k", "matched 4.75k", "aug 4.75k",
              "clean->12.2k", "matched 12.2k", "aug 12.2k"]
    colour_of = {"clean_to_clean": "#BBBBBB",
                 "clean_to_4.75": "#EE6677", "matched_4.75": "#4477AA", "aug_4.75": "#228833",
                 "clean_to_12.2": "#EE6677", "matched_12.2": "#4477AA", "aug_12.2": "#228833"}
    means, lo_err, hi_err, colours = [], [], [], []
    rng = np.random.default_rng(42)
    for k in order:
        f1 = np.array(regimes[k]["f1all"]); m = f1.mean()
        means.append(m); colours.append(colour_of[k])
        draws = [np.mean(rng.choice(f1, len(f1), True)) for _ in range(5000)]
        lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
        lo_err.append(m - lo); hi_err.append(hi - m)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x, means, color=colours, alpha=0.85)
    ax.errorbar(x, means, yerr=[lo_err, hi_err], fmt="none",
                color="black", capsize=4, lw=1.3)
    ax.axhline(means[0], ls="--", color="grey", lw=1,
               label=f"clean baseline ({means[0]:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0.3, 0.85)
    ax.set_title("Training regime comparison with 95% CI (ResNet-50)")
    ax.legend(fontsize=8)
    _save(fig, "fig_regimes")


# ---------------------------------------------------------------- Fig 3
def fig_bandlimit_comparison():
    comp = load("bandlimit_comparison.json")
    stats = load("bandlimit_statistics.json")
    A = np.array(comp["A_clean"]["f1all"])
    B = np.array(comp["B_bandlimit"]["f1all"])
    C = np.array(comp["C_amr475"]["f1all"])
    d = stats["decomposition"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ["A: Clean", "B: Band-limited\n(bandwidth only)",
              "C: AMR-NB 4.75k\n(bandwidth + codec)"]
    pos = np.array([0, 1, 2]); cols = ["#228833", "#4477AA", "#EE6677"]
    means = [A.mean(), B.mean(), C.mean()]; stds = [A.std(), B.std(), C.std()]
    ax.bar(pos, means, color=cols, alpha=0.85, width=0.55)
    ax.errorbar(pos, means, yerr=stds, fmt="none", color="black", capsize=5)
    rng = np.random.default_rng(42)
    for p, vals in zip(pos, [A, B, C]):
        ax.scatter(p + rng.uniform(-0.1, 0.1, len(vals)), vals,
                   s=16, color="black", alpha=0.4, zorder=5)
    ax.annotate("", xy=(1, B.mean()), xytext=(0, A.mean()),
                arrowprops=dict(arrowstyle="->", color="#4477AA", lw=1.5))
    ax.annotate("", xy=(2, C.mean()), xytext=(1, B.mean()),
                arrowprops=dict(arrowstyle="->", color="#EE6677", lw=1.5))
    ax.text(0.5, (A.mean() + B.mean()) / 2,
            f"-{d['bandwidth_drop']:.3f}\n({d['bandwidth_pct']:.0f}%)",
            ha="center", va="center", fontsize=8, color="#4477AA")
    ax.text(1.5, (B.mean() + C.mean()) / 2,
            f"-{d['codec_drop']:.3f}\n({d['codec_pct']:.0f}%)",
            ha="center", va="center", fontsize=8, color="#EE6677")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0.2, 0.95)
    ax.set_title("Bandwidth restriction versus codec-specific distortion")
    _save(fig, "fig_bandlimit_comparison")


# ---------------------------------------------------------------- Fig 4
def fig_multicodec():
    sc = load("second_codec_results.json")
    codecs = ["AMR-NB\n4.75k", "Opus\n6k", "AMR-WB\n6.6k"]
    # AMR-NB reference values come from the main regimes.json
    reg = load("regimes.json")["resnet50"]
    amrnb_clean = np.mean(reg["clean_to_4.75"]["f1all"])
    amrnb_aug   = np.mean(reg["aug_4.75"]["f1all"])
    clean_drop = [amrnb_clean, sc["clean_to_opus_6k"]["mean"], sc["clean_to_amrwb_6k6"]["mean"]]
    aug_val    = [amrnb_aug,   sc["aug_opus_6k"]["mean"],      sc["aug_amrwb_6k6"]["mean"]]
    x = np.arange(len(codecs)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, clean_drop, w, label="Clean-trained (degraded test)",
           color="#EE6677", alpha=0.85)
    ax.bar(x + w/2, aug_val, w, label="Codec-aware augmentation",
           color="#228833", alpha=0.85)
    ax.axhline(CLEAN_BASELINE, ls="--", color="grey", lw=1,
               label=f"Clean baseline ({CLEAN_BASELINE})")
    ax.set_xticks(x); ax.set_xticklabels(codecs)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 0.9)
    ax.set_title("Degradation and recovery across three codecs at aggressive bitrates")
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, "fig_multicodec")


# ---------------------------------------------------------------- Fig 5
def fig_bandlimit_acoustic():
    av = load("acoustic_verification.json")
    bl  = [av["per_band_bandlimit"][f"{b[0]}-{b[1]}"] for b in BANDS]
    amr = [av["per_band_amr"][f"{b[0]}-{b[1]}"] for b in BANDS]
    band_labels = [f"{b[0]}-{b[1]}" for b in BANDS]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(BANDS))
    ax.plot(x, bl,  marker="s", color="#4477AA", lw=1.5, label="Band-limited")
    ax.plot(x, amr, marker="o", color="#EE6677", lw=1.5, label="AMR-NB 4.75k")
    ax.fill_between(x, bl, amr, alpha=0.15, color="#EE6677", label="Codec-specific")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(3.5, ls=":", color="grey", lw=1)
    ax.text(3.6, min(amr) - 4, "Nyquist\nlimit", fontsize=7, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(band_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Energy change (dB)")
    ax.set_title("Per-band energy change relative to clean audio")
    ax.legend(fontsize=7.5)
    _save(fig, "fig_bandlimit_acoustic")


# ---------------------------------------------------------------- Fig 6
def fig_freqdep_esc50():
    fd = load("experiment3_frequency_dependence.json")
    drop = fd["per_class_drop"]; hf = fd["per_class_hf"]
    classes = list(drop.keys())
    xs = [hf[c] for c in classes]; ys = [drop[c] for c in classes]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(xs, ys, s=28, color="#4477AA", alpha=0.7, zorder=3)
    m, b = np.polyfit(xs, ys, 1); xl = np.linspace(min(xs), max(xs), 50)
    ax.plot(xl, m * xl + b, "--", color="#EE6677", lw=1.5)
    order = np.argsort(ys)
    for i in list(order[:2]) + list(order[-3:]):
        ax.annotate(classes[i].replace("_", " "), (xs[i], ys[i]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("High-frequency energy fraction $H_c$ (clean audio)")
    ax.set_ylabel("F1 drop (clean $\\rightarrow$ AMR-NB 4.75k)")
    rho = fd["spearman_rho"]; p = fd["spearman_p"]
    ax.set_title(f"Per-class vulnerability vs high-frequency reliance\n"
                 f"ESC-50, $n=50$ ($\\rho={rho:.2f}$, $p={p:.3f}$)")
    _save(fig, "fig_freqdep_esc50")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {name}.pdf / .png")


if __name__ == "__main__":
    fig_bitrate_curve()
    fig_regimes()
    fig_bandlimit_comparison()
    fig_multicodec()
    fig_bandlimit_acoustic()
    fig_freqdep_esc50()
    print("\nAll six manuscript figures regenerated from the JSON results.")
    print("Note: the graphical abstract (fig_graphical_abstract) is a schematic")
    print("diagram, not a data plot; it is provided directly in figures/.")
