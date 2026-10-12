"""Volcano plot, one panel per program."""
import logging

import numpy as np

logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence font-subsetting chatter in PDF export


def volcano(res, path_stem, fdr_cut=0.05, n_label=12, program_sizes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    progs = list(dict.fromkeys(res["program"]))
    ncol = min(3, len(progs))
    nrow = int(np.ceil(len(progs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), squeeze=False)
    for ax, p in zip(axes.ravel(), progs):
        d = res[(res["program"] == p) & res["p_perm"].notna()]
        x, y = d["effect_sd"].to_numpy(), -np.log10(d["p_perm"].to_numpy())
        hit = (d["fdr"] < fdr_cut).to_numpy()
        ax.scatter(x[~hit], y[~hit], s=4, c="#bdbdbd", lw=0, rasterized=True)
        ax.scatter(x[hit & (x > 0)], y[hit & (x > 0)], s=9, c="#c0392b", lw=0)
        ax.scatter(x[hit & (x < 0)], y[hit & (x < 0)], s=9, c="#2471a3", lw=0)
        for _, r in d[hit].nsmallest(n_label, "p_perm").iterrows():
            ax.annotate(r["unit"], (r["effect_sd"], -np.log10(r["p_perm"])), fontsize=5.5,
                        xytext=(2, 2), textcoords="offset points")
        size = f", {program_sizes[p]} genes" if program_sizes and p in program_sizes else ""
        ax.set_title(f"{p}{size}: {int(hit.sum())} at FDR < {fdr_cut:g}", fontsize=7)
        ax.axvline(0, c="#d0d0d0", lw=0.5, zorder=0)
        ax.set_xlabel("effect (residual SD)")
        ax.set_ylabel("−log10 permutation P")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(progs):]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300)
    plt.close(fig)
