"""P2 figures -> report/fig{1..4}_*.png. Usage: plots.py (after eval/metrics.py, eval/summary.py,
eval/roboflow_bench.py). Palette: validated categorical slots 1-3 (dataviz reference palette);
aqua is < 3:1 on the surface, so every series also gets a marker shape and a direct label."""
import pickle, sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from common import CFG, sequences, conditions
import metrics as M

REP = ROOT / "report"
COL = {"mediapipe": "#2a78d6", "wilor": "#eb6834", "hamer": "#1baf7a"}
MRK = {"mediapipe": "o", "wilor": "s", "hamer": "^"}
NAME = {"mediapipe": "MediaPipe", "wilor": "WiLoR", "hamer": "HaMeR"}
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.dpi": 200, "lines.linewidth": 2})
res = pd.read_csv(REP / "results.csv")
CONDS = conditions()
MAIN_CONDS = ["G0", "G1", "G2", "G3a", "G3b_k4"]


def series(model, metric, conds, scope="per_view_detected"):
    r = res[(res.scope == scope) & (res.model == model) & (res.metric == metric)].set_index("cond").reindex(conds)
    return r["mean"].values, r.ci_lo.values, r.ci_hi.values


def errbars(ax, x, m, lo, hi, model, scale=1.0, label_last=False):
    ax.errorbar(x, m * scale, yerr=[(m - lo) * scale, (hi - m) * scale], color=COL[model], marker=MRK[model],
                ms=6, capsize=3, lw=2, label=NAME[model])
    if label_last and np.isfinite(m[-1]):
        ax.annotate(NAME[model], (x[-1], m[-1] * scale), xytext=(6, 0), textcoords="offset points",
                    color=INK, va="center", fontsize=8)


def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    x = np.arange(len(MAIN_CONDS))
    for i, model in enumerate(["mediapipe", "wilor"]):   # HaMeR shares WiLoR's detector
        errbars(axs[0], x + (i - 0.5) * 0.08, *series(model, "det_rate", MAIN_CONDS), model, 100)
    axs[0].set_ylabel("Detection rate (%)"); axs[0].set_ylim(0, 105)
    axs[0].set_title("Detection (HaMeR = WiLoR boxes)", fontsize=9, color=INK, loc="left")
    for i, model in enumerate(["mediapipe", "wilor", "hamer"]):
        errbars(axs[1], x + (i - 1) * 0.08, *series(model, "pck02", MAIN_CONDS), model, 100)
    axs[1].set_ylabel("2D PCK@0.2 on detected frames (%)"); axs[1].set_ylim(0, 105)
    axs[1].set_title("2D accuracy", fontsize=9, color=INK, loc="left")
    for i, model in enumerate(["wilor", "hamer"]):
        errbars(axs[2], x + (i - 0.5) * 0.08, *series(model, "pa_mpjpe_mm", MAIN_CONDS), model)
    axs[2].set_ylabel("PA-MPJPE on detected frames (mm)"); axs[2].set_ylim(bottom=0)
    axs[2].set_title("3D accuracy (MediaPipe 3D not metric)", fontsize=9, color=INK, loc="left")
    for ax in axs:
        ax.set_xticks(x, MAIN_CONDS); ax.set_xlim(-0.4, len(x) - 0.2)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Fig 1. Degradation by glove condition (mean over sequences, 95% bootstrap CI)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(REP / "fig1_degradation.png"); plt.close(fig)


def fig2():
    ks = ["G3a"] + [f"G3b_k{k}" for k in CFG["gloves"]["G3b"]["grow_px"]]
    x = np.array([0] + CFG["gloves"]["G3b"]["grow_px"])
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    for model in ["mediapipe", "wilor"]:
        errbars(axs[0], x, *series(model, "det_rate", ks), model, 100)
    for model in ["wilor", "hamer"]:
        errbars(axs[1], x, *series(model, "pa_mpjpe_mm", ks), model)
    axs[0].set_ylabel("Detection rate (%)"); axs[0].set_ylim(0, 105)
    axs[1].set_ylabel("PA-MPJPE (mm)"); axs[1].set_ylim(bottom=0)
    for ax in axs:
        ax.set_xlabel("Outline growth k (px), k=0 = G3a"); ax.set_xticks(x); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 2. G3b sensitivity: dark insulated glove + 2D outline growth", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(REP / "fig2_g3b_sensitivity.png"); plt.close(fig)


def fig3(examples):
    traces = pickle.load(open(ROOT / "eval/out/traces.pkl", "rb"))
    ev = pd.read_csv(ROOT / "eval/out/events.csv")
    gt_ev = pd.read_csv(ROOT / "data/events.csv").set_index("seq").frame
    fig, axs = plt.subplots(1, len(examples), figsize=(9, 3), sharey=True)
    for ax, seq in zip(axs, examples):
        ap0, gap, frames = traces[("wilor", "G0", seq)]
        t = np.array(frames) / CFG["dexycb"]["fps_src"]
        ax.plot(t, [gap[f] for f in frames], color=MUTED, lw=1.5, ls="--", label="GT aperture")
        for cond, model, ls in [("G0", "wilor", "-"), ("G3a", "wilor", ":"), ("G0", "mediapipe", "-")]:
            ap, _, fr = traces[(model, cond, seq)]
            ff = sorted(ap)
            ax.plot(np.array(ff) / 30, [ap[f] for f in ff], color=COL[model], ls=ls, lw=1.8, label=f"{NAME[model]} {cond}")
            r = ev[(ev.model == model) & (ev.cond == "G0")].iloc[0]
            p = M.onset(ap, fr, r.rho)
            if p is not None:
                ax.axvline((p + r.lag_frames) / 30, color=COL[model], ls=ls, lw=1)
        g = gt_ev[seq] / 30
        ax.axvspan(g - 0.1, g + 0.1, color=GRID, alpha=0.9, lw=0)
        ax.axvline(g, color=INK, lw=1.2)
        ax.set_title(f"{seq} ({pd.read_csv(ROOT / 'data/meta.csv').drop_duplicates('seq').set_index('seq').task[seq]})", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("time (s)")
    axs[0].set_ylabel("thumb–index aperture, 2-view (mm)")
    axs[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(1.05, -0.22), ncol=4)
    fig.suptitle("Fig 3. Aperture traces vs logged grasp onset (black line, grey band = ±100 ms); "
                 "thin verticals = detected onsets after lag correction", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(REP / "fig3_aperture_events.png"); plt.close(fig)


def fig4():
    rf = pd.read_csv(ROOT / "eval/out/roboflow_recall.csv")
    rf = rf[(rf.subset == "main_worn") & (rf["size"] == "all")].set_index(["method", "class"])
    rows = []
    for model in ["mediapipe", "wilor"]:
        bare, glove = rf.loc[(model, "bare")], rf.loc[(model, "glove")]
        rows.append((model, "Real: Roboflow bare→worn glove (IoU≥0.5)", 100 * (bare.recall - glove.recall)))
        for cond in ["G1", "G2", "G3a"]:
            r = res[(res.scope == "per_view_detected") & (res.model == model) & (res.metric == "det_rate") & (res.cond == cond)].iloc[0]
            rows.append((model, f"Synthetic: DexYCB G0→{cond}", -100 * r.delta_vs_G0))
    labels = list(dict.fromkeys(r[1] for r in rows))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, model in enumerate(["mediapipe", "wilor"]):
        v = [next(r[2] for r in rows if r[0] == model and r[1] == lab) for lab in labels]
        ax.barh(y + (i - 0.5) * 0.38, v, height=0.36, color=COL[model], label=NAME[model], edgecolor=SURF, linewidth=2)
        for yy, vv in zip(y + (i - 0.5) * 0.38, v):
            ax.annotate(f"{abs(vv):.0f}" if abs(vv) < 0.5 else f"{vv:.0f}", (max(vv, 0), yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel("Detection drop vs bare (pp). Real WiLoR drop is 35–48 pp across match rules")
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Fig 4. Real vs synthetic glove detection drop", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(REP / "fig4_real_vs_synth.png"); plt.close(fig)


def fig5():
    """Episode gate pass rate vs threshold X, per condition (WiLoR, 2-view). Ordinal one-hue ramp
    G0 -> G3b_k4 (light->dark would recede; use blue 250..700 steps, all >= 3:1 except lightest)."""
    g = pd.read_csv(ROOT / "eval/out/p3_gate.csv"); g = g[g.model == "wilor"]
    conds = ["G0", "G1", "G2", "G3a", "G3b_k4"]
    ramp = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    mk = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for c, col, m in zip(conds, ramp, mk):
        d = g[g.cond == c].groupby("X")["pass"].mean() * 100
        ax.plot(d.index, d.values, color=col, marker=m, ms=6, label=c)
    ax.set_xlabel(f"Gate: min % of frames valid (2-view reproj < {g.tau_px.iloc[0]:.1f} px)")
    ax.set_ylabel("Episodes passing (%)"); ax.set_ylim(-3, 105); ax.set_xlim(48, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Fig 5. Trainability gate pass rate by glove (WiLoR, 19 episodes)", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(REP / "fig5_gate.png"); plt.close(fig)



# ------------------------------------------------------------------ compact figures for the 1-page PDF
PDF_CONDS = ["G0", "G1", "G2", "G3a", "G3b_k4", "G4", "G4b_k4"]
PDF_LABELS = ["G0\nbare", "G1\nthin", "G2\nknit", "G3a\ndark", "G3b\nk=4", "G4\nbright", "G4b\nk=4"]


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    bs = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return v.mean(), *np.percentile(bs, [2.5, 97.5])


def pdf_fig_a():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(len(PDF_CONDS))
    for i, model in enumerate(["mediapipe", "wilor"]):
        m, lo, hi = series(model, "det_rate", PDF_CONDS)
        errbars(axs[0], x + (i - 0.5) * 0.12, m, lo, hi, model, 100)
    axs[0].set_ylabel("Detection (%)"); axs[0].set_ylim(-3, 105)
    for i, model in enumerate(["wilor", "hamer"]):
        m, lo, hi = series(model, "mpjpe_rel_mm", PDF_CONDS)
        errbars(axs[1], x + (i - 0.5) * 0.12, m, lo, hi, model)
    axs[1].set_ylabel("Wrist-relative MPJPE (mm)"); axs[1].set_ylim(bottom=0)
    for ax in axs:
        ax.set_xticks(x, PDF_LABELS, fontsize=6.5); ax.legend(frameon=False, fontsize=7, loc="best")
        ax.axvspan(4.5, 6.5, color=GRID, alpha=0.5, lw=0)
    axs[0].set_title("A. Detection (HaMeR uses WiLoR boxes)", fontsize=8, color=INK, loc="left")
    axs[1].set_title("A. Articulation error, detected frames", fontsize=8, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(REP / "pdf_fig_a.png", dpi=250); plt.close(fig)


def pdf_fig_b():
    pp = pd.read_csv(ROOT / "eval/out/p3_pipelines.csv")
    pp = pp[pp.pipeline == "mono"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for model, ls in [("wilor", "-"), ("hamer", "--")]:
        for fam, conds, ks, col, mk in [("dark (G3)", ["G3a", "G3b_k1", "G3b_k2", "G3b_k3", "G3b_k4"], [0, 1, 2, 3, 4], "#184f95", MRK[model]),
                                       ("bright (G4)", ["G4", "G4b_k4"], [0, 4], "#6da7ec", MRK[model])]:
            st = [boot_ci(pp[(pp.model == model) & (pp.cond == c)].root_z_bias_mm) for c in conds]
            if any(np.isnan(t[0]) for t in st):
                continue
            m = np.array([t[0] for t in st]); lo = np.array([t[1] for t in st]); hi = np.array([t[2] for t in st])
            ax.errorbar(np.array(ks) + (0.06 if model == "hamer" else -0.06), m, yerr=[m - lo, hi - m], color=col, ls=ls,
                        marker=mk, ms=5, capsize=2, lw=1.6, label=f"{NAME[model]} {fam}")
        g0 = boot_ci(pp[(pp.model == model) & (pp.cond == "G0")].root_z_bias_mm)[0]
        ax.axhline(g0, color=MUTED, lw=0.8, ls=ls)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("Glove outline growth k (px)"); ax.set_ylabel("Mono root-depth bias (mm)\n(− = hand placed too close)")
    ax.set_xticks([0, 1, 2, 3, 4]); ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.28))
    ax.set_title("B. Single-camera depth bias (grey = bare)", fontsize=8, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(REP / "pdf_fig_b.png", dpi=250); plt.close(fig)


def pdf_fig_c():
    g = pd.read_csv(ROOT / "eval/out/p3_gate.csv"); g = g[g.model == "wilor"]
    conds = ["G0", "G1", "G2", "G3a", "G3b_k4", "G4"]
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b", "#eb6834"]
    mk = ["o", "s", "^", "D", "v", "P"]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for c, col, m in zip(conds, ramp, mk):
        d = g[g.cond == c].groupby("X")["pass"].mean() * 100
        if d.empty:
            continue
        ax.plot(d.index, d.values, color=col, marker=m, ms=4, lw=1.6, label=c)
    ax.set_xlabel(f"Min % frames valid (reproj < {g.tau_px.iloc[0]:.1f} px)"); ax.set_ylabel("Episodes passing (%)")
    ax.set_ylim(-3, 105); ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("C. Episode gate (WiLoR, 2-view)", fontsize=8, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(REP / "pdf_fig_c.png", dpi=250); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig4()
    if (ROOT / "eval/out/p3_gate.csv").exists():
        fig5(); pdf_fig_a(); pdf_fig_b(); pdf_fig_c()
    ex = [a for a in sys.argv[1:] if not a.startswith("--")] or sequences()[:2]
    fig3(ex)
    print("figures written")
