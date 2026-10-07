"""report/recovery.md from report/recovery.csv + eval/out (report-facing: gap reduction only).
Usage: recovery_md.py"""
import numpy as np, pandas as pd
from common import ROOT, sequences

REP = ROOT / "report"; OUT = ROOT / "eval/out"
r = pd.read_csv(REP / "recovery.csv")
pp = pd.read_csv(OUT / "p3_pipelines.csv"); pp = pp[pp.seq.isin(sequences())]
rng = np.random.default_rng(0)
NAMES = {"mpjpe_abs_mm": "position (abs MPJPE)", "mpjpe_rel_mm": "articulation (wrist-rel MPJPE)", "jitter_mm_f2": "jitter"}


def gr(model, pipe, cond, met):
    g = lambda p, c: pp[(pp.model == model) & (pp.pipeline == p) & (pp.cond == c)].set_index("seq")[met]
    df = pd.concat([g("mono", cond), g("mono", "G0"), g(pipe, cond), g(pipe, "G0")], axis=1).dropna().values
    if len(df) < 3:
        return "–"
    f = lambda d: 100 * (1 - (d[:, 2].mean() - d[:, 3].mean()) / (d[:, 0].mean() - d[:, 1].mean()))
    bs = [f(df[rng.integers(0, len(df), len(df))]) for _ in range(2000)]
    return f"{f(df):.0f}% [{np.percentile(bs, 2.5):.0f}, {np.percentile(bs, 97.5):.0f}]"


lines = ["| Model | Intervention (vs mono) | Metric | G2 | G3a | G3b_k4 |", "|---|---|---|---|---|---|"]
for model in ["wilor", "hamer"]:
    for pipe in ["2view", "2view_oe", "hybrid", "8cam"]:
        for met in NAMES:
            if pipe == "8cam" and (model != "wilor" or met != "mpjpe_abs_mm"):
                continue
            lines.append(f"| {model} | {pipe} | {NAMES[met]} | " + " | ".join(gr(model, pipe, c, met) for c in ["G2", "G3a", "G3b_k4"]) + " |")
spec = r[r.intervention.str.startswith("glove spec")]
sl = ["| Metric (WiLoR, 2-view) | G2 → G1 | G3a → G1 | G3b_k4 → G1 |", "|---|---|---|---|"]
for met in NAMES:
    row = spec[spec.metric == met].set_index("cond")
    sl.append(f"| {NAMES[met]} | " + " | ".join(f"{row.loc[c].recovery_pct:.0f}% [{row.loc[c].ci_lo:.0f}, {row.loc[c].ci_hi:.0f}]" for c in ["G2", "G3a", "G3b_k4"]) + " |")
txt = f"""# P3: interventions (gap reduction)

**Gap reduction** = 1 − (int_Gx − int_G0) / (mono_Gx − mono_G0), computed per model.
- It is the share of the glove-induced degradation of the monocular pipeline that the
  intervention removes. 100% = no glove degradation left; 0% = degrades as much as mono.
- Mean over the 19 main sequences; bootstrap 95% CI over sequences.
- 8cam: WiLoR only, every 2nd frame.

{chr(10).join(lines)}

## Glove spec: thin nitrile (G1) instead of the work glove

Share of the 2-view G0→Gx degradation that disappears when the operator wears G1 instead:

{chr(10).join(sl)}

**Colour alone does not explain this.** G4 has G1's bright blue on an insulated glove (G3a's
texture, blur and shading). Paired deltas, 95% CI:

| Comparison | Detection | Articulation | Position |
|---|---|---|---|
| G4 vs G3a (colour, at fixed insulation) | +3 pp | no reliable change | +0.7 mm (WiLoR), +3.3 mm (HaMeR) |
| G4 vs G1 (thickness/detail, at fixed colour) | – | +12.8 mm wrist-rel MPJPE, +7.8 mm aperture (WiLoR) | – |

The thin glove's advantage comes from preserved finger shading and detail, not from colour.

- **One-Euro:** mincutoff 4.0, β 0.2, tuned on G0 only.
- **Hybrid:** triangulated wrist + monocular articulation averaged over the two views.
"""
(REP / "recovery.md").write_text(txt)
print(txt[:2500])
