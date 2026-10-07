"""Build report/glove_gap.pdf (one A4 page). Numbers are read from report/ and eval/out/ CSVs
where they are single values; the prose numbers below were checked against those files.
Usage: build_pdf.py"""
import json, pathlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP = ROOT / "report"; OUT = ROOT / "eval/out"
INK, MUTED, RULE, BOX = colors.HexColor("#1f1f1e"), colors.HexColor("#5f5e58"), colors.HexColor("#d9d8d2"), colors.HexColor("#f3f2ee")

base = dict(fontName="Helvetica", fontSize=7.4, leading=9.0, textColor=INK, alignment=TA_LEFT)
S = {
    "title": ParagraphStyle("t", **{**base, "fontName": "Helvetica-Bold", "fontSize": 13.5, "leading": 16}),
    "q": ParagraphStyle("q", **{**base, "fontName": "Helvetica-Oblique", "fontSize": 8.4, "leading": 10.4, "textColor": MUTED}),
    "h": ParagraphStyle("h", **{**base, "fontName": "Helvetica-Bold", "fontSize": 8.2, "leading": 10, "spaceBefore": 3, "spaceAfter": 1}),
    "b": ParagraphStyle("b", **base),
    "li": ParagraphStyle("li", **{**base, "leftIndent": 9, "bulletIndent": 0, "spaceAfter": 1.2,
                                  "bulletFontName": "Helvetica-Bold", "bulletFontSize": 7.4}),
    "cell": ParagraphStyle("c", **{**base, "fontSize": 6.9, "leading": 8.2}),
    "cellb": ParagraphStyle("cb", **{**base, "fontSize": 6.9, "leading": 8.2, "fontName": "Helvetica-Bold"}),
    "foot": ParagraphStyle("f", **{**base, "fontSize": 6.2, "leading": 7.4, "textColor": MUTED}),
}
P = lambda t, s="b": Paragraph(t, S[s])

prm = json.load(open(OUT / "p3_params.json"))
tau = prm["gate_tau_px"]
gate = pd.read_csv(OUT / "p3_gate.csv")
g80 = gate[(gate.model == "wilor") & (gate.X == 80)].groupby("cond")["pass"].mean().mul(100).round().astype(int)

story = [
    P("Glove gap: what work gloves do to the 3D hand-pose front end of human-video → robot learning", "title"),
    P("How much does 3D hand-pose estimation degrade when the operator wears work gloves, and which "
      "capture-spec choices recover it?", "q"),
    Spacer(1, 3),
    P("Setup", "h"),
    P("<b>Data:</b> DexYCB subject-01, 19 held-out S0-val grasp sequences (+1 in appendix), GT 3D joints; a 2-view rig "
      "(1.03 m baseline, 70°) plus an 8-camera bound. <b>Gloves</b> (synthetic, on refined hand masks): G0 bare, "
      "G1 thin bright nitrile, G2 grey knit, G3a dark insulated, G3b = G3a + outline growth k = 1–4 px, G4 = bright insulated (k = 0, 4). "
      "<b>Models:</b> MediaPipe Hands, WiLoR, HaMeR (on WiLoR boxes). <b>Real-glove check:</b> Roboflow photos, 83 images. "
      "CIs: bootstrap over sequences (95%); Wilson for recall."),
    P("Findings", "h"),
    Paragraph(f"<b>Real gloves break detection; synthetic gloves underestimate it.</b> On real worn-glove photos WiLoR's "
              "detector finds 52–65% of gloves vs 100% of bare hands (−35 to −48 pp, depending on how strictly boxes must match: "
              "glove boxes include the cuff); MediaPipe finds 7–13%. On synthetic gloves WiLoR loses only 0–4 pp "
              "(MediaPipe 87–90 pp), so synthetic detection numbers for WiLoR are optimistic. ", S["li"], bulletText="a"),
    Paragraph("<b>Gloves bias single-camera depth.</b> Monocular WiLoR places the hand 45 mm too close with a dark "
              "insulated glove and about 20 mm more per px of outline growth (−124 mm at k = 4). Bright colour does not "
              "remove it for WiLoR (G4: −41 mm). Take hand position from depth or ≥2 views, never from one RGB camera.",
              S["li"], bulletText="b"),
    Paragraph("<b>Multi-view fixes position, not articulation.</b> 2-view triangulation removes 72–88% of the glove-induced "
              "position gap (8 cameras: 83–93%) but only 3–49% of the articulation (wrist-relative) gap, for both WiLoR "
              "and HaMeR. Articulation is set by the glove: thin nitrile instead of the work glove removes 73–92% of it, "
              "while bright colour on an insulated glove does not help (G4 vs G3a: −1.7 mm [−3.6, +0.1]).",
              S["li"], bulletText="c"),
    Paragraph(f"<b>A reprojection gate is a batch-level glove alarm, not an episode ranker.</b> With frames valid at 2-view "
              f"reprojection &lt; {tau:.1f} px and episodes passing at ≥80% valid frames: G0 {g80['G0']}%, G1 {g80['G1']}%, "
              f"G2 {g80['G2']}%, G3a {g80['G3a']}%, G4 {g80['G4']}% pass. Within a glove type, the valid fraction does not "
              "predict error (|ρ| ≤ 0.36, p &gt; 0.1).", S["li"], bulletText="d"),
    Spacer(1, 2),
    Image(str(REP / "pdf_fig_a.png"), width=15.0 * cm, height=15.0 * cm * 2.5 / 6.6),
]

# Figures B and C side by side with the capture-spec table
spec = [
    [P("Capture spec", "cellb"), P("Number", "cellb"), P("Evidence (this PoC)", "cellb")],
    [P("Pose model", "cell"), P("ViT/MANO class (WiLoR, HaMeR); not MediaPipe", "cell"),
     P("MediaPipe detection 93% → ≤6% synthetic, 7% real", "cell")],
    [P("Glove", "cell"), P("Thin, form-fitting nitrile; colour secondary", "cell"),
     P("G1: PA-MPJPE +0.6 mm vs bare; insulated +2.4 mm and wrist-rel ×2.9", "cell")],
    [P("Thickness", "cell"), P("Minimise: each px of apparent growth ≈ +0.4 mm PA-MPJPE, −20 mm mono depth", "cell"),
     P("G3b sweep k = 0–4", "cell")],
    [P("Views", "cell"), P("≥2 calibrated views, ~1 m baseline, ~70° (or RGB-D for position)", "cell"),
     P("Dark glove, position error 14.6 mm (2-view) vs 48.8 mm (mono)", "cell")],
    [P("Gate", "cell"), P(f"Per batch: τ = {tau:.1f} px, X = 80%; investigate batches below ~75% pass", "cell"),
     P(f"Pass {g80['G0']}% bare → {g80['G2']}% knit → {g80['G3a']}% insulated", "cell")],
]
t = Table(spec, colWidths=[1.7 * cm, 4.7 * cm, 4.5 * cm])
t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
                       ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                       ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                       ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
limits = Table([[P("<b>Limits.</b> Synthetic gloves are 2D image edits: no 3D thickening, finger gaps close, mask-edge "
                   "residuals. Possible training leakage: HaMeR (and likely WiLoR) train on DexYCB S0-train; held-out "
                   "S0-val may have been used for model selection, so relative degradation is the claim, not absolute "
                   "accuracy. One subject, YCB lab objects, room temperature, not a warehouse. The real-glove check is "
                   "detection-only, single-frame, n = 83 images (31 worn-glove boxes). Grasp-onset F1 is secondary: the "
                   "object-motion label is noisy (GT-aperture ceiling 0.79).", "cell")]],
               colWidths=[7.0 * cm])
limits.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX), ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
figw = 8.2 * cm
figs = Table([[Image(str(REP / "pdf_fig_b.png"), width=figw, height=figw * 2.5 / 3.3),
               Image(str(REP / "pdf_fig_c.png"), width=figw, height=figw * 2.5 / 3.3)]],
             colWidths=[9.1 * cm, 9.1 * cm])
figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
bottom = Table([[t, limits]], colWidths=[11.0 * cm, 7.2 * cm])
bottom.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("LEFTPADDING", (1, 0), (1, 0), 6)]))
story += [figs, Spacer(1, 3), bottom, Spacer(1, 4),
          P("Code: <font face='Courier'>make all</font> reproduces every number and figure from cached predictions. "
            "Licences: DexYCB CC BY-NC 4.0; WiLoR CC-BY-NC-ND (+ MANO, Ultralytics); HaMeR MIT (+ MANO); "
            "Roboflow 'Gloves and bare hands detection' CC BY 4.0. Non-commercial research PoC.", "foot")]

doc = SimpleDocTemplate(str(REP / "glove_gap.pdf"), pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                        topMargin=1.2 * cm, bottomMargin=1.0 * cm, title="Glove gap", author="Adithya Balaji")
doc.build(story)
print("pages:", len(__import__("pypdf").PdfReader(str(REP / "glove_gap.pdf")).pages))
