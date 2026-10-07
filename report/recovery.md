# P3: interventions (gap reduction)

**Gap reduction** = 1 − (int_Gx − int_G0) / (mono_Gx − mono_G0), computed per model.
- It is the share of the glove-induced degradation of the monocular pipeline that the
  intervention removes. 100% = no glove degradation left; 0% = degrades as much as mono.
- Mean over the 19 main sequences; bootstrap 95% CI over sequences.
- 8cam: WiLoR only, every 2nd frame.

| Model | Intervention (vs mono) | Metric | G2 | G3a | G3b_k4 |
|---|---|---|---|---|---|
| wilor | 2view | position (abs MPJPE) | 72% [61, 80] | 81% [74, 85] | 88% [86, 90] |
| wilor | 2view | articulation (wrist-rel MPJPE) | 36% [21, 50] | 49% [36, 59] | 3% [-9, 14] |
| wilor | 2view | jitter | 66% [58, 73] | 65% [58, 71] | 51% [44, 58] |
| wilor | 2view_oe | position (abs MPJPE) | 75% [64, 82] | 82% [76, 87] | 89% [87, 91] |
| wilor | 2view_oe | articulation (wrist-rel MPJPE) | 37% [22, 51] | 49% [36, 60] | 4% [-8, 15] |
| wilor | 2view_oe | jitter | 77% [72, 82] | 76% [71, 80] | 66% [61, 70] |
| wilor | hybrid | position (abs MPJPE) | 45% [23, 59] | 50% [32, 62] | 75% [72, 78] |
| wilor | hybrid | articulation (wrist-rel MPJPE) | 21% [15, 27] | 19% [14, 24] | 21% [15, 27] |
| wilor | hybrid | jitter | 60% [52, 67] | 55% [44, 64] | 52% [43, 59] |
| wilor | 8cam | position (abs MPJPE) | 83% [76, 88] | 88% [84, 91] | 93% [91, 94] |
| hamer | 2view | position (abs MPJPE) | 72% [65, 78] | 79% [74, 83] | 86% [84, 88] |
| hamer | 2view | articulation (wrist-rel MPJPE) | 47% [34, 59] | 49% [37, 58] | 6% [-6, 18] |
| hamer | 2view | jitter | 71% [65, 76] | 61% [55, 66] | 54% [48, 60] |
| hamer | 2view_oe | position (abs MPJPE) | 75% [68, 80] | 81% [76, 84] | 87% [84, 89] |
| hamer | 2view_oe | articulation (wrist-rel MPJPE) | 47% [34, 59] | 49% [38, 58] | 7% [-6, 18] |
| hamer | 2view_oe | jitter | 80% [76, 83] | 72% [68, 76] | 67% [63, 71] |
| hamer | hybrid | position (abs MPJPE) | 59% [46, 69] | 57% [44, 67] | 74% [70, 77] |
| hamer | hybrid | articulation (wrist-rel MPJPE) | 18% [14, 23] | 18% [14, 22] | 21% [17, 25] |
| hamer | hybrid | jitter | 65% [60, 69] | 56% [51, 61] | 53% [48, 58] |

## Glove spec: thin nitrile (G1) instead of the work glove

Share of the 2-view G0→Gx degradation that disappears when the operator wears G1 instead:

| Metric (WiLoR, 2-view) | G2 → G1 | G3a → G1 | G3b_k4 → G1 |
|---|---|---|---|
| position (abs MPJPE) | 67% [56, 77] | 74% [66, 80] | 87% [83, 91] |
| articulation (wrist-rel MPJPE) | 73% [62, 82] | 81% [74, 88] | 92% [88, 95] |
| jitter | 66% [52, 79] | 80% [71, 87] | 88% [82, 92] |

**Colour alone does not explain this.** G4 has G1's bright blue on an insulated glove (G3a's
texture, blur and shading). Paired deltas, 95% CI:

| Comparison | Detection | Articulation | Position |
|---|---|---|---|
| G4 vs G3a (colour, at fixed insulation) | +3 pp | no reliable change | +0.7 mm (WiLoR), +3.3 mm (HaMeR) |
| G4 vs G1 (thickness/detail, at fixed colour) | – | +12.8 mm wrist-rel MPJPE, +7.8 mm aperture (WiLoR) | – |

The thin glove's advantage comes from preserved finger shading and detail, not from colour.

- **One-Euro:** mincutoff 4.0, β 0.2, tuned on G0 only.
- **Hybrid:** triangulated wrist + monocular articulation averaged over the two views.
