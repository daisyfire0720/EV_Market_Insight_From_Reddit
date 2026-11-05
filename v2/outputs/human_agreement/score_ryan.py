#!/usr/bin/env python3
"""Score Ryan's annotation sheet against the Qwen2.5-14B labels and the small-model key.

Run from v2/:  python outputs/human_agreement/score_ryan.py
Inputs (read only):
  annotation/annotation_sheet_ryan.xlsx   Ryan's upload (commit f98c050)
  annotation/annotation_sheet.csv         blank sheet, used to check ids and text
  annotation/sample_manifest.csv          audience per id
  outputs/llm_labels.csv                  Qwen2.5-14B labels (funnel-v2 prompt)
  annotation/annotation_model_key.csv     small zero-shot model (deberta-v3-xsmall), stage + aspects only
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

V2 = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261030)
NBOOT = 2000
STAGES = ["awareness","consideration","evaluation","purchase","onboarding","long_term_ownership","unclear"]
FAMS = ["purchase_selection","charging_energy","battery_range","cost_incentives","dealer_access",
        "reliability_service","policy_market","transition_narratives"]
ASPS = ["charging_access","range","price_tco","incentives_policy","battery_reliability","dealer_friction","environment"]
STANCES = ["support","oppose","mixed","no_stance"]

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b); po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)

def alpha(a, b):
    """Krippendorff's alpha, nominal, two coders, no missing values."""
    a, b = np.asarray(a), np.asarray(b); n = 2 * len(a)
    vals = np.concatenate([a, b]); cats, cnt = np.unique(vals, return_counts=True)
    do = (a != b).mean()
    de = (n * n - (cnt ** 2).sum()) / (n * (n - 1))
    return np.nan if de == 0 else 1 - do / de

def stats(a, b):
    a, b = np.asarray(a), np.asarray(b); n = len(a)
    pt = dict(n=n, pct_agree=(a == b).mean(), kappa=kappa(a, b), alpha=alpha(a, b))
    boots = {k: [] for k in ("pct_agree", "kappa", "alpha")}
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        boots["pct_agree"].append((a[i] == b[i]).mean()); boots["kappa"].append(kappa(a[i], b[i])); boots["alpha"].append(alpha(a[i], b[i]))
    for k, v in boots.items():
        v = np.array(v, float); v = v[~np.isnan(v)]
        pt[k + "_lo"], pt[k + "_hi"] = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
    return pt

def split(s):
    return set() if pd.isna(s) or str(s).strip() == "" else set(str(s).split("|"))

# ---------- load
ryan = pd.read_excel(V2 / "annotation/annotation_sheet_ryan.xlsx", dtype=str)
blank = pd.read_csv(V2 / "annotation/annotation_sheet.csv", dtype=str)
man = pd.read_csv(V2 / "annotation/sample_manifest.csv", dtype=str).set_index("id")
q14 = pd.read_csv(V2 / "outputs/llm_labels.csv", dtype=str)
q14 = q14[q14.text_type == "submission"].set_index("id")
small = pd.read_csv(V2 / "annotation/annotation_model_key.csv", dtype=str).set_index("id")

# ---------- validation
code_cols = FAMS + ASPS + ["none", "stage", "stance", "annotator"]
filled = ryan[ryan[code_cols].notna().any(axis=1)].copy()
v = {}
v["rows_in_sheet"] = len(ryan); v["rows_in_blank_sheet"] = len(blank)
v["ids_same_set_as_blank"] = set(ryan.id) == set(blank.id)
v["duplicate_ids"] = int(ryan.id.duplicated().sum())
b = blank.set_index("id").loc[ryan.id]
for c in ["title", "text_excerpt", "subreddit", "calibration"]:
    v[f"{c}_mismatch_vs_blank_by_id"] = int((ryan.set_index("id")[c].fillna("").str.strip() != b[c].fillna("").str.strip()).sum())
v["rows_order_differs_from_blank"] = not (ryan.id.values == blank.id.values).all()
v["rows_with_any_code"] = len(filled)
v["filled_rows_calibration_yes"] = int((filled.calibration == "yes").sum())
v["filled_rows_main_750"] = int((filled.calibration != "yes").sum())
v["missing_cells_in_filled_rows"] = int(filled[code_cols].isna().sum().sum())
bad = []
for c in FAMS + ASPS + ["none"]:
    bad += [(i, c, x) for i, x in zip(filled.id, filled[c]) if x not in ("0", "1")]
bad += [(i, "stage", x) for i, x in zip(filled.id, filled.stage) if x not in STAGES]
bad += [(i, "stance", x) for i, x in zip(filled.id, filled.stance) if x not in STANCES]
v["out_of_codebook_values"] = len(bad)
asp_sum = filled[ASPS].astype(int).sum(axis=1)
v["none_flag_inconsistent"] = int(((asp_sum == 0).astype(int) != filled["none"].astype(int)).sum())
v["rows_all_families_0"] = int((filled[FAMS].astype(int).sum(axis=1) == 0).sum())
v["filled_ids_in_14b_labels"] = int(filled.id.isin(q14.index).sum())
v["filled_ids_14b_ok_true"] = int((q14.loc[filled.id[filled.id.isin(q14.index)], "ok"] == "True").sum())
v["filled_ids_in_small_key"] = int(filled.id.isin(small.index).sum())
v["daisy_label_column_present"] = bool((ryan.annotator.fillna("") == "Daisy").any())
pd.Series(v, name="value").rename_axis("check").to_csv(OUT / "validation.csv")

# ---------- align
f = filled.set_index("id")
f["audience"] = man.loc[f.index, "audience"]
q = q14.loc[f.index]
k = small.loc[f.index]
qfam = q.families.map(split); qasp = q.aspects.map(split)

rows, conf_rows = [], []
def add(field, pair, aud, a, bb):
    s = stats(a, bb); s.update(field=field, comparison=pair, audience=aud); rows.append(s)

for aud in ["all", "enthusiast", "shopper"]:
    m = np.ones(len(f), bool) if aud == "all" else (f.audience == aud).values
    R = f[m]; Q = q[m]; K = k[m]
    add("stage", "Ryan vs Qwen2.5-14B", aud, R.stage.values, Q.stage.values)
    coll = lambda s: s.replace({"onboarding": "ownership", "long_term_ownership": "ownership"})
    add("stage_ownership_collapsed", "Ryan vs Qwen2.5-14B", aud, coll(R.stage).values, coll(Q.stage).values)
    add("stage_ownership_collapsed", "Ryan vs small model", aud, coll(R.stage).values, K.stage.values)
    add("stance", "Ryan vs Qwen2.5-14B", aud, R.stance.values, Q.stance.values)
    # families: binary per family, pooled cells
    rf = np.concatenate([R[c].astype(int).values for c in FAMS])
    qf = np.concatenate([qfam[m].map(lambda s, c=c: int(c in s)).values for c in FAMS])
    add("family_pooled_8x_binary", "Ryan vs Qwen2.5-14B", aud, rf, qf)
    ra = np.concatenate([R[c].astype(int).values for c in ASPS])
    qa = np.concatenate([qasp[m].map(lambda s, c=c: int(c in s)).values for c in ASPS])
    ka = np.concatenate([K[c].astype(int).values for c in ASPS])
    add("aspect_pooled_7x_binary", "Ryan vs Qwen2.5-14B", aud, ra, qa)
    add("aspect_pooled_7x_binary", "Ryan vs small model", aud, ra, ka)
    if aud == "all":
        for c in FAMS:
            add(f"family:{c}", "Ryan vs Qwen2.5-14B", aud, R[c].astype(int).values, qfam.map(lambda s: int(c in s)).values)
        for c in ASPS:
            add(f"aspect:{c}", "Ryan vs Qwen2.5-14B", aud, R[c].astype(int).values, qasp.map(lambda s: int(c in s)).values)
            add(f"aspect:{c}", "Ryan vs small model", aud, R[c].astype(int).values, K[c].astype(int).values)

res = pd.DataFrame(rows)
cols = ["field","comparison","audience","n","pct_agree","pct_agree_lo","pct_agree_hi","kappa","kappa_lo","kappa_hi","alpha","alpha_lo","alpha_hi"]
res[cols].round(3).to_csv(OUT / "agreement_scores.csv", index=False)

# confusion matrices
pd.crosstab(f.stage.rename("ryan"), q.stage.rename("qwen14b")).reindex(index=STAGES, columns=STAGES, fill_value=0).to_csv(OUT / "confusion_stage_ryan_vs_qwen14b.csv")
coll = lambda s: s.replace({"onboarding": "ownership", "long_term_ownership": "ownership"})
pd.crosstab(coll(f.stage).rename("ryan"), k.stage.rename("small_model")).to_csv(OUT / "confusion_stage_ryan_vs_small.csv")

# family-level counts: Ryan yes / Qwen yes / both
fam_tab = pd.DataFrame({c: {"ryan_1": int(f[c].astype(int).sum()), "qwen14b_1": int(qfam.map(lambda s: c in s).sum()),
                            "both_1": int(((f[c] == "1") & qfam.map(lambda s: c in s)).sum())} for c in FAMS}).T
fam_tab.rename_axis("family").to_csv(OUT / "family_counts.csv")

# row-level disagreement file
rowlvl = pd.DataFrame({"id": f.index, "audience": f.audience, "subreddit": f.subreddit, "title": f.title,
    "ryan_stage": f.stage, "qwen14b_stage": q.stage, "small_stage": k.stage,
    "ryan_families": f[FAMS].apply(lambda r: "|".join(c for c in FAMS if r[c] == "1"), axis=1),
    "qwen14b_families": qfam.map(lambda s: "|".join(c for c in FAMS if c in s)),
    "ryan_stance": f.stance, "qwen14b_stance": q.stance, "ryan_note_zh": f.notes})
rowlvl["stage_agree"] = rowlvl.ryan_stage == rowlvl.qwen14b_stage
rowlvl["families_agree"] = rowlvl.ryan_families == rowlvl.qwen14b_families
rowlvl.to_csv(OUT / "row_level_ryan_vs_models.csv", index=False)
print(pd.Series(v).to_string()); print(res[cols].round(2).to_string())
