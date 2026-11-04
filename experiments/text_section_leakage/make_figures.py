"""Publication figures (aggregate inputs only) and their source-data CSVs."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_evaluation import load_wide
from common import load_config, results_dir

# Style of the paper's existing figures: DejaVu Serif, IBM colour-blind-safe palette, boxed axes, dotted grid.
BLUE, MAGENTA, GOLD = "#648fff", "#dc267f", "#ffb000"
INK, MUTED, GRID = "#000000", "#8c8c8c", "#b0b0b0"
SHORT = {"T0_COPIV": "T0 COP-IV", "T1_ROBUST": "T1 robust", "T2_SCREENED": "T2 screened",
         "T3_RETROSPECTIVE": "T3 + retrospective", "T4_OTHERPATIENT": "T4 + other patient"}
SECTION_LABEL = {"chief_complaint": "Chief Complaint", "history_of_present_illness": "History of Present Illness",
                 "past_medical_history": "Past Medical History", "medications_on_admission": "Medications on Admission",
                 "allergies": "Allergies", "physical_exam": "Physical Exam (admission part)",
                 "family_history": "Family History", "social_history": "Social History"}

LEGEND_FRAME = dict(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1, fancybox=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9, "axes.edgecolor": INK,
                     "axes.linewidth": 0.8, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linestyle": ":", "grid.linewidth": 0.7,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def fig_sections(res, figdir):
    d = pd.read_csv(res / "t1_candidates_by_section.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    sections = [s for s in SECTION_LABEL if s in set(d.section)]
    y = np.arange(len(sections))
    for ax, cat, title in zip(axes, ["outcome", "post_anchor"],
                              ["Explicit outcome disclosure", "Post-admission clinical information"]):
        sub = d[d.category == cat].set_index("section").reindex(sections).fillna(0)
        ax.scatter(sub.pct_of_survived, y + 0.12, s=28, color=BLUE, label="Survived", zorder=3,
                   edgecolor="white", linewidth=1)
        ax.scatter(sub.pct_of_died, y - 0.12, s=28, color=MAGENTA, label="Died in hospital", zorder=3,
                   edgecolor="white", linewidth=1)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Admissions with a rule-flagged span (%)")
        ax.set_xlim(left=0)
    axes[0].set_yticks(y, [SECTION_LABEL[s] for s in sections])
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()     # shared by both panels: above them
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, **LEGEND_FRAME)
    fig.savefig(figdir / "fig1_section_candidates.png")
    fig.savefig(figdir / "fig1_section_candidates.pdf")
    plt.close(fig)


def fig_prediction_change(cfg, res, figdir):
    wide = load_wide(cfg, "core")
    rows = []
    for c in ["T0_COPIV", "T2_SCREENED", "T3_RETROSPECTIVE", "T4_OTHERPATIENT"]:
        if c not in wide:
            continue
        for lab, name in [(1, "Died in hospital"), (0, "Survived")]:
            d = (wide[c] - wide["T1_ROBUST"])[wide.label == lab].values
            q = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append({"condition": c, "outcome": name, "n": len(d), "mean": d.mean(),
                         **dict(zip(["q05", "q25", "median", "q75", "q95"], q))})
    q = pd.DataFrame(rows)
    q.to_csv(res / "fig2_source_prediction_change_quantiles.csv", index=False)
    conds = list(dict.fromkeys(q.condition))
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(conds) * 2 + 0.8))
    for k, c in enumerate(conds):
        for off, (name, col) in zip([-0.15, 0.15], [("Died in hospital", MAGENTA), ("Survived", BLUE)]):
            r = q[(q.condition == c) & (q.outcome == name)].iloc[0]
            yy = k + off
            ax.plot([r.q05, r.q95], [yy, yy], color=col, lw=1, solid_capstyle="round")
            ax.plot([r.q25, r.q75], [yy, yy], color=col, lw=4, solid_capstyle="round")
            ax.scatter([r["median"]], [yy], s=30, color=col, edgecolor="white", linewidth=1, zorder=3,
                       label=name if k == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yticks(range(len(conds)), [SHORT[c] + " − T1" for c in conds])
    ax.invert_yaxis()
    ax.set_xlabel("Paired change in CORe mortality probability vs T1 (symlog axis)")
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.75), **LEGEND_FRAME)   # empty band below the T0 rows
    fig.savefig(figdir / "fig2_core_prediction_change.png")
    fig.savefig(figdir / "fig2_core_prediction_change.pdf")
    plt.close(fig)


def fig_performance(res, figdir):
    frames = []
    for model, name in [("core", "CORe (frozen)"), ("tfidf", "TF-IDF + LR (retrained)")]:
        p = res / f"{model}_metrics.csv"
        if p.exists():
            frames.append(pd.read_csv(p).assign(model=name))
    d = pd.concat(frames)
    d.to_csv(res / "fig3_source_performance.csv", index=False)
    conds = [c for c in SHORT if c in set(d.condition)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, metric, label in zip(axes, ["auroc", "auprc"], ["AUROC", "AUPRC"]):
        for off, (name, col) in zip([-0.13, 0.13], [("CORe (frozen)", BLUE), ("TF-IDF + LR (retrained)", GOLD)]):
            sub = d[d.model == name].set_index("condition").reindex(conds)
            yy = np.arange(len(conds)) + off
            ax.errorbar(sub[metric], yy, xerr=[sub[metric] - sub[f"{metric}_ci_low"],
                                               sub[f"{metric}_ci_high"] - sub[metric]],
                        fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.2, capsize=0, label=name,
                        markeredgecolor="white")
        ax.set_xlabel(f"{label} (95% CI)")
    axes[0].set_yticks(range(len(conds)), [SHORT[c] for c in conds])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=2, **LEGEND_FRAME)
    fig.savefig(figdir / "fig3_performance.png")
    fig.savefig(figdir / "fig3_performance.pdf")
    plt.close(fig)


SEVERITY = [  # (condition, label, kind) for the controlled-perturbation table and figure
    ("T3_RETROSPECTIVE", "+ discharge-phase sections of the same note", "injection"),
    ("T3A_DCOND", "+ Discharge Condition only", "injection"),
    ("T3B_DDIAG", "+ Discharge Diagnosis only", "injection"),
    ("T3C_BHC", "+ Brief Hospital Course only", "injection"),
    ("T3D_BHC_NOOUTCOME", "+ Hospital Course, outcome sentences removed", "injection"),
    ("T1_ICD", "+ stay-level ICD codes of the admission", "injection"),
    ("RANDOM_SPLIT", "random admission-level split (unpaired)", "injection"),
    ("MICRO", "microbiology by charttime, not storetime (48 h; own ref.)", "injection"),
    ("T0_COPIV", "original COP-IV section parser", "parser"),
    ("T2_SCREENED", "− automatically flagged candidate statements", "screening"),
    ("T1_NO_HPI", "− History of Present Illness (ablation)", "ablation"),
    ("T1_NO_PE", "− Physical Exam (ablation)", "ablation"),
    ("T1_PE_LABELLED", "− Physical Exam unless sub-exams labelled", "ablation"),
    ("T4_OTHERPATIENT", "+ another patient's discharge sections (control)", "control"),
    ("T4M_OUTCOMEMATCHED", "+ same-outcome patient's sections (oracle control)", "control"),
]


def severity_table(res) -> pd.DataFrame:
    t = pd.read_csv(res / "tfidf_metrics.csv").set_index("condition")
    rows = []
    for cond, label, kind in SEVERITY:
        if cond == "RANDOM_SPLIT":
            p = res / "tfidf_random_split.csv"
            if not p.exists():
                continue
            r = pd.read_csv(p).set_index("test_set").loc["random_admission_split"]
            vals = [r[f"delta_{m}_vs_patient_split"] if f"delta_{m}_vs_patient_split" in r else np.nan
                    for m in ["auroc"]] + [r["delta_auroc_ci_low"], r["delta_auroc_ci_high"], r["auroc"],
                                           r["delta_auprc_vs_patient_split"], r["delta_auprc_ci_low"],
                                           r["delta_auprc_ci_high"], r["auprc"]]
        elif cond == "MICRO":   # T1 text + microbiology, reference: results windowed by storetime
            p = res / "micro_metrics.csv"
            if not p.exists():
                continue
            m = pd.read_csv(p)
            r = m[(m.reference == "T1_MICRO_STORETIME") & (m.condition == "T1_MICRO_CHARTTIME")].iloc[0]
            vals = [r.delta_auroc_vs_ref, r.delta_auroc_ci_low, r.delta_auroc_ci_high, r.auroc,
                    r.delta_auprc_vs_ref, r.delta_auprc_ci_low, r.delta_auprc_ci_high, r.auprc]
        elif cond in t.index:
            r = t.loc[cond]
            vals = [r.delta_auroc_vs_ref, r.delta_auroc_ci_low, r.delta_auroc_ci_high, r.auroc,
                    r.delta_auprc_vs_ref, r.delta_auprc_ci_low, r.delta_auprc_ci_high, r.auprc]
        else:
            continue
        rows.append(dict(zip(["condition", "label", "kind", "d_auroc", "d_auroc_lo", "d_auroc_hi", "auroc",
                              "d_auprc", "d_auprc_lo", "d_auprc_hi", "auprc"], [cond, label, kind, *vals])))
    d = pd.DataFrame(rows)
    d.to_csv(res / "fig4_source_severity.csv", index=False)
    return d


def fig_severity(res, figdir):
    d = severity_table(res)
    colors = {"injection": MAGENTA, "parser": BLUE, "screening": BLUE, "ablation": GOLD, "control": MUTED}
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(d) + 0.9))
    y = np.arange(len(d))
    for k, r in d.iterrows():
        c = colors[r.kind]
        ax.plot([100 * r.d_auroc_lo, 100 * r.d_auroc_hi], [k, k], color=c, lw=1.5, solid_capstyle="round")
        ax.scatter([100 * r.d_auroc], [k], s=34, color=c, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(f"{100 * r.d_auroc:+.2f}".replace("-", "\u2212"), (100 * r.d_auroc_hi, k), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, d.label)
    ax.invert_yaxis()
    ax.set_xlabel("ΔAUROC vs T1 robust reconstruction (percentage points, 95% patient-clustered CI)")
    ax.set_xlim(right=max(9.0, 100 * d.d_auroc_hi.max() + 1.5))
    fig.savefig(figdir / "fig4_severity_ranking.png")
    fig.savefig(figdir / "fig4_severity_ranking.pdf")
    plt.close(fig)


def _ci(r, m, scale=100, nd=1):
    return f"{scale * r[m]:.{nd}f} [{scale * r[m + '_ci_low']:.{nd}f}, {scale * r[m + '_ci_high']:.{nd}f}]"


def _delta(r, m, scale=100, nd=2):
    if pd.isna(r.get(f"delta_{m}_ci_low")):
        return "ref."
    return (f"{scale * r[f'delta_{m}_vs_ref']:+.{nd}f} [{scale * r[f'delta_{m}_ci_low']:+.{nd}f}, "
            f"{scale * r[f'delta_{m}_ci_high']:+.{nd}f}]")


def model_table(res, model, title) -> list[str]:
    p = res / f"{model}_metrics.csv"
    if not p.exists():
        return []
    t = pd.read_csv(p)
    ch = pd.read_csv(res / f"{model}_prediction_change.csv").set_index("condition")
    out = [f"\n## {title}\n", "Point estimate [95% patient-clustered bootstrap CI]; Δ = condition − T1 "
           "(percentage points, paired).\n",
           "| Condition | N | AUROC % | ΔAUROC | AUPRC % | ΔAUPRC | Brier ×100 | Median abs. Δp vs T1 |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        out.append(f"| {r.condition} | {int(r.n_admissions):,} | {_ci(r, 'auroc')} | {_delta(r, 'auroc')} | "
                   f"{_ci(r, 'auprc')} | {_delta(r, 'auprc')} | {_ci(r, 'brier', nd=2)} | "
                   f"{ch.loc[r.condition, 'median_abs_change_vs_ref']:.4f} |")
    rs = res / f"{model}_random_split.csv"
    if rs.exists():
        out += ["\nRandom admission-level split (own test set; Δ vs the patient-level T1 test set, independent "
                "bootstraps):\n", "| Test set | N | Deaths | AUROC % | ΔAUROC | AUPRC % |", "|---|---|---|---|---|---|"]
        for _, r in pd.read_csv(rs).iterrows():
            d = ("–" if pd.isna(r.get("delta_auroc_vs_patient_split")) else
                 f"{100 * r.delta_auroc_vs_patient_split:+.2f} [{100 * r.delta_auroc_ci_low:+.2f}, "
                 f"{100 * r.delta_auroc_ci_high:+.2f}]")
            out.append(f"| {r.test_set} | {int(r.n_admissions):,} | {int(r.n_deaths)} | {_ci(r, 'auroc')} | {d} | "
                       f"{_ci(r, 'auprc')} |")
    return out


def tables_markdown(res, report_dir):
    out = ["# Tables (generated by make_figures.py from results/*.csv)\n"]
    a = pd.read_csv(res / "table_a_candidates.csv")
    out += ["## Table A — Extraction and automated retrospective-content candidates (analysis cohort, all splits)\n",
            "Candidate rates are automated rule hits (not human-verified). T2 is 0 by construction "
            "(flagged sentences removed); residual undetected content is not measured by these rules.\n",
            "| Condition | Admissions | Median words | Any candidate % | Post-admission % | Outcome disclosure % "
            "| Outcome disclosure % (died) | Outcome disclosure % (survived) |", "|---|---|---|---|---|---|---|---|"]
    for r in a.itertuples():
        mw = "–" if pd.isna(r.median_words) else f"{r.median_words:.0f}"
        out.append(f"| {r.condition} | {r.admissions:,} | {mw} | {r.cand_any_pct_all:.2f} | "
                   f"{r.cand_post_anchor_pct_all:.2f} | {r.cand_outcome_pct_all:.2f} | "
                   f"{r.cand_outcome_pct_died:.2f} | {r.cand_outcome_pct_survived:.2f} |")
    for model, title in [("core", "Table B — Frozen CORe (test split; 382 admission tokens, +128 injected for T3/T4)"),
                         ("tfidf", "Table C — TF-IDF + logistic regression retrained per condition (test split)")]:
        out += model_table(res, model, title)
    if (res / "fig4_source_severity.csv").exists():
        d = pd.read_csv(res / "fig4_source_severity.csv")
        out += ["\n## Table D — Task-specific sensitivity to controlled preprocessing perturbations (TF-IDF + LR "
                "retrained per condition, test split)\n",
                "Δ relative to T1 (percentage points) with 95% patient-clustered bootstrap CI; paired except the "
                "random split, which has its own test set (independent bootstraps).\n",
                "| Manipulation | Type | AUROC % | ΔAUROC | AUPRC % | ΔAUPRC |", "|---|---|---|---|---|---|"]
        for r in d.itertuples():
            out.append(f"| {r.label} | {r.kind} | {100 * r.auroc:.2f} | {100 * r.d_auroc:+.2f} "
                       f"[{100 * r.d_auroc_lo:+.2f}, {100 * r.d_auroc_hi:+.2f}] | {100 * r.auprc:.1f} | "
                       f"{100 * r.d_auprc:+.2f} [{100 * r.d_auprc_lo:+.2f}, {100 * r.d_auprc_hi:+.2f}] |")
        out.append("\nThe microbiology row uses its own cohort (admissions alive and in hospital at 48 h) and "
                   "reference (T1 text + microbiology windowed by storetime); see Table G.")
    p = res / "tfidf_metrics_by_pe_status.csv"
    if p.exists():
        out += ["\n## Table F — Physical Exam effect by exam documentation type (TF-IDF, test split)\n",
                "Paired Δ vs T1 within each stratum (percentage points, 95% patient-clustered CI). "
                "T1_PE_LABELLED keeps the exam only in notes with labelled admission/discharge sub-exams.\n",
                "| PE status | Condition | N | Deaths | AUROC % | ΔAUROC | AUPRC % | ΔAUPRC |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in pd.read_csv(p).iterrows():
            out.append(f"| {r.pe_status} | {r.condition} | {int(r.n_admissions):,} | {int(r.n_deaths)} | "
                       f"{_ci(r, 'auroc')} | {_delta(r, 'auroc')} | {_ci(r, 'auprc')} | {_delta(r, 'auprc')} |")
    p = res / "micro_metrics.csv"
    if p.exists():
        sm = json.loads((res / "micro_summary.json").read_text())
        out += ["\n## Table G — Microbiology windowed by charttime vs storetime (48 h anchor, TF-IDF/LR)\n",
                f"{sm['n_admissions']:,} admissions alive and in hospital at 48 h ({sm['n_test']:,} test, "
                f"{sm['deaths_test']} deaths). {sm['pct_admissions_with_micro_in_window']:.1f}% had a specimen "
                f"collected in the window; for {sm['pct_admissions_docs_differ']:.1f}% the two windowing rules give "
                f"different features. {sm['pct_rows_result_stored_after_anchor']:.1f}% of in-window result rows "
                f"were stored after the anchor (median delay {sm['median_store_delay_hours']:.0f} h).\n",
                "| Reference | Condition | AUROC % | ΔAUROC | AUPRC % | ΔAUPRC |", "|---|---|---|---|---|---|"]
        for _, r in pd.read_csv(p).iterrows():
            out.append(f"| {r.reference} | {r.condition} | {_ci(r, 'auroc')} | {_delta(r, 'auroc')} | "
                       f"{_ci(r, 'auprc')} | {_delta(r, 'auprc')} |")
    p = res / "exposure_stratified.csv"
    if p.exists():
        e = pd.read_csv(p)
        out += ["\n## Table H — Paired T1 → T2 prediction change by exposure to screened text (test split)\n",
                "Exposure comes from the screening log (automated candidates, not verified). For CORe, "
                "'in window' means the removal changed its 382-token input (|Δp| > 1e-5; 0% of unexposed "
                "admissions exceed this). Δ log loss > 0: T2 predicts the true outcome worse. 95% "
                "patient-clustered bootstrap CI.\n",
                "| Model | Stratum | Outcome | N | Mean Δp (pp) | % decreased | Mean Δ log loss [95% CI] |",
                "|---|---|---|---|---|---|---|"]
        keep = {"core": ["none", "outcome_in_window", "outcome_truncated_away", "post_anchor_only_in_window"],
                "tfidf": ["none", "outcome", "post_anchor_only"]}
        for _, r in e.iterrows():
            kind = "exposure_in_core_window" if r.model == "core" else "exposure"
            if r.stratification != kind or r.stratum not in keep[r.model]:
                continue
            out.append(f"| {r.model} | {r.stratum} | {r.outcome} | {int(r.n):,} | {100 * r.mean_delta_p:+.2f} | "
                       f"{r.pct_decreased:.0f} | {r.mean_delta_logloss:+.3f} [{r.delta_logloss_ci_low:+.3f}, "
                       f"{r.delta_logloss_ci_high:+.3f}] |")
    p = res / "ed_metrics.csv"
    if p.exists():
        sm = json.loads((res / "ed_reference_summary.json").read_text())
        out += ["\n## Table I — Reconstructed sections vs a time-available ED reference (TF-IDF/LR, ED-linked test admissions)\n",
                f"{sm['n_ed_linked']:,} analysis admissions linked to an ED stay ({sm['pct_of_analysis_cohort']:.1f}%); "
                f"test {sm['n_test']:,} with {sm['deaths_test']} deaths. ED_REF uses triage, ED vitals and medications "
                "charted and laboratory results stored before the hospital admission time (median "
                f"{sm['median_hours_ed_arrival_to_admission']:.0f} h after ED arrival); ED diagnoses are excluded. "
                "Each model is retrained within this subset; Δ is paired against the stated reference.\n",
                "| Reference | Condition | AUROC % | ΔAUROC | AUPRC % | ΔAUPRC |", "|---|---|---|---|---|---|"]
        for _, r in pd.read_csv(p).iterrows():
            out.append(f"| {r.reference} | {r.condition} | {_ci(r, 'auroc')} | {_delta(r, 'auroc')} | "
                       f"{_ci(r, 'auprc')} | {_delta(r, 'auprc')} |")
        out += ["\nPhysical Exam beyond ED data, by exam documentation type (ED_T1_NO_PE vs ED_T1):\n",
                "| PE status | N | Deaths | ED_T1 AUROC % | ΔAUROC | ED_T1 AUPRC % | ΔAUPRC |", "|---|---|---|---|---|---|---|"]
        e = pd.read_csv(res / "ed_metrics_by_pe_status.csv")
        for status, g in e.groupby("pe_status"):
            ref, r = g[g.condition == "ED_T1"].iloc[0], g[g.condition == "ED_T1_NO_PE"].iloc[0]
            out.append(f"| {status} | {int(r.n_admissions):,} | {int(r.n_deaths)} | {100 * ref.auroc:.1f} | "
                       f"{_delta(r, 'auroc')} | {100 * ref.auprc:.1f} | {_delta(r, 'auprc')} |")
    for model in ["tfidf"]:
        p = res / f"{model}_patient_overlap.csv"
        if not p.exists():
            continue
        o = pd.read_csv(p)
        o = o[o.condition.str.startswith("T1_OVL_B")]
        out += [f"\n## Table J ({model}) — Patient overlap with a fixed evaluation set\n",
                "One admission per multi-admission test patient (same set in both arms). Arm B adds the evaluation "
                "patients' other admissions to training and removes an equal number of training admissions (same "
                "deaths, also from repeat patients); Δ = B − A, paired, per training seed. Deaths are always a "
                "patient's last admission, so AUROC is shown for all evaluation admissions and for last admissions.\n",
                "| Seed | Stratum | N | Deaths | AUROC % (A) | ΔAUROC (B − A) | ΔAUPRC (B − A) |", "|---|---|---|---|---|---|---|"]
        a_rows = pd.read_csv(p)
        for _, r in o.iterrows():
            ref = a_rows[(a_rows.training_seed == r.training_seed) & (a_rows.stratum == r.stratum)
                         & a_rows.condition.str.startswith("T1_OVL_A")].iloc[0]
            out.append(f"| {r.training_seed} | {r.stratum} | {int(r.n_admissions):,} | {int(r.n_deaths)} | "
                       f"{100 * ref.auroc:.1f} | {_delta(r, 'auroc')} | {_delta(r, 'auprc')} |")
        sh = res / f"{model}_patient_overlap_shift.csv"
        if sh.exists():
            out += ["\nShift of predictions (B − A) by outcome and position of the evaluation admission:\n",
                    "| Seed | Last admission | Outcome | N | Mean Δp (pp) | Mean Δ log loss [95% CI] |", "|---|---|---|---|---|---|"]
            for _, r in pd.read_csv(sh).iterrows():
                out.append(f"| {r.training_seed} | {r.eval_is_last} | {r.outcome} | {int(r.n):,} | "
                           f"{100 * r.mean_delta_p:+.2f} | {r.mean_delta_logloss:+.4f} [{r.delta_logloss_ci_low:+.4f}, "
                           f"{r.delta_logloss_ci_high:+.4f}] |")
    p = res / "icd_decomposition_metrics.csv"
    if p.exists():
        sm = json.loads((res / "icd_decomposition_summary.json").read_text())
        names = {"T1_ICD": "+ all current codes (hashed with the text, Table D row)",
                 "T1_ICD_SEP": "+ all current codes (separate code block)",
                 "T1_ICD_NO_DEATH": "+ current codes without death/end-of-life codes (palliative care, DNR, brain death)",
                 "T1_ICD_NO_TERMINAL": "+ current codes without death/end-of-life and cardiac/respiratory arrest",
                 "T1_ICD_NO_ACUTE": "+ current codes without the above and organ dysfunction, sepsis, shock",
                 "T1_NCODES": "+ number of current codes only",
                 "T1_ICD_PRIOR": "+ codes of earlier admissions (known at admission)"}
        rows = pd.concat([pd.read_csv(res / "icd_ablation_metrics.csv"),
                          pd.read_csv(p).query("reference == 'T1_ROBUST'").drop(columns="reference")])
        rows = rows.drop_duplicates("condition").set_index("condition").loc[["T1_ROBUST", *names]].reset_index()
        out += ["\n## Table K — What the ICD-code row leaks (TF-IDF/LR, test split; `icd_feature_weights.py`, "
                "`icd_decomposition.py`)\n",
                "Codes enter as a separate, unhashed TF-IDF block next to the T1 text features, so each code has its "
                "own weight; Δ vs T1, paired. Code groups are defined by title (death/end-of-life, arrest) and by "
                "code prefix (organ dysfunction, sepsis, shock); per-code weights in `results/icd_feature_weights.csv`. "
                f"{sm['pct_test_with_prior_codes']:.1f}% of test admissions have codes from an earlier admission; "
                f"median number of current codes {sm['median_n_codes_test_deaths']:.0f} in deaths vs "
                f"{sm['median_n_codes_test_survivors']:.0f} in survivors.\n",
                "| Condition | AUROC % | ΔAUROC | AUPRC % | ΔAUPRC |", "|---|---|---|---|---|"]
        for _, r in rows.iterrows():
            out.append(f"| {names.get(r.condition, 'T1 text only')} | {_ci(r, 'auroc')} | {_delta(r, 'auroc')} | "
                       f"{_ci(r, 'auprc')} | {_delta(r, 'auprc')} |")
    (report_dir / "TABLES.md").write_text("\n".join(out) + "\n")


def main():
    cfg = load_config()
    res = results_dir(cfg)
    figdir = cfg["paths"]["report_dir"] / "figures"
    figdir.mkdir(exist_ok=True)
    fig_sections(res, figdir)
    fig_prediction_change(cfg, res, figdir)
    fig_performance(res, figdir)
    fig_severity(res, figdir)
    tables_markdown(res, cfg["paths"]["report_dir"])


if __name__ == "__main__":
    main()
