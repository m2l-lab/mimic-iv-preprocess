import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# Global styling for all plots
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

# mimic3-benchmark column names misspell "Glasgow" as "Glascow"
#cleaned up for plots below.
_DISPLAY_NAME_FIXES = [
    ("Glascow coma scale", "Glasgow Coma Scale"),
]


def _pretty_feature_name(name):
    for old, new in _DISPLAY_NAME_FIXES:
        name = name.replace(old, new)
    return name

VARIANT_PALETTE = ("#648FFF", "#FFB000") + tuple(plt.cm.tab10.colors[2:])


def _load_variant(results_dir, variant_label):
    feature_importance = pd.read_csv(
        os.path.join(results_dir, "{}_feature_importance.csv".format(variant_label)))
    modality_importance = pd.read_csv(
        os.path.join(results_dir, "{}_modality_importance.csv".format(variant_label)),
        index_col=0).iloc[:, 0]
    with open(os.path.join(results_dir, "{}_metrics.json".format(variant_label))) as f:
        metrics = json.load(f)
    shap_values = np.load(os.path.join(results_dir, "{}_shap_values.npy".format(variant_label)))
    test_features = pd.read_csv(os.path.join(results_dir, "{}_test_features.csv".format(variant_label)))
    return {
        "feature_importance": feature_importance,
        "modality_importance": modality_importance,  # Series: modality -> pct
        "metrics": metrics,
        "shap_values": shap_values,
        "test_features": test_features,
    }


def plot_top_features_bar(variant_data, output_path, top_n=10, title=None, bar_alpha=0.5):
    top = variant_data["feature_importance"].head(top_n).iloc[::-1]
    y = np.arange(len(top))
    vals = top["mean_abs_shap"].values

    fig, ax = plt.subplots(figsize=(6.2, max(3, 0.3 * len(top) + 1.0)))
    ax.grid(False)
    for yi in y:
        ax.axhline(yi, color="#d9d9d9", lw=0.5, ls=":", zorder=0)
    ax.barh(y, vals, height=0.6, color=VARIANT_PALETTE[0], alpha=bar_alpha, zorder=1)
    xmax = vals.max()
    if "mean_abs_shap_ci_lower" in top.columns and "mean_abs_shap_ci_upper" in top.columns:
        lo = np.clip(vals - top["mean_abs_shap_ci_lower"].values, 0, None)
        hi = np.clip(top["mean_abs_shap_ci_upper"].values - vals, 0, None)
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=VARIANT_PALETTE[0],
                    elinewidth=1.0, capsize=2.0, zorder=1.5)
        xmax = max(xmax, top["mean_abs_shap_ci_upper"].max())

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_feature_name(f) for f in top["feature"]])
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.set_xlim(0, xmax * 1.05)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("mean |SHAP value|")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def apply_bootstrap_shap_ci(importance_df, boot_df, label):
    sub = boot_df[(boot_df["run"] == label) & (boot_df["task"] == "mortality")].set_index("feature")
    missing = [f for f in importance_df["feature"] if f not in sub.index]
    if sub.empty or missing:
        raise ValueError("Bootstrap SHAP CSV lacks run '{}' or {} of its features".format(label, len(missing)))
    out = importance_df.copy()
    out["mean_abs_shap_ci_lower"] = sub["ci_lower"].reindex(out["feature"]).values
    out["mean_abs_shap_ci_upper"] = sub["ci_upper"].reindex(out["feature"]).values
    return out


def apply_bootstrap_image_share_ci(per_phenotype, boot_df, label):
    sub = boot_df[(boot_df["run"] == label) & (boot_df["feature"] == "image_share_pct")].set_index("task")
    for entry in per_phenotype:
        if entry["phenotype"] not in sub.index:
            raise ValueError("Bootstrap SHAP CSV lacks image share for '{}' / '{}'".format(label, entry["phenotype"]))
        r = sub.loc[entry["phenotype"]]
        entry["image_share_ci"] = (float(r["ci_lower"]), float(r["ci_upper"]))
    return per_phenotype


def _feature_value_lookup(variant_data_dict):
    return {
        label: dict(zip(vd["feature_importance"]["feature"], vd["feature_importance"]["mean_abs_shap"]))
        for label, vd in variant_data_dict.items()
    }


def _feature_ci_lookup(variant_data_dict):
    result = {}
    for label, vd in variant_data_dict.items():
        df = vd["feature_importance"]
        if "mean_abs_shap_ci_lower" in df.columns and "mean_abs_shap_ci_upper" in df.columns:
            result[label] = dict(zip(df["feature"], zip(df["mean_abs_shap_ci_lower"], df["mean_abs_shap_ci_upper"])))
    return result


def _render_dumbbell_panel(ax, data, variants, panel_caption, top_n, display_names, show_legend):
    value_lookup = _feature_value_lookup(data)
    ci_lookup = _feature_ci_lookup(data)
    v1, v2 = variants
    all_features = set(value_lookup.get(v1, {}).keys()) | set(value_lookup.get(v2, {}).keys())
    deltas = {f: abs(value_lookup.get(v1, {}).get(f, 0.0) - value_lookup.get(v2, {}).get(f, 0.0))
              for f in all_features}
    top_feats = sorted(all_features, key=deltas.get, reverse=True)[:top_n]
    top_feats_display = top_feats[::-1]  # largest change ends up at the top

    y = np.arange(len(top_feats_display))
    vals1 = [value_lookup.get(v1, {}).get(f, 0.0) for f in top_feats_display]
    vals2 = [value_lookup.get(v2, {}).get(f, 0.0) for f in top_feats_display]
    for yi in y:
        ax.axhline(y=yi, color="#BBBBBB", linewidth=0.8, linestyle=":", zorder=0)

    x_extent = max(vals1 + vals2, default=0.0)
    for variant_label, vals, color in [(v1, vals1, VARIANT_PALETTE[0]), (v2, vals2, VARIANT_PALETTE[1])]:
        var_ci = ci_lookup.get(variant_label)
        if var_ci:
            lower_err = [max(v - var_ci.get(f, (v, v))[0], 0.0) for f, v in zip(top_feats_display, vals)]
            upper_err = [max(var_ci.get(f, (v, v))[1] - v, 0.0) for f, v in zip(top_feats_display, vals)]
            ax.errorbar(vals, y, xerr=[lower_err, upper_err], fmt="none",
                        ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.6, zorder=1.5)
            x_extent = max([x_extent] + [v + e for v, e in zip(vals, upper_err)])
    ax.scatter(vals1, y, color=VARIANT_PALETTE[0], s=70,
               label=(display_names[0] if display_names else v1), zorder=2, edgecolor="white", linewidth=0.6)
    ax.scatter(vals2, y, color=VARIANT_PALETTE[1], s=70,
               label=(display_names[1] if display_names else v2), zorder=2, edgecolor="white", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_feature_name(f) for f in top_feats_display])
    ax.set_xlabel("mean |SHAP value|")
    ax.grid(axis="y", visible=False)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.text(0.0, 1.02, "({})".format(panel_caption), transform=ax.transAxes,
            fontsize=10, fontweight="normal", va="bottom", ha="left")
    return x_extent


def _nice_axis_max(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(value))
    for step in (1.0, 2.0, 2.5, 3.0, 5.0, 10.0):
        if value <= step * magnitude + 1e-12:
            return step * magnitude
    return 10.0 * magnitude


def plot_feature_importance_dumbbell_panels(data_panel_a, data_panel_b, output_path,
                                             variants_a, variants_b, display_names=None,
                                             panel_a_caption="a", panel_b_caption="b",
                                             data_panel_c=None, data_panel_d=None,
                                             variants_c=None, variants_d=None,
                                             panel_c_caption="c", panel_d_caption="d",
                                             top_n=10, xmax=None):

    has_cd = data_panel_c is not None and data_panel_d is not None
    if has_cd and (variants_c is None or variants_d is None):
        raise ValueError("variants_c and variants_d are required when data_panel_c/data_panel_d are given.")

    panels = [(data_panel_a, variants_a, panel_a_caption), (data_panel_b, variants_b, panel_b_caption)]
    if has_cd:
        panels += [(data_panel_c, variants_c, panel_c_caption), (data_panel_d, variants_d, panel_d_caption)]
    n_panels = len(panels)

    panel_height = max(4, 0.4 * top_n)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, n_panels * panel_height + 0.2 * n_panels),
                              gridspec_kw={"hspace": 0.15})

    extents = []
    for i, (data, variants, panel_caption) in enumerate(panels):
        extents.append(_render_dumbbell_panel(axes[i], data, variants, panel_caption, top_n, display_names,
                                              show_legend=(i == n_panels - 1)))

    data_max = max(extents, default=0.0)
    if xmax is None:
        xmax = _nice_axis_max(data_max)
    elif data_max > xmax:
        print("WARNING: dumbbell --xmax {} is smaller than the largest plotted value/CI ({:.4f}); "
              "points beyond it will be cut off.".format(xmax, data_max))
    ticks = matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(0, xmax)
    ticks = [t for t in ticks if -1e-12 <= t <= xmax + 1e-12]
    for ax in axes:
        ax.set_xlim(-0.03 * xmax, xmax * 1.03)
        ax.set_xticks(ticks)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Saved dumbbell panel plot to {}".format(output_path))


def load_phenotype_run(results_dir, variant_label):

    path = os.path.join(results_dir, "{}_all_metrics.json".format(variant_label))
    with open(path) as f:
        summary = json.load(f)
    per_phenotype = summary["per_phenotype"]

    for entry in per_phenotype:
        prefix = "{}_{}".format(variant_label, _phenotype_safe_name(entry["phenotype"]))
        perseed_path = os.path.join(results_dir, "{}_perseed_metrics.json".format(prefix))
        if os.path.exists(perseed_path):
            with open(perseed_path) as f:
                entry["perseed"] = json.load(f)
        modality_path = os.path.join(results_dir, "{}_modality_importance_detailed.csv".format(prefix))
        if os.path.exists(modality_path):
            mdf = pd.read_csv(modality_path)
            img = mdf[mdf["modality"] == "image"]
            if len(img) and {"ci_lower", "ci_upper"} <= set(mdf.columns):
                entry["image_share_ci"] = (float(img["ci_lower"].iloc[0]), float(img["ci_upper"].iloc[0]))
    return per_phenotype


def _phenotype_safe_name(phenotype):
    return re.sub(r"[^A-Za-z0-9]+", "_", phenotype).strip("_")


def _delta_ci(entry_with, entry_without, metric_key, confidence=0.95):
    from scipy import stats as scipy_stats

    ps_w, ps_wo = entry_with.get("perseed"), entry_without.get("perseed")
    if not ps_w or not ps_wo:
        return None
    w = {r["seed"]: r[metric_key] for r in ps_w if r.get(metric_key) == r.get(metric_key)}
    wo = {r["seed"]: r[metric_key] for r in ps_wo if r.get(metric_key) == r.get(metric_key)}
    if len(w) < 2 or len(wo) < 2:
        return None

    if set(w) == set(wo):
        d = np.array([w[s] - wo[s] for s in sorted(w)])
        n = len(d)
        mean = float(d.mean())
        half = float(scipy_stats.t.ppf((1 + confidence) / 2, n - 1) * d.std(ddof=1) / np.sqrt(n))
        return mean, mean - half, mean + half, "paired"

    a, b = np.array(list(w.values())), np.array(list(wo.values()))
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    mean = float(a.mean() - b.mean())
    se = np.sqrt(va + vb)
    if se == 0:
        return mean, mean, mean, "welch"
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = float(scipy_stats.t.ppf((1 + confidence) / 2, df) * se)
    return mean, mean - half, mean + half, "welch"


_BOOTSTRAP_METRICS = {"test_auroc": "auroc", "test_auprc": "auprc"}


def load_bootstrap_deltas(csv_path, results_dir_with, label_with, results_dir_without, label_without):
    df = pd.read_csv(csv_path)

    def pred_path(results_dir, label):
        return os.path.normpath(os.path.join(results_dir, "{}_test_predictions.csv".format(label)))

    want_a = pred_path(results_dir_with, label_with)
    want_b = pred_path(results_dir_without, label_without)
    sel = df[(df["a"].map(os.path.normpath) == want_a) & (df["b"].map(os.path.normpath) == want_b)]
    if sel.empty:
        pairs = df[["a", "b"]].drop_duplicates().values.tolist()
        raise ValueError("No row in {} for a={} / b={}. Pairs present: {}".format(
            csv_path, want_a, want_b, pairs))
    result = {}
    for _, r in sel.iterrows():
        result.setdefault(r["metric"], {})[r["task"]] = (
            float(r["diff_point"]), float(r["diff_ci_lower"]), float(r["diff_ci_upper"]))
    return result


def _phenotype_display_name(name):

    return re.sub(r"\s*\([^)]*\)", "", name).strip()


_DELTA_METRIC_STYLE = {
    "test_auroc": {"color": VARIANT_PALETTE[1], "marker": "o", "s": 30, "ms": 6.5},
    "test_auprc": {"color": "#DC267F", "marker": "D", "s": 26, "ms": 5.5},
}


_DELTA_METRICS = ("test_auroc", "test_auprc")


def _shap_delta_rows(per_phenotype_with, per_phenotype_without, show_ci, bootstrap):
    values_with = {e["phenotype"]: e for e in per_phenotype_with}
    values_without = {e["phenotype"]: e for e in per_phenotype_without}
    rows, ci_methods = {}, set()
    for p in values_with:
        if p not in values_without:
            continue
        img_pct = values_with[p].get("modality_importance_pct", {}).get("image")
        if img_pct is None:
            continue
        deltas, cis = {}, {}
        for mk in _DELTA_METRICS:
            v_with, v_without = values_with[p].get(mk), values_without[p].get(mk)
            if v_with is None or v_without is None or v_with != v_with or v_without != v_without:
                deltas = None  # NaN-safe: drop phenotype if any metric is missing
                break
            if bootstrap is not None and show_ci:
                entry = bootstrap.get(_BOOTSTRAP_METRICS[mk], {}).get(p)
                if entry is None:
                    raise ValueError("Phenotype '{}' / metric '{}' not found in bootstrap CSV.".format(
                        p, _BOOTSTRAP_METRICS[mk]))
                deltas[mk] = entry[0]
                cis[mk] = (entry[1], entry[2])
                ci_methods.add("bootstrap")
                continue
            ci = _delta_ci(values_with[p], values_without[p], mk) if show_ci else None
            if ci is not None:
                deltas[mk] = ci[0]
                cis[mk] = (ci[1], ci[2])
                ci_methods.add(ci[3])
            else:
                deltas[mk] = v_with - v_without
        if deltas is not None:
            rows[p] = (img_pct, deltas, cis, values_with[p].get("image_share_ci") if show_ci else None)
    return rows, ci_methods


def plot_shap_vs_performance_delta(runs_with, per_phenotype_without, output_path,
                                    bar_alpha=0.5, min_delta_lim=0.10, show_ci=True):
    """runs_with: list of (per_phenotype_with, bootstrap_or_None); one panel per entry, rows ordered
    by the image SHAP share of the first panel, shared delta axis."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.font_manager import FontProperties

    panel_rows, ci_methods = [], set()
    for per_phenotype_with, bootstrap in runs_with:
        rows, methods = _shap_delta_rows(per_phenotype_with, per_phenotype_without, show_ci, bootstrap)
        panel_rows.append(rows)
        ci_methods |= methods
    phenotypes = [p for p in panel_rows[0] if all(p in r for r in panel_rows[1:])]
    if not phenotypes:
        raise ValueError("No phenotypes with image SHAP share and valid metrics found in all runs.")
    phenotypes.sort(key=lambda p: panel_rows[0][p][0])  # ascending -> highest share at the top

    y = np.arange(len(phenotypes))
    labels = [_phenotype_display_name(p) for p in phenotypes]
    label_fs = FontProperties(size=plt.rcParams["axes.labelsize"]).get_size_in_points()
    offsets = {"test_auroc": 0.15, "test_auprc": -0.15}

    extent = []
    for rows in panel_rows:
        for p in phenotypes:
            for mk in _DELTA_METRICS:
                extent.append(abs(rows[p][1][mk]))
                extent.extend(abs(c) for c in rows[p][2].get(mk, ()))
    lim = max(min_delta_lim, np.ceil(max(extent) * 1.05 / 0.05) * 0.05)

    n_panels = len(panel_rows)
    fig_h = max(4, 0.3 * len(phenotypes) + 1.2)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, fig_h),
                             sharey=True, gridspec_kw={"wspace": 0.15})
    axes = np.atleast_1d(axes)

    for i, (ax, rows) in enumerate(zip(axes, panel_rows)):
        ax.grid(False)
        for yi in y:
            ax.axhline(yi, color="#d9d9d9", lw=0.5, ls=":", zorder=0)

        shap_share = np.array([rows[p][0] for p in phenotypes])
        ax.barh(y, shap_share, height=0.6, color=VARIANT_PALETTE[0], alpha=bar_alpha, zorder=1)
        share_ci = [rows[p][3] for p in phenotypes]
        if all(c is not None for c in share_ci):
            lo = np.clip(shap_share - np.array([c[0] for c in share_ci]), 0, None)
            hi = np.clip(np.array([c[1] for c in share_ci]) - shap_share, 0, None)
            ax.errorbar(shap_share, y, xerr=[lo, hi], fmt="none", ecolor=VARIANT_PALETTE[0],
                        elinewidth=1.0, capsize=2.0, zorder=1.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("image modality share of total mean |SHAP| (%)")
        ax.set_ylim(-0.7, len(phenotypes) - 0.3)
        ax.tick_params(axis="y", length=0, labelsize=label_fs)
        ax.tick_params(axis="x", labelsize=label_fs)
        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labels)

        top = ax.twiny()
        top.grid(False)
        top.set_xlim(-lim, lim)
        top.set_xticks(np.round(np.linspace(-lim, lim, 5), 3))
        top.tick_params(axis="x", labelsize=label_fs)
        top.spines["top"].set_visible(True)
        top.spines["right"].set_visible(False)
        top.set_xlabel("Δ AUROC / Δ AUPRC (structured+CXR − structured)", labelpad=6)
        top.axvline(0, color="#555555", lw=0.8, ls="--", zorder=2)

        for mk in _DELTA_METRICS:
            style = _DELTA_METRIC_STYLE[mk]
            deltas = np.array([rows[p][1][mk] for p in phenotypes])
            ci_arr = [rows[p][2].get(mk) for p in phenotypes]
            yy = y + offsets[mk]
            if all(c is not None for c in ci_arr):
                lo = np.clip(deltas - np.array([c[0] for c in ci_arr]), 0, None)
                hi = np.clip(np.array([c[1] for c in ci_arr]) - deltas, 0, None)
                top.errorbar(deltas, yy, xerr=[lo, hi], fmt="none", ecolor=style["color"],
                             elinewidth=1.0, capsize=2.0, zorder=2.5)
            top.scatter(deltas, yy, s=style["s"], color=style["color"], marker=style["marker"],
                        edgecolor="white", linewidth=0.9, zorder=3)

        if n_panels > 1:
            ax.annotate("({})".format("ab"[i]), xy=(0, 1), xycoords="axes fraction",
                        xytext=(0, 36), textcoords="offset points",
                        fontsize=label_fs, fontweight="normal", va="bottom", ha="left")

    handles = [Patch(facecolor=VARIANT_PALETTE[0], alpha=bar_alpha,
                     label="image modality share of total mean |SHAP| (bottom axis)")] + [
        Line2D([], [], marker=_DELTA_METRIC_STYLE[mk]["marker"], ls="",
               color=_DELTA_METRIC_STYLE[mk]["color"], markeredgecolor="white",
               markersize=_DELTA_METRIC_STYLE[mk]["ms"],
               label="Δ {} (top axis)".format("AUROC" if mk == "test_auroc" else "AUPRC"))
        for mk in _DELTA_METRICS]
    y0 = axes[0].get_position().y0 - 0.55 / fig_h
    x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x_mid, y0),
               ncol=len(handles) if n_panels > 1 else 1, frameon=False, fontsize=label_fs)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    if ci_methods == {"bootstrap"}:
        print("  delta CIs: 95% patient-clustered paired bootstrap (test-set sampling)")
    elif ci_methods:
        print("  delta CIs: {} t-interval over seeds (training stochasticity only)".format(
            " / ".join(sorted(ci_methods))))
    elif show_ci:
        print("  no per-seed results found -> plotted without CIs")
    print("Saved SHAP-vs-performance-delta plot to {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description="Generate HAIM-style plots from train_mortality.py outputs.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Top-10 features of one variant (mean |SHAP|, seed CIs).")
    p_single.add_argument("--results_dir", required=True)
    p_single.add_argument("--variant_label", required=True)
    p_single.add_argument("--output_dir", default=None,
                           help="Defaults to --results_dir if not given.")
    p_single.add_argument("--format", choices=["png", "svg", "pdf"], default="pdf",
                           help="Output file format (default: pdf).")
    p_single.add_argument("--bootstrap_shap_csv", default=None,
                           help="bootstrap_shap output; whiskers become patient-clustered bootstrap CIs.")

    p_dumbbell = sub.add_parser("dumbbell",
                                 help="Two-panel dumbbell plot")
    p_dumbbell.add_argument("--results_dir_a1", required=True, help="Panel A, first variant's results dir.")
    p_dumbbell.add_argument("--variant_label_a1", required=True)
    p_dumbbell.add_argument("--results_dir_a2", required=True, help="Panel A, second variant's results dir.")
    p_dumbbell.add_argument("--variant_label_a2", required=True)
    p_dumbbell.add_argument("--results_dir_b1", required=True, help="Panel B, first variant's results dir.")
    p_dumbbell.add_argument("--variant_label_b1", required=True)
    p_dumbbell.add_argument("--results_dir_b2", required=True, help="Panel B, second variant's results dir.")
    p_dumbbell.add_argument("--variant_label_b2", required=True)
    p_dumbbell.add_argument("--panel_a_caption", type=str, default="a")
    p_dumbbell.add_argument("--panel_b_caption", type=str, default="b")
    p_dumbbell.add_argument("--legend_name_1", type=str, default=None,
                             help="Display name for the first variant")
    p_dumbbell.add_argument("--legend_name_2", type=str, default=None,
                             help="Display name for the second variant")
    p_dumbbell.add_argument("--output_path", required=True, help="Full output file path")
    p_dumbbell.add_argument("--top_n", type=int, default=10)
    p_dumbbell.add_argument("--xmax", type=float, default=None,
                             help="Shared upper x-axis limit for ALL panels")
    p_dumbbell.add_argument("--results_dir_c1", default=None, help="Optional panel C, first variant's results dir.")
    p_dumbbell.add_argument("--variant_label_c1", default=None)
    p_dumbbell.add_argument("--results_dir_c2", default=None, help="Optional panel C, second variant's results dir.")
    p_dumbbell.add_argument("--variant_label_c2", default=None)
    p_dumbbell.add_argument("--results_dir_d1", default=None, help="Optional panel D, first variant's results dir.")
    p_dumbbell.add_argument("--variant_label_d1", default=None)
    p_dumbbell.add_argument("--results_dir_d2", default=None, help="Optional panel D, second variant's results dir.")
    p_dumbbell.add_argument("--variant_label_d2", default=None)
    p_dumbbell.add_argument("--panel_c_caption", type=str, default="c")
    p_dumbbell.add_argument("--panel_d_caption", type=str, default="d")
    p_dumbbell.add_argument("--bootstrap_shap_csv", default=None,
                             help="bootstrap_shap output; whiskers become patient-clustered bootstrap CIs.")

    p_svd = sub.add_parser("shap_vs_delta",
                            help="One row per phenotype: image SHAP share as bars (bottom axis) overlaid with AUROC/AUPRC")
    p_svd.add_argument("--results_dir_with", required=True, help="Results dir of the WITH-images run.")
    p_svd.add_argument("--variant_label_with", required=True)
    p_svd.add_argument("--results_dir_without", required=True, help="Results dir of the --no_image_features run.")
    p_svd.add_argument("--variant_label_without", required=True)
    p_svd.add_argument("--output_path", required=True, help="Full output file path (extension sets format).")
    p_svd.add_argument("--results_dir_with_b", default=None,
                        help="Optional second WITH-images run (e.g. PCA-20) drawn as right panel (b).")
    p_svd.add_argument("--variant_label_with_b", default=None)
    p_svd.add_argument("--bar_alpha", type=float, default=0.5,
                        help="Opacity of the image-SHAP-share bars (default 0.5).")
    p_svd.add_argument("--min_delta_lim", type=float, default=0.10,
                        help="Minimum half-width of the symmetric delta axis (default 0.10).")
    p_svd.add_argument("--no_ci", dest="show_ci", action="store_false", default=True,
                        help="Do not draw across-seed CI whiskers (default: drawn when per-seed files exist).")
    p_svd.add_argument("--bootstrap_shap_csv", default=None,
                        help="bootstrap_shap phenotype output; image-share whiskers become bootstrap CIs.")
    p_svd.add_argument("--bootstrap_csv", default=None,
                        help="bootstrap_compare output (with-images run as a, without as b); "
                             "delta points and CIs are then taken from it instead of across-seed t-intervals.")

    args = parser.parse_args()

    if args.mode == "single":
        output_dir = args.output_dir or args.results_dir
        os.makedirs(output_dir, exist_ok=True)
        vd = _load_variant(args.results_dir, args.variant_label)
        if args.bootstrap_shap_csv:
            vd["feature_importance"] = apply_bootstrap_shap_ci(
                vd["feature_importance"], pd.read_csv(args.bootstrap_shap_csv), args.variant_label)

        plot_top_features_bar(vd, os.path.join(output_dir, "{}_top_features.{}".format(args.variant_label, args.format)),
                               title="Top features -- {}".format(args.variant_label))
        print("Saved top features plot to {}".format(output_dir))

    elif args.mode == "dumbbell":
        data_panel_a = {
            args.variant_label_a1: _load_variant(args.results_dir_a1, args.variant_label_a1),
            args.variant_label_a2: _load_variant(args.results_dir_a2, args.variant_label_a2),
        }
        data_panel_b = {
            args.variant_label_b1: _load_variant(args.results_dir_b1, args.variant_label_b1),
            args.variant_label_b2: _load_variant(args.results_dir_b2, args.variant_label_b2),
        }
        display_names = None
        if args.legend_name_1 or args.legend_name_2:
            display_names = (args.legend_name_1 or args.variant_label_a1, args.legend_name_2 or args.variant_label_a2)

        data_panel_c = data_panel_d = variants_c = variants_d = None
        has_cd = args.results_dir_c1 and args.results_dir_c2 and args.results_dir_d1 and args.results_dir_d2
        if has_cd:
            data_panel_c = {
                args.variant_label_c1: _load_variant(args.results_dir_c1, args.variant_label_c1),
                args.variant_label_c2: _load_variant(args.results_dir_c2, args.variant_label_c2),
            }
            data_panel_d = {
                args.variant_label_d1: _load_variant(args.results_dir_d1, args.variant_label_d1),
                args.variant_label_d2: _load_variant(args.results_dir_d2, args.variant_label_d2),
            }
            variants_c = (args.variant_label_c1, args.variant_label_c2)
            variants_d = (args.variant_label_d1, args.variant_label_d2)

        if args.bootstrap_shap_csv:
            boot_df = pd.read_csv(args.bootstrap_shap_csv)
            for panel in (data_panel_a, data_panel_b, data_panel_c, data_panel_d):
                for label, vd in (panel or {}).items():
                    vd["feature_importance"] = apply_bootstrap_shap_ci(vd["feature_importance"], boot_df, label)
        plot_feature_importance_dumbbell_panels(
            data_panel_a, data_panel_b, args.output_path,
            variants_a=(args.variant_label_a1, args.variant_label_a2),
            variants_b=(args.variant_label_b1, args.variant_label_b2),
            display_names=display_names,
            panel_a_caption=args.panel_a_caption, panel_b_caption=args.panel_b_caption,
            data_panel_c=data_panel_c, data_panel_d=data_panel_d,
            variants_c=variants_c, variants_d=variants_d,
            panel_c_caption=args.panel_c_caption, panel_d_caption=args.panel_d_caption,
            top_n=args.top_n, xmax=args.xmax)

    elif args.mode == "shap_vs_delta":
        per_phenotype_without = load_phenotype_run(args.results_dir_without, args.variant_label_without)
        with_runs = [(args.results_dir_with, args.variant_label_with)]
        if args.results_dir_with_b:
            with_runs.append((args.results_dir_with_b, args.variant_label_with_b))
        runs_with = []
        for results_dir, label in with_runs:
            bootstrap = None
            if args.bootstrap_csv:
                bootstrap = load_bootstrap_deltas(args.bootstrap_csv, results_dir, label,
                                                  args.results_dir_without, args.variant_label_without)
            per_phenotype = load_phenotype_run(results_dir, label)
            if args.bootstrap_shap_csv:
                apply_bootstrap_image_share_ci(per_phenotype, pd.read_csv(args.bootstrap_shap_csv), label)
            runs_with.append((per_phenotype, bootstrap))
        plot_shap_vs_performance_delta(
            runs_with, per_phenotype_without, args.output_path,
            bar_alpha=args.bar_alpha, min_delta_lim=args.min_delta_lim, show_ci=args.show_ci)

if __name__ == "__main__":
    main()
