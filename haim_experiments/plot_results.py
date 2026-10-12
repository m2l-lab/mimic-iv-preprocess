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
import shap

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
MODALITY_COLORS = {"timeseries": "#4C72B0", "tabular": "#DD8452", "image": "#55A868", "other": "#8C8C8C"}


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


def plot_shap_summary(variant_data, output_path, max_display=20, title=None):
    shap.summary_plot(
        variant_data["shap_values"], variant_data["test_features"],
        max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_features_bar(variant_data, output_path, top_n=20, title=None):
    top = variant_data["feature_importance"].head(top_n).iloc[::-1]
    bar_colors = [MODALITY_COLORS.get(m, "#8C8C8C") for m in top["modality"]]
    display_labels = [_pretty_feature_name(f) for f in top["feature"]]
    has_ci = "mean_abs_shap_ci_lower" in top.columns and "mean_abs_shap_ci_upper" in top.columns

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(top))))
    bars = ax.barh(display_labels, top["mean_abs_shap"], color=bar_colors,
                    edgecolor="white", linewidth=0.6, height=0.7)
    ax.set_xlabel("mean |SHAP value|")
    ax.grid(axis="y", visible=False)

    xmax = top["mean_abs_shap"].max()
    label_offset = xmax * 0.01
    if has_ci:
        lower_err = (top["mean_abs_shap"] - top["mean_abs_shap_ci_lower"]).clip(lower=0)
        upper_err = (top["mean_abs_shap_ci_upper"] - top["mean_abs_shap"]).clip(lower=0)
        ax.errorbar(top["mean_abs_shap"], display_labels, xerr=[lower_err, upper_err],
                     fmt="none", ecolor="#333333", elinewidth=1.2, capsize=3, zorder=3)
        xmax = max(xmax, top["mean_abs_shap_ci_upper"].max())
        label_offset = xmax * 0.01

    for i, (bar, val) in enumerate(zip(bars, top["mean_abs_shap"])):
        x_end = top["mean_abs_shap_ci_upper"].iloc[i] if has_ci else val
        ax.text(x_end + label_offset, bar.get_y() + bar.get_height() / 2,
                "{:.3f}".format(val), va="center", fontsize=8.5, color="#333333")

    present_modalities = [m for m in MODALITY_COLORS if m in set(top["modality"])]
    if len(present_modalities) > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLORS[m]) for m in present_modalities]
        ax.legend(handles, present_modalities, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _grouped_feature_bars(value_lookup, variant_labels, features_display, output_path,
                           xlabel="mean |SHAP value|", title=None, ci_lookup=None):
    n_variants = len(variant_labels)
    height = 0.8 / n_variants
    y = np.arange(len(features_display))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(features_display))))
    xmax = max((value_lookup[l].get(f, 0.0) for l in variant_labels for f in features_display), default=0.0)
    if ci_lookup:
        ci_uppers = [ci_lookup[l][f][1] for l in variant_labels if l in ci_lookup
                     for f in features_display if f in ci_lookup[l]]
        if ci_uppers:
            xmax = max(xmax, max(ci_uppers))

    for i, label in enumerate(variant_labels):
        vals = [value_lookup[label].get(f, 0.0) for f in features_display]
        offset = ((n_variants - 1) / 2 - i) * height
        y_positions = y + offset
        bars = ax.barh(y_positions, vals, height=height * 0.92,
                        label=label, color=VARIANT_PALETTE[i % len(VARIANT_PALETTE)],
                        edgecolor="white", linewidth=0.5)

        val_ci = ci_lookup.get(label) if ci_lookup else None
        if val_ci:
            lower_err = [max(v - val_ci.get(f, (v, v))[0], 0.0) for f, v in zip(features_display, vals)]
            upper_err = [max(val_ci.get(f, (v, v))[1] - v, 0.0) for f, v in zip(features_display, vals)]
            ax.errorbar(vals, y_positions, xerr=[lower_err, upper_err],
                        fmt="none", ecolor="#333333", elinewidth=1.0, capsize=2.5, zorder=3)

        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 0:
                x_end = val + upper_err[j] if val_ci else val
                ax.text(x_end + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                        "{:.3f}".format(val), va="center", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_feature_name(f) for f in features_display])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance_comparison(variant_data_dict, output_path, top_n=10, title=None):
    variant_labels = list(variant_data_dict.keys())

    value_lookup = {}
    all_features = set()
    for label, vd in variant_data_dict.items():
        fi = vd["feature_importance"]
        all_features.update(fi.head(top_n)["feature"])
        value_lookup[label] = dict(zip(fi["feature"], fi["mean_abs_shap"]))

    def combined_score(feat):
        return sum(value_lookup[label].get(feat, 0.0) for label in variant_labels)

    features = sorted(all_features, key=combined_score, reverse=True)
    features_display = features[::-1]  # so highest ends up at the top of a barh plot

    ci_lookup = _feature_ci_lookup(variant_data_dict)
    _grouped_feature_bars(value_lookup, variant_labels, features_display, output_path,
                          title=title, ci_lookup=ci_lookup)
    print("Saved feature importance comparison to {}".format(output_path))


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


def _feature_spreads(value_lookup, variant_labels):
    all_features = set()
    for v in value_lookup.values():
        all_features.update(v.keys())
    spreads = {}
    for feat in all_features:
        vals = [value_lookup[l].get(feat, 0.0) for l in variant_labels]
        spreads[feat] = max(vals) - min(vals)
    return spreads


def plot_feature_importance_diff(variant_data_dict, output_path, top_n=10, title=None):
    variant_labels = list(variant_data_dict.keys())
    value_lookup = _feature_value_lookup(variant_data_dict)
    spreads = _feature_spreads(value_lookup, variant_labels)

    top_feats = sorted(spreads, key=spreads.get, reverse=True)[:top_n]
    features_display = top_feats[::-1]

    ci_lookup = _feature_ci_lookup(variant_data_dict)
    _grouped_feature_bars(value_lookup, variant_labels, features_display, output_path,
                           xlabel="mean |SHAP value|  (features sorted by spread across variants)",
                           title=title, ci_lookup=ci_lookup)
    print("Saved feature importance difference plot to {}".format(output_path))


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


def plot_feature_importance_spread_histogram(variant_data_dict, output_path, bins=30,
                                               highlight_features=None, title=None):

    variant_labels = list(variant_data_dict.keys())
    value_lookup = _feature_value_lookup(variant_data_dict)
    spreads = _feature_spreads(value_lookup, variant_labels)

    values = np.array(list(spreads.values()))
    median = np.median(values)
    p90 = np.percentile(values, 90)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color=VARIANT_PALETTE[0], edgecolor="white", alpha=0.85)
    ymax = ax.get_ylim()[1]
    ax.axvline(median, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(median, ymax * 0.98, " median={:.3f}".format(median),
            rotation=90, va="top", ha="left", fontsize=8.5, color="#444444")
    ax.axvline(p90, color="#444444", linestyle=":", linewidth=1.2)
    ax.text(p90, ymax * 0.75, " 90th pct={:.3f}".format(p90),
            rotation=90, va="top", ha="left", fontsize=8.5, color="#444444")

    if highlight_features:
        for feat in highlight_features:
            if feat not in spreads:
                continue
            v = spreads[feat]
            ax.axvline(v, color=VARIANT_PALETTE[1], linewidth=1.5)
            ax.text(v, ymax * 0.5, " {}={:.3f}".format(feat, v),
                    rotation=90, va="top", ha="left", fontsize=8.5, color=VARIANT_PALETTE[1])

    ax.set_xlabel("spread of mean |SHAP value| across variants  (max - min)")
    ax.set_ylabel("number of features")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Saved feature importance spread histogram to {}".format(output_path))


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


def _sort_key(entry, sort_by):
    if sort_by == "auroc":
        return entry.get("test_auroc") if entry.get("test_auroc") == entry.get("test_auroc") else -1  # NaN-safe
    if sort_by == "auprc":
        return entry.get("test_auprc", -1)
    return entry["phenotype"]  # name, sorted alphabetically ascending below


def _phenotype_label(entry):
    auroc = entry.get("test_auroc")
    auprc = entry.get("test_auprc")
    auroc_s = "{:.3f}".format(auroc) if auroc == auroc else "n/a"  # NaN-safe
    auprc_s = "{:.3f}".format(auprc) if auprc == auprc else "n/a"
    return "{}  (AUROC {}, AUPRC {})".format(entry["phenotype"], auroc_s, auprc_s)


def plot_phenotype_modality_stacked(per_phenotype, output_path, sort_by="auroc", title=None):
    entries = sorted(per_phenotype, key=lambda e: _sort_key(e, sort_by),
                      reverse=(sort_by in ("auroc", "auprc")))
    entries = entries[::-1]  

    labels = [_phenotype_label(e) for e in entries]
    modalities = ["tabular", "timeseries", "image"]
    y = np.arange(len(entries))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(entries))))
    left = np.zeros(len(entries))
    for m in modalities:
        vals = np.array([e.get("modality_importance_pct", {}).get(m, 0.0) for e in entries])
        ax.barh(y, vals, left=left, height=0.7, color=MODALITY_COLORS[m], label=m,
                edgecolor="white", linewidth=0.5)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("share of total mean |SHAP| (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Saved per-phenotype modality stacked bars to {}".format(output_path))


def plot_phenotype_image_importance(per_phenotype, output_path, sort_by="auroc", title=None):
    """Single horizontal bar per phenotype = image modality's share of
    total mean |SHAP| only -- a simpler, more direct read than the stacked
    version when the image contribution itself is the whole point.
    """
    entries = sorted(per_phenotype, key=lambda e: _sort_key(e, sort_by),
                      reverse=(sort_by in ("auroc", "auprc")))
    entries = entries[::-1]

    labels = [_phenotype_label(e) for e in entries]
    vals = [e.get("modality_importance_pct", {}).get("image", 0.0) for e in entries]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(entries))))
    bars = ax.barh(labels, vals, color=MODALITY_COLORS["image"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("image modality share of total mean |SHAP| (%)")
    ax.grid(axis="y", visible=False)

    xmax = max(vals) if vals else 1.0
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                "{:.1f}%".format(val), va="center", fontsize=8.5, color="#333333")


    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Saved per-phenotype image importance plot to {}".format(output_path))


def plot_phenotype_performance_delta(per_phenotype_with, per_phenotype_without, output_path,
                                      label_with="with images", label_without="without images",
                                      metric="test_auroc", sort_by="delta", title=None):

    if metric not in ("test_auroc", "test_auprc"):
        raise ValueError("metric must be 'test_auroc' or 'test_auprc'")

    values_with = {e["phenotype"]: e.get(metric) for e in per_phenotype_with}
    values_without = {e["phenotype"]: e.get(metric) for e in per_phenotype_without}
    common = [p for p in values_with if p in values_without
              and values_with[p] == values_with[p] and values_without[p] == values_without[p]]  # NaN-safe

    if not common:
        raise ValueError("No phenotypes with a valid '{}' found in both runs.".format(metric))

    entries = []
    for p in common:
        delta = values_with[p] - values_without[p]
        entries.append((p, values_with[p], values_without[p], delta))

    if sort_by == "delta":
        entries.sort(key=lambda e: e[3], reverse=True)
    else:
        entries.sort(key=lambda e: e[0])
    entries = entries[::-1]  # highest delta ends up at the top of a barh plot

    labels = ["{}  (delta {:+.3f})".format(e[0], e[3]) for e in entries]
    y = np.arange(len(entries))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(entries))))
    ax.barh(y + height / 2, [e[1] for e in entries], height=height * 0.92,
            label=label_with, color=VARIANT_PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.barh(y - height / 2, [e[2] for e in entries], height=height * 0.92,
            label=label_without, color=VARIANT_PALETTE[1], edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    metric_name = "AUROC" if metric == "test_auroc" else "AUPRC"
    ax.set_xlabel(metric_name)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Saved phenotype performance delta plot to {}".format(output_path))


def _phenotype_display_name(name):

    return re.sub(r"\s*\([^)]*\)", "", name).strip()


_DELTA_METRIC_STYLE = {
    "test_auroc": {"color": VARIANT_PALETTE[1], "marker": "o", "s": 30, "ms": 6.5},
    "test_auprc": {"color": "#DC267F", "marker": "D", "s": 26, "ms": 5.5},
}


def plot_shap_vs_performance_delta(per_phenotype_with, per_phenotype_without, output_path,
                                    include_auprc=False, panel_a_caption="a", panel_b_caption="b",
                                    bar_alpha=0.5, min_delta_lim=0.10, show_ci=True,
                                    combine_metrics=False):

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    values_with = {e["phenotype"]: e for e in per_phenotype_with}
    values_without = {e["phenotype"]: e for e in per_phenotype_without}
    metric_keys = ["test_auroc"] + (["test_auprc"] if include_auprc else [])

    rows = []
    ci_methods = set()
    for p in values_with:
        if p not in values_without:
            continue
        img_pct = values_with[p].get("modality_importance_pct", {}).get("image")
        if img_pct is None:
            continue
        deltas, cis = {}, {}
        for mk in metric_keys:
            v_with, v_without = values_with[p].get(mk), values_without[p].get(mk)
            if v_with is None or v_without is None or v_with != v_with or v_without != v_without:
                deltas = None  # NaN-safe: drop phenotype if any shown metric is missing
                break
            ci = _delta_ci(values_with[p], values_without[p], mk) if show_ci else None
            if ci is not None:
                deltas[mk] = ci[0]  # mean of per-seed deltas (== difference of means)
                cis[mk] = (ci[1], ci[2])
                ci_methods.add(ci[3])
            else:
                deltas[mk] = v_with - v_without
        if deltas is not None:
            rows.append((p, img_pct, deltas, cis, values_with[p].get("image_share_ci") if show_ci else None))

    if not rows:
        raise ValueError("No phenotypes with image SHAP share and valid metrics found in both runs.")

    rows.sort(key=lambda r: r[1])  # ascending -> highest SHAP share ends up at the top
    y = np.arange(len(rows))
    labels = [_phenotype_display_name(r[0]) for r in rows]
    shap_share = np.array([r[1] for r in rows])

    from matplotlib.font_manager import FontProperties
    label_fs = FontProperties(size=plt.rcParams["axes.labelsize"]).get_size_in_points()

    panels = [metric_keys] if combine_metrics else [[mk] for mk in metric_keys]
    n_panels = len(panels)
    fig_h = max(4, 0.3 * len(rows) + 1.2)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, fig_h),
                             sharey=True, gridspec_kw={"wspace": 0.15})
    axes = np.atleast_1d(axes)

    for i, (ax, panel_metrics) in enumerate(zip(axes, panels)):
        ax.grid(False)
        for yi in y:
            ax.axhline(yi, color="#d9d9d9", lw=0.5, ls=":", zorder=0)

        # bottom axis: image SHAP share
        ax.barh(y, shap_share, height=0.6, color=VARIANT_PALETTE[0], alpha=bar_alpha, zorder=1)
        share_ci = [r[4] for r in rows]
        if all(c is not None for c in share_ci):
            lo = np.clip(shap_share - np.array([c[0] for c in share_ci]), 0, None)
            hi = np.clip(np.array([c[1] for c in share_ci]) - shap_share, 0, None)
            ax.errorbar(shap_share, y, xerr=[lo, hi], fmt="none", ecolor=VARIANT_PALETTE[0],
                        elinewidth=1.0, capsize=2.0, zorder=1.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("image modality share of total mean |SHAP| (%)")
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.tick_params(axis="y", length=0, labelsize=label_fs)
        ax.tick_params(axis="x", labelsize=label_fs)
        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labels)

        per_metric = []
        extent = []
        for mk in panel_metrics:
            deltas = np.array([r[2][mk] for r in rows])
            ci_arr = [r[3].get(mk) for r in rows]
            has_ci = all(c is not None for c in ci_arr)
            extent.append(np.abs(deltas))
            if has_ci:
                extent.append(np.abs([c[0] for c in ci_arr]))
                extent.append(np.abs([c[1] for c in ci_arr]))
            per_metric.append((mk, deltas, ci_arr, has_ci))
        lim = max(min_delta_lim, np.ceil(np.max(np.concatenate(extent)) * 1.05 / 0.05) * 0.05)

        top = ax.twiny()
        top.grid(False)
        top.set_xlim(-lim, lim)
        top.set_xticks(np.round(np.linspace(-lim, lim, 5), 3))
        top.tick_params(axis="x", labelsize=label_fs)
        top.spines["top"].set_visible(True)
        top.spines["right"].set_visible(False)
        metric_names = " / ".join("\u0394 " + ("AUROC" if mk == "test_auroc" else "AUPRC")
                                  for mk in panel_metrics)
        top.set_xlabel("{} (structured+CXR \u2212 structured)".format(metric_names), labelpad=6)
        top.axvline(0, color="#555555", lw=0.8, ls="--", zorder=2)

        offsets = [0.0] if len(per_metric) == 1 else np.linspace(0.15, -0.15, len(per_metric))
        for (mk, deltas, ci_arr, has_ci), off in zip(per_metric, offsets):
            style = _DELTA_METRIC_STYLE[mk]
            if has_ci:
                lo = np.clip(deltas - np.array([c[0] for c in ci_arr]), 0, None)
                hi = np.clip(np.array([c[1] for c in ci_arr]) - deltas, 0, None)
                top.errorbar(deltas, y + off, xerr=[lo, hi], fmt="none", ecolor=style["color"],
                             elinewidth=1.0, capsize=2.0 if len(per_metric) > 1 else 2.5, zorder=2.5)
            top.scatter(deltas, y + off, s=style["s"], color=style["color"], marker=style["marker"],
                        edgecolor="white", linewidth=0.9, zorder=3)

        if n_panels > 1:
            caption = panel_a_caption if i == 0 else panel_b_caption
            ax.annotate("({})".format(caption), xy=(0, 1), xycoords="axes fraction",
                        xytext=(0, 36), textcoords="offset points",
                        fontsize=label_fs, fontweight="normal", va="bottom", ha="left")

    handles = [Patch(facecolor=VARIANT_PALETTE[0], alpha=bar_alpha,
                     label="image modality share of total mean |SHAP| (bottom axis)"),
               ] + [Line2D([], [], marker=_DELTA_METRIC_STYLE[mk]["marker"], ls="",
                           color=_DELTA_METRIC_STYLE[mk]["color"], markeredgecolor="white",
                           markersize=_DELTA_METRIC_STYLE[mk]["ms"],
                           label="Δ {} (top axis)".format("AUROC" if mk == "test_auroc" else "AUPRC"))
                    for mk in metric_keys]
    y0 = axes[0].get_position().y0 - 0.55 / fig_h
    x_mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x_mid, y0),
               ncol=len(handles) if n_panels > 1 else 1, frameon=False, fontsize=label_fs)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    if ci_methods:
        print("  delta CIs: {} t-interval over seeds (training stochasticity only)".format(
            " / ".join(sorted(ci_methods))))
    elif show_ci:
        print("  no per-seed results found -> plotted without CIs")
    print("Saved SHAP-vs-performance-delta plot to {}".format(output_path))


def plot_modality_waterfall(variant_data, output_path, title=None):

    modality_pct = variant_data["modality_importance"].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    cumulative = 0.0
    colors = plt.cm.Set2(np.linspace(0, 1, len(modality_pct)))
    for i, (modality, pct) in enumerate(modality_pct.items()):
        ax.bar(modality, pct, bottom=0, color=colors[i], edgecolor="black")
        ax.text(i, pct + 1, "{:.1f}%".format(pct), ha="center", va="bottom", fontsize=9)
        cumulative += pct

    ax.set_ylabel("Share of total |SHAP| (%)")
    ax.set_ylim(0, max(modality_pct.max() * 1.2, 10))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_variant_comparison(variant_data_dict, output_dir):
 
    os.makedirs(output_dir, exist_ok=True)

    all_modalities = sorted(set(
        m for vd in variant_data_dict.values() for m in vd["modality_importance"].index))
    variant_labels = list(variant_data_dict.keys())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(variant_labels)
    x = np.arange(len(all_modalities))
    for i, label in enumerate(variant_labels):
        pcts = [variant_data_dict[label]["modality_importance"].get(m, 0.0) for m in all_modalities]
        ax.bar(x + i * width, pcts, width=width, label=label)
    ax.set_xticks(x + width * (len(variant_labels) - 1) / 2)
    ax.set_xticklabels(all_modalities)
    ax.set_ylabel("Share of total |SHAP| (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modality_importance_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(variant_labels))
    aurocs = [variant_data_dict[l]["metrics"]["test_auroc"] for l in variant_labels]
    auprcs = [variant_data_dict[l]["metrics"]["test_auprc"] for l in variant_labels]
    ax.bar(x - 0.2, aurocs, width=0.4, label="AUROC")
    ax.bar(x + 0.2, auprcs, width=0.4, label="AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved comparison plots to {}".format(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Generate HAIM-style plots from train_mortality.py outputs.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Plots for one variant.")
    p_single.add_argument("--results_dir", required=True)
    p_single.add_argument("--variant_label", required=True)
    p_single.add_argument("--output_dir", default=None,
                           help="Defaults to --results_dir if not given.")
    p_single.add_argument("--format", choices=["png", "svg", "pdf"], default="pdf",
                           help="Output file format (default: pdf).")

    p_compare = sub.add_parser("compare", help="Side-by-side comparison of multiple variants.")
    p_compare.add_argument("--results_dir", action="append", required=True,
                            help="Repeat for each variant, paired in order with --variant_label.")
    p_compare.add_argument("--variant_label", action="append", required=True)
    p_compare.add_argument("--output_dir", required=True)
    p_compare.add_argument("--format", choices=["png", "svg", "pdf"], default="pdf")
    p_compare.add_argument("--top_n", type=int, default=10,
                            help="Union of each variant's top-N features to compare (default: 10).")
    p_compare.add_argument("--comparison_title", type=str, default="Feature importance across variants",
                            help="Title for feature_importance_comparison plot.")
    p_compare.add_argument("--diff_title", type=str, default="Biggest feature importance differences",
                            help="Title for feature_importance_diff plot.")
    p_compare.add_argument("--highlight_features", action="append", default=None,
                            help="Feature name to mark in the spread histogram (repeatable), e.g. tab_Height.")
    p_compare.add_argument("--histogram_title", type=str,
                            default="Distribution of feature importance spread across variants",
                            help="Title for feature_importance_spread_histogram plot.")

    p_pheno = sub.add_parser("phenotype", help="Per-phenotype modality overview for one train_phenotyping.py run.")
    p_pheno.add_argument("--results_dir", required=True)
    p_pheno.add_argument("--variant_label", required=True,
                          help="The --variant_label used in the train_phenotyping.py")
    p_pheno.add_argument("--output_dir", default=None, help="Defaults to --results_dir if not given.")
    p_pheno.add_argument("--format", choices=["png", "svg", "pdf"], default="pdf")
    p_pheno.add_argument("--sort_by", choices=["auroc", "auprc", "name"], default="auroc")
    p_pheno.add_argument("--stacked_title", type=str, default="Modality contribution by phenotype")
    p_pheno.add_argument("--image_title", type=str, default="Image modality importance by phenotype")

    p_ablation = sub.add_parser("phenotype_ablation",
                                 help="Compare AUROC/AUPRC per phenotype between image and no-image run")
    p_ablation.add_argument("--results_dir_with", required=True, help="Results dir of the WITH-images run.")
    p_ablation.add_argument("--variant_label_with", required=True)
    p_ablation.add_argument("--results_dir_without", required=True, help="Results dir of the --no_image_features run.")
    p_ablation.add_argument("--variant_label_without", required=True)
    p_ablation.add_argument("--output_dir", required=True)
    p_ablation.add_argument("--format", choices=["png", "svg", "pdf"], default="pdf")
    p_ablation.add_argument("--metric", choices=["auroc", "auprc"], default="auroc")
    p_ablation.add_argument("--sort_by", choices=["delta", "name"], default="delta")
    p_ablation.add_argument("--title", type=str, default=None,
                             help="Defaults to 'Phenotype {AUROC|AUPRC}: with vs. without image features'.")

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

    p_svd = sub.add_parser("shap_vs_delta",
                            help="One row per phenotype: image SHAP share as bars (bottom axis) overlaid with AUROC/AUPRC")
    p_svd.add_argument("--results_dir_with", required=True, help="Results dir of the WITH-images run.")
    p_svd.add_argument("--variant_label_with", required=True)
    p_svd.add_argument("--results_dir_without", required=True, help="Results dir of the --no_image_features run.")
    p_svd.add_argument("--variant_label_without", required=True)
    p_svd.add_argument("--output_path", required=True, help="Full output file path (extension sets format).")
    p_svd.add_argument("--panel_a_caption", type=str, default="a")
    p_svd.add_argument("--panel_b_caption", type=str, default="b")
    p_svd.add_argument("--include_auprc", action="store_true", default=False,
                        help="Add a second panel for delta AUPRC (default: AUROC only).")
    p_svd.add_argument("--bar_alpha", type=float, default=0.5,
                        help="Opacity of the image-SHAP-share bars (default 0.5).")
    p_svd.add_argument("--min_delta_lim", type=float, default=0.10,
                        help="Minimum half-width of the symmetric delta axis (default 0.10).")
    p_svd.add_argument("--no_ci", dest="show_ci", action="store_false", default=True,
                        help="Do not draw across-seed CI whiskers (default: drawn when per-seed files exist).")
    p_svd.add_argument("--combine_metrics", action="store_true", default=False,
                        help="With --include_auprc: draw delta AUROC and delta AUPRC as offset points "
                             "in ONE panel instead of two side-by-side panels.")

    args = parser.parse_args()

    if args.mode == "single":
        output_dir = args.output_dir or args.results_dir
        os.makedirs(output_dir, exist_ok=True)
        vd = _load_variant(args.results_dir, args.variant_label)

        plot_shap_summary(vd, os.path.join(output_dir, "{}_shap_summary.{}".format(args.variant_label, args.format)),
                           title="SHAP summary -- {}".format(args.variant_label))
        plot_top_features_bar(vd, os.path.join(output_dir, "{}_top_features.{}".format(args.variant_label, args.format)),
                               title="Top features -- {}".format(args.variant_label))
        print("Saved 2 plots to {}".format(output_dir))

    elif args.mode == "compare":
        if len(args.results_dir) != len(args.variant_label):
            raise ValueError("--results_dir and --variant_label must be given the same number of times.")
        os.makedirs(args.output_dir, exist_ok=True)
        variant_data_dict = {}
        for rdir, label in zip(args.results_dir, args.variant_label):
            variant_data_dict[label] = _load_variant(rdir, label)
        plot_feature_importance_comparison(
            variant_data_dict,
            os.path.join(args.output_dir, "feature_importance_comparison.{}".format(args.format)),
            top_n=args.top_n,
            title=args.comparison_title)
        plot_feature_importance_diff(
            variant_data_dict,
            os.path.join(args.output_dir, "feature_importance_diff.{}".format(args.format)),
            top_n=args.top_n,
            title=args.diff_title)
        plot_feature_importance_spread_histogram(
            variant_data_dict,
            os.path.join(args.output_dir, "feature_importance_spread_histogram.{}".format(args.format)),
            highlight_features=args.highlight_features,
            title=args.histogram_title)

    elif args.mode == "phenotype":
        output_dir = args.output_dir or args.results_dir
        os.makedirs(output_dir, exist_ok=True)
        per_phenotype = load_phenotype_run(args.results_dir, args.variant_label)
        plot_phenotype_modality_stacked(
            per_phenotype,
            os.path.join(output_dir, "{}_phenotype_modality_stacked.{}".format(args.variant_label, args.format)),
            sort_by=args.sort_by, title=args.stacked_title)
        plot_phenotype_image_importance(
            per_phenotype,
            os.path.join(output_dir, "{}_phenotype_image_importance.{}".format(args.variant_label, args.format)),
            sort_by=args.sort_by, title=args.image_title)

    elif args.mode == "phenotype_ablation":
        os.makedirs(args.output_dir, exist_ok=True)
        per_phenotype_with = load_phenotype_run(args.results_dir_with, args.variant_label_with)
        per_phenotype_without = load_phenotype_run(args.results_dir_without, args.variant_label_without)
        metric = "test_auroc" if args.metric == "auroc" else "test_auprc"
        title = args.title or "Phenotype {}: with vs. without image features".format(args.metric.upper())
        plot_phenotype_performance_delta(
            per_phenotype_with, per_phenotype_without,
            os.path.join(args.output_dir, "phenotype_{}_ablation.{}".format(args.metric, args.format)),
            label_with=args.variant_label_with, label_without=args.variant_label_without,
            metric=metric, sort_by=args.sort_by, title=title)

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
        per_phenotype_with = load_phenotype_run(args.results_dir_with, args.variant_label_with)
        per_phenotype_without = load_phenotype_run(args.results_dir_without, args.variant_label_without)
        plot_shap_vs_performance_delta(
            per_phenotype_with, per_phenotype_without, args.output_path,
            include_auprc=args.include_auprc,
            panel_a_caption=args.panel_a_caption, panel_b_caption=args.panel_b_caption,
            bar_alpha=args.bar_alpha, min_delta_lim=args.min_delta_lim, show_ci=args.show_ci,
            combine_metrics=args.combine_metrics)


if __name__ == "__main__":
    main()
