import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bootstrap_evaluation import RankedScores, paired_bootstrap, patient_weights, point_metrics


def test_metrics_match_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 1)          # many ties
    m = point_metrics(y, p)
    assert np.isclose(m["auroc"], roc_auc_score(y, p))
    assert np.isclose(m["auprc"], average_precision_score(y, p))
    assert np.isclose(m["brier"], brier_score_loss(y, p))


def test_integer_weights_equal_replication():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    w = rng.integers(0, 4, 300).astype(float)
    rep = np.repeat(np.arange(300), w.astype(int))
    au, ap, br = RankedScores(p, y).metrics(w)
    assert np.isclose(au, roc_auc_score(y[rep], p[rep]))
    assert np.isclose(ap, average_precision_score(y[rep], p[rep]))
    assert np.isclose(br, brier_score_loss(y[rep], p[rep]))


def test_patient_weights_keep_admissions_of_a_patient_together():
    groups = np.array([5, 5, 5, 9, 9, 1])
    w = patient_weights(groups, np.random.default_rng(3))
    for g in np.unique(groups):
        assert len(set(w[groups == g])) == 1
    assert w.sum() == np.sum([w[groups == g][0] * (groups == g).sum() for g in np.unique(groups)])


def test_one_class_replicates_are_skipped():
    wide = pd.DataFrame({"subject_id": [1, 2, 3], "label": [1, 0, 0], "A": [0.9, 0.1, 0.2],
                         "B": [0.8, 0.3, 0.1]})
    t = paired_bootstrap(wide, ["A", "B"], "A", 200, 0)
    assert (t.n_valid_replicates < 200).all() and (t.n_valid_replicates > 0).all()
    assert np.isclose(t.loc[0, "auroc"], 1.0)


def test_identical_conditions_have_zero_difference():
    rng = np.random.default_rng(2)
    wide = pd.DataFrame({"subject_id": rng.integers(0, 100, 400), "label": rng.integers(0, 2, 400)})
    wide["A"] = rng.random(400)
    wide["B"] = wide["A"]
    t = paired_bootstrap(wide, ["A", "B"], "A", 100, 0)
    assert t.loc[1, "delta_auroc_ci_low"] == 0 and t.loc[1, "delta_auroc_ci_high"] == 0


def test_point_delta_uses_reference_metrics():
    rng = np.random.default_rng(4)
    wide = pd.DataFrame({"subject_id": np.arange(300), "label": rng.integers(0, 2, 300)})
    wide["A"], wide["B"] = rng.random(300), rng.random(300)
    t = paired_bootstrap(wide, ["A", "B"], "A", 50, 0)
    assert np.isclose(t.loc[1, "delta_auroc_vs_ref"], t.loc[1, "auroc"] - t.loc[0, "auroc"])
    assert t.loc[0, "delta_auroc_vs_ref"] == 0
