"""Phase 5: frozen CORe mortality model on all text conditions (inference only).

Text-budget policy (max_length 512 = [CLS] + 510 + [SEP]):
  primary   T0/T1/T2: first ``base_budget`` (382) admission-text tokens
            T3/T4:    the identical first 382 T1 tokens + up to ``injection_budget``
                      (128) tokens of injected later-section text
  -> T1, T3 and T4 share the same admission prefix; injected text cannot displace
     admission evidence.
  sensitivity T0/T1/T2 with ``full_budget`` (510) admission tokens.

Each text is tokenised per section ("LABEL: content") and the ids are
concatenated, which makes section reach exact. Predictions (restricted) go to
private_dir; truncation statistics (aggregate) go to the report directory.
"""
from __future__ import annotations

import hashlib
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ADMISSION_SECTIONS, INJECTION_LABELS, load_config, results_dir, write_json

PART_RE = re.compile(r"\n\n(?=(?:" + "|".join(
    re.escape(l) for l in [l for _, l in ADMISSION_SECTIONS] + list(INJECTION_LABELS.values())) + r"): )")


def tokenize_parts(tok, texts: list[str], chunk: int = 2000) -> tuple[list[np.ndarray], list[list[int]]]:
    """int32 token ids per text (concatenated per-part ids) and per-part token lengths."""
    ids, lens = [], []
    for s in range(0, len(texts), chunk):
        parts = [PART_RE.split(t) if t else [] for t in texts[s:s + chunk]]
        flat = [p for ps in parts for p in ps]
        enc = tok(flat, add_special_tokens=False)["input_ids"] if flat else []
        k = 0
        for ps in parts:
            c = enc[k:k + len(ps)]
            k += len(ps)
            ids.append(np.fromiter((i for x in c for i in x), dtype=np.int32))
            lens.append([len(x) for x in c])
    return ids, lens


class Condition:
    """[CLS] + base[:budget] + inj[:inj_budget] + [SEP], assembled per batch (never materialised)."""

    def __init__(self, tok, base, budget, inj=None, inj_budget=0):
        self.cls, self.sep = tok.cls_token_id, tok.sep_token_id
        self.base, self.budget, self.inj, self.inj_budget = base, budget, inj, inj_budget

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i) -> np.ndarray:
        seq = [self.base[i][:self.budget]]
        if self.inj is not None:
            seq.append(self.inj[i][:self.inj_budget])
        return np.concatenate([[self.cls], *seq, [self.sep]]).astype(np.int64)

    def lengths(self) -> np.ndarray:
        n = np.array([min(len(b), self.budget) for b in self.base]) + 2
        if self.inj is not None:
            n += np.array([min(len(x), self.inj_budget) for x in self.inj])
        return n


PAD_BUCKET = 64
MPS_MEMORY_LIMIT_GB = 8.0


@torch.no_grad()
def predict(model, cond, device, batch_size: int, mort_idx: int, max_len: int = 512) -> np.ndarray:
    lengths = cond.lengths() if isinstance(cond, Condition) else np.array([len(x) for x in cond])
    order = np.argsort(lengths, kind="mergesort")
    probs = np.empty(len(lengths), dtype=np.float64)
    for b, s in enumerate(range(0, len(order), batch_size)):
        idx = order[s:s + batch_size]
        # Pad to 64-token buckets: MPS caches a graph per input shape, and per-batch
        # dynamic lengths made that cache grow to tens of GB. Masked padding does not
        # change the [CLS] output.
        L = int(min(max_len, -(-int(lengths[idx].max()) // PAD_BUCKET) * PAD_BUCKET))
        ids = torch.zeros((len(idx), L), dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            x = torch.from_numpy(np.asarray(cond[i], dtype=np.int64))
            ids[r, :len(x)] = x
            att[r, :len(x)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
        probs[idx] = torch.softmax(logits, dim=-1)[:, mort_idx].cpu().numpy()
        if device.type == "mps" and b % 50 == 0:
            torch.mps.empty_cache()
            gb = torch.mps.driver_allocated_memory() / 1e9
            if b % 500 == 0:
                print(f"  batch {b}: mps driver memory {gb:.1f} GB", flush=True)
            assert gb < MPS_MEMORY_LIMIT_GB, f"MPS memory {gb:.1f} GB exceeds limit"
    return probs


def section_reach(lens: list[list[int]], budget: int, labels: list[str], texts_parts) -> dict:
    """Share of admissions whose non-empty section k is fully / partly / not seen within ``budget``."""
    stats = {lab: {"full": 0, "partial": 0, "none": 0, "present": 0} for lab in labels}
    for ls, names in zip(lens, texts_parts):
        pos = 0
        for (lab, nonempty), n in zip(names, ls):
            if lab not in stats or not nonempty:
                pos += n
                continue
            stats[lab]["present"] += 1
            key = "full" if pos + n <= budget else ("partial" if pos < budget else "none")
            stats[lab][key] += 1
            pos += n
    return {lab: {k: (round(100 * v / s["present"], 2) if k != "present" else v) for k, v in s.items()}
            for lab, s in stats.items() if s["present"]}


def part_labels(texts):
    """(label, has_content) per rendered part."""
    return [[(p.split(":", 1)[0], bool(p.split(":", 1)[1].strip())) for p in PART_RE.split(t)] if t else []
            for t in texts]


def main():
    cfg = load_config()
    cc = cfg["core"]
    priv = cfg["paths"]["private_dir"]
    out_path = priv / "predictions_core.parquet"

    weights = hf_hub_download(cc["model_id"], "pytorch_model.bin", revision=cc["revision"])
    sha = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
    assert sha == cc["weights_sha256"], f"checkpoint hash mismatch: {sha}"
    tok = AutoTokenizer.from_pretrained(cc["model_id"], revision=cc["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(cc["model_id"], revision=cc["revision"])
    device = torch.device(cc["device"] if (cc["device"] != "mps" or torch.backends.mps.is_available()) else "cpu")
    model.eval().to(device)
    torch.manual_seed(cfg["seed"])
    assert model.config.num_labels == 2 and model.config.max_position_embeddings == cc["max_length"]

    # Sanity check of the class mapping with the model-card example and its negation.
    probe = ["CHIEF COMPLAINT: Headaches\n\nPRESENT ILLNESS: 58yo man w/ hx of hypertension, AFib on coumadin "
             "presented to ED with the worst headache of his life.",
             "CHIEF COMPLAINT: cardiac arrest\n\nPRESENT ILLNESS: 85yo found pulseless, prolonged CPR, "
             "fixed dilated pupils, on three pressors, family discussing comfort measures only."]
    p_probe = predict(model, [[tok.cls_token_id] + tok(t, add_special_tokens=False)["input_ids"] +
                              [tok.sep_token_id] for t in probe], device, 2, cc["mortality_index"])

    v = pd.read_parquet(priv / "variants.parquet",
                        columns=["study_id", "split", "in_analysis", "t0_text", "t1_text", "t2_text",
                                 "inj_text", "t4_donor"])
    inj_by_id = dict(zip(v.study_id, v.inj_text))
    v = v[(v.split == cc["eval_split"]) & v.in_analysis].reset_index(drop=True)
    v["t4_inj"] = v.t4_donor.map(inj_by_id)
    extra = []   # section-specific positive controls and the outcome-matched control (build_injection_parts.py)
    if (priv / "injection_parts.parquet").exists():
        from train_tfidf_baseline import PART_CONDITIONS, one_section
        v = v.merge(pd.read_parquet(priv / "injection_parts.parquet"), on="study_id", how="left")
        v["T4M_INJ"] = v.t4m_donor.map(inj_by_id)
        extra = [("T4M_INJ", "T4M_INJ", "T4M_OUTCOMEMATCHED")]
        for cond, (col, name) in PART_CONDITIONS.items():
            v[cond + "_INJ"] = one_section(v[col], name)
            extra.append((cond + "_INJ", cond + "_INJ", cond))

    t1b = []      # improved robust parser and its screened version (build_t1b.py)
    if (priv / "variants_t1b.parquet").exists():
        v = v.merge(pd.read_parquet(priv / "variants_t1b.parquet"), on="study_id", how="left")
        t1b = [("T1B", "t1b_text"), ("T2B", "t2b_text")]
    t = time.time()
    ids, lens = {}, {}
    for key, col in [("T0", "t0_text"), ("T1", "t1_text"), ("T2", "t2_text"), ("INJ", "inj_text"),
                     ("INJ4", "t4_inj")] + [(k, c) for k, c, _ in extra] + t1b:
        ids[key], lens[key] = tokenize_parts(tok, v[col].fillna("").tolist())
    print(f"tokenised {len(v)} admissions in {time.time() - t:.0f}s", flush=True)
    del inj_by_id
    part_names = {"T1": part_labels(v.t1_text), "INJ": part_labels(v.inj_text.fillna(""))}
    v = v[["study_id"]]

    B, R, F = cc["base_budget"], cc["injection_budget"], cc["full_budget"]
    conditions = {
        "T0_COPIV": Condition(tok, ids["T0"], B),
        "T1_ROBUST": Condition(tok, ids["T1"], B),
        "T2_SCREENED": Condition(tok, ids["T2"], B),
        "T3_RETROSPECTIVE": Condition(tok, ids["T1"], B, ids["INJ"], R),
        "T4_OTHERPATIENT": Condition(tok, ids["T1"], B, ids["INJ4"], R),
        "T0_COPIV@510": Condition(tok, ids["T0"], F),
        "T1_ROBUST@510": Condition(tok, ids["T1"], F),
        "T2_SCREENED@510": Condition(tok, ids["T2"], F),
        **{cond: Condition(tok, ids["T1"], B, ids[key], R) for key, _, cond in extra},
        **({"T1B_ROBUST": Condition(tok, ids["T1B"], B), "T2B_SCREENED": Condition(tok, ids["T2B"], B),
            "T1B_ROBUST@510": Condition(tok, ids["T1B"], F), "T2B_SCREENED@510": Condition(tok, ids["T2B"], F)}
           if t1b else {}),
    }
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    for name, inputs in conditions.items():
        if len(done) and name in set(done.condition):
            continue
        t = time.time()
        p = predict(model, inputs, device, cc["batch_size"], cc["mortality_index"])
        part = pd.DataFrame({"study_id": v.study_id, "condition": name, "p_mortality": p,
                             "n_input_tokens": inputs.lengths()})
        done = pd.concat([done, part], ignore_index=True)
        done.to_parquet(out_path, index=False)
        print(f"{name}: {len(inputs)} in {time.time() - t:.0f}s", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()

    # Aggregate truncation / reach statistics (public).
    n_t1 = np.array([len(x) for x in ids["T1"]])
    n_t2 = np.array([len(x) for x in ids["T2"]])
    n_inj = np.array([len(x) for x in ids["INJ"]])
    stats = {
        "n_admissions": int(len(v)),
        "budgets": {"base": B, "injection": R, "full": F, "max_length": cc["max_length"]},
        "available_tokens_median": {k: float(np.median([len(x) for x in ids[k]])) for k in ["T0", "T1", "T2", "INJ"]},
        "pct_truncated_at_base": {k: round(100 * float(np.mean([len(x) > B for x in ids[k]])), 2)
                                  for k in ["T0", "T1", "T2"]},
        "pct_truncated_at_full": {k: round(100 * float(np.mean([len(x) > F for x in ids[k]])), 2)
                                  for k in ["T0", "T1", "T2"]},
        "pct_of_admission_tokens_dropped_at_base_median": {
            k: round(100 * float(np.median([max(0, len(x) - B) / max(1, len(x)) for x in ids[k]])), 2)
            for k in ["T0", "T1", "T2"]},
        "t2_tokens_removed": {"admissions_with_removal": int((n_t1 > n_t2).sum()),
                              "median_removed_among_those": float(np.median((n_t1 - n_t2)[n_t1 > n_t2]))
                              if (n_t1 > n_t2).any() else 0.0,
                              "admissions_where_input_window_changes": int(
                                  sum(not np.array_equal(a[:B], b[:B]) for a, b in zip(ids["T1"], ids["T2"])))},
        "t3_injection": {"pct_with_injected_tokens": round(100 * float(np.mean(n_inj > 0)), 2),
                         "injected_tokens_available_median": float(np.median(n_inj)),
                         "injected_tokens_in_input_median": float(np.median(np.minimum(n_inj, R))),
                         "pct_injection_truncated": round(100 * float(np.mean(n_inj > R)), 2),
                         "section_reach_within_injection_budget": section_reach(
                             lens["INJ"], R, list(INJECTION_LABELS.values()), part_names["INJ"])},
        "t1_section_reach_at_base": section_reach(lens["T1"], B, [l for _, l in ADMISSION_SECTIONS],
                                                  part_names["T1"]),
        "t1_section_reach_at_full": section_reach(lens["T1"], F, [l for _, l in ADMISSION_SECTIONS],
                                                  part_names["T1"]),
    }
    env = {"model_id": cc["model_id"], "revision": cc["revision"], "weights_sha256": sha,
           "tokenizer_class": type(tok).__name__, "tokenizer_do_lower_case": getattr(tok, "do_lower_case", None),
           "torch": torch.__version__, "transformers": transformers.__version__, "device": str(device),
           "python": platform.python_version(), "num_labels": model.config.num_labels,
           "max_position_embeddings": model.config.max_position_embeddings,
           "probe_model_card_example_p_mortality": round(float(p_probe[0]), 4),
           "probe_high_risk_synthetic_p_mortality": round(float(p_probe[1]), 4)}
    write_json({"environment": env, "truncation": stats}, results_dir(cfg) / "core_inference_stats.json")
    print(env)


if __name__ == "__main__":
    main()
