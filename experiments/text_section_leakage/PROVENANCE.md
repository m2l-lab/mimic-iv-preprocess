# Import provenance

Source snapshot: `medllm-multiagents/experiments/text_section_leakage`, imported
on 2026-11-04. This repository includes the automated hospital-admission cohort,
text extraction and screening, mortality baselines, paired evaluation,
screening summaries, and synthetic tests.

Human-review workflows and their protocols, report templates, validation
records, tests, and manuscript-specific figures are excluded. The external
parser baseline, S-Proto extension, and cluster fine-tuning are also excluded.
No private inputs, row-level outputs, model weights, or source-study findings
are bundled.

Integration changes include configurable local paths, configuration reads
without filesystem writes during variant-module import, and an independent
`summarize_screening.py` entry point containing the automated rule-hit summary.
The two shared patterns needed by the improved parser live in
`section_patterns.py`; extraction and screening behavior are retained and
covered by synthetic regression tests. T0 remains the attributed historical
COP-IV baseline. Cohort selection, labels, patient splits, token budgets, and
model pins retain the source experiment's settings.
