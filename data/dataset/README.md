# Grammarly & QuillBot Grade GEC Dataset

A high-precision, multi-domain Grammatical Error Correction (GEC), Typo Repair, and Fluency Refinement dataset engineered for training and evaluating next-generation edge and server NLP models.

---

## Dataset Summary

* **Total Rows**: 25,000 paired sentence instances
* **File Size**: ~8.91 MB
* **Format**: Standard RFC 4180 CSV (`UTF-8`)
* **Split Partitions**:
  * **Master Dataset**: [`data/dataset/grammarly_quillbot_dataset.csv`](grammarly_quillbot_dataset.csv) (25,000 rows)
  * **Train Split (90%)**: [`data/dataset/train.csv`](train.csv) (22,500 rows)
  * **Validation Split (10%)**: [`data/dataset/val.csv`](val.csv) (2,500 rows)
* **Token-Aligned GEC JSONL**:
  * [`data/processed/gec/grammarly_quillbot_train.jsonl`](../processed/gec/grammarly_quillbot_train.jsonl) (22,500 rows)
  * [`data/processed/gec/grammarly_quillbot_val.jsonl`](../processed/gec/grammarly_quillbot_val.jsonl) (2,500 rows)
  * [`data/processed/gec/grammarly_quillbot_maps.json`](../processed/gec/grammarly_quillbot_maps.json) (Dynamic edit confusion maps)

---

## Column Schema

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Unique identifier | `GQ_000001` |
| `input_text` | `string` | Noisy, unpolished, or clean source text | `"We verified that their planning to deploy."` |
| `target_text` | `string` | Polished, grammatically accurate target text | `"We verified that they're planning to deploy."` |
| `error_type` | `string` | High-level linguistic error family | `grammar`, `spelling`, `syntax`, `fluency`, `punctuation`, `clean` |
| `error_category` | `string` | Fine-grained error taxonomy | `homophone_confusion`, `subject_verb_agreement`, `spelling_typo`, etc. |
| `error_count` | `integer` | Number of errors present (0 for clean rows) | `1` |
| `correction_explanation` | `string` | Pedagogical Grammarly-style rationale | `"Use 'they\'re' as a contraction for 'they are'."` |
| `domain` | `string` | Contextual writing domain | `technical_code`, `academic_essay`, `professional_email`, etc. |
| `complexity_level` | `string` | Linguistic difficulty tier | `elementary`, `intermediate`, `advanced`, `professional` |

---

## Error Taxonomy & Distribution

```text
=== ERROR TYPES ===
  grammar             :  8,992 ( 36.0%)  [Homophones, Verb Tenses, Countability]
  clean               :  5,860 ( 23.4%)  [Pristine Negative Anchors (0 edits)]
  spelling            :  3,230 ( 12.9%)  [Cognitive & Transposition Typos]
  fluency             :  3,226 ( 12.9%)  [QuillBot Wordiness & Collocations]
  syntax              :  2,841 ( 11.4%)  [Subject-Verb Agreement & Inversion]
  punctuation         :    851 (  3.4%)  [Mechanics, Splices, Capitalization]

=== CATEGORY BREAKDOWN ===
  homophone_confusion       :  6,217 ( 24.9%)
  clean_preservation        :  5,860 ( 23.4%)
  spelling_typo             :  3,230 ( 12.9%)
  subject_verb_agreement    :  2,841 ( 11.4%)
  preposition_collocation   :  1,663 (  6.7%)
  wordiness_redundancy      :  1,563 (  6.3%)
  verb_tense_aspect         :  1,447 (  5.8%)
  countable_uncountable     :  1,328 (  5.3%)
  punctuation_mechanics     :    851 (  3.4%)
```

---

## Contextual Domain Coverage

* **Technical & Software Engineering (42.4%)**: Distributed systems, microservices, cloud deployments, CUDA kernels, concurrency, API endpoints, pull requests.
* **Academic & Scientific (23.3%)**: Literature reviews, double-blind trials, empirical hypotheses, statistical regression, astrophysical models.
* **Professional Workplace & Business Email (18.9%)**: Executive status memos, stakeholder briefs, quarterly reviews, HR compliance, contracts.
* **Corporate Finance & Business Strategy (8.4%)**: Capital allocation, venture rounds, ROI analysis, market expansions, audits.
* **Casual Messaging & Everyday Chat (4.3%)**: Conversational notes, meetups, reminders.
* **Narrative & Creative Prose (2.7%)**: Descriptive prose, sensory phrasing, storytelling.

---

## Training TypoFormer on this Dataset

### 1. Re-generate or Scale Dataset
To re-generate or synthesize additional rows:
```bash
python scripts/generate_grammarly_dataset.py
```

### 2. Convert CSV to TypoFormer GEC JSONL
To re-index and compute token-level edit tags:
```bash
python scripts/convert_csv_to_gec.py \
  --csv data/dataset/train.csv \
  --tokenizer data/tokenizer/tokenizer.json \
  --out data/processed/gec/grammarly_quillbot_train.jsonl \
  --maps data/processed/gec/grammarly_quillbot_maps.json
```

### 3. Fine-Tune TypoFormer
Launch fine-tuning using the dedicated recipe:
```bash
python -m typotaker train --config configs/train/gec_stage2_grammarly.yaml
```
