# Stage 1 Quality Gate

Date started: 2026-10-19

This report records the Stage 0 versus Stage 1 comparison, Stage 1 GEC training, evaluation, data repairs, retraining, and final findings. Commands are recorded so the results can be reproduced.

## Objective

Determine whether the 39.1M-parameter Stage 1 model is a meaningful improvement over the 25.4M-parameter Stage 0 model, then improve and validate grammar correction before export or desktop integration.

## Checkpoints under evaluation

| Model | Path | Status |
|---|---|---|
| Stage 0 LM | `checkpoints/lm_stage0` | Existing baseline |
| Stage 1 LM | `checkpoints/lm_stage1` | Existing checkpoint; latest local run stopped at step 16,684/20,000 |
| Stage 1 GEC | `checkpoints/gec_stage1` | To be trained |

## Workflow

- [x] 1. Fixed Stage 0 versus Stage 1 autocomplete comparison
- [ ] 2. Train Stage 1 GEC from the Stage 1 LM checkpoint
- [ ] 3. Evaluate Stage 1 GEC
- [ ] 4. Repair synthetic GEC data, alignment, decoding, and resume behavior
- [ ] 5. Retrain and reevaluate Stage 1 GEC
- [ ] 6. Decide whether export and desktop integration are justified

## Step 1: Fixed autocomplete comparison

Command:

```powershell
python .\\scripts\\compare_lm.py --stage0 checkpoints/lm_stage0 --stage1 checkpoints/lm_stage1
```

Terminal output:

```text
MODEL stage0 RUN checkpoints/lm_stage0
PROMPT 'The server listens'
  0.2347  ' to'
  0.1421  ' for'
  0.1119  ' as'
  0.0786  ' the'
  0.0397  ' on'
PROMPT 'The user should be able'
  0.9900  ' to'
  0.0013  ' and'
  0.0012  ','
  0.0008  ' for'
  0.0004  ' not'
PROMPT 'Please check the configuration'
  0.1019  ' of'
  0.0883  ' file'
  0.0701  ' for'
  0.0374  ' and'
  0.0362  ' name'
PROMPT 'I would like to'
  0.0825  ' see'
  0.0563  ' talk'
  0.0431  ' visit'
  0.0394  ' help'
  0.0353  ' have'
PROMPT 'When the model finishes'
  0.2808  ' the'
  0.0783  ' its'
  0.0712  ','
  0.0579  ' a'
  0.0130  ' and'
PROMPT 'The most important thing is'
  0.3860  ' to'
  0.1444  ' that'
  0.1046  ' the'
  0.0170  ' a'
  0.0169  ' not'
PROMPT 'If you want to improve'
  0.7468  ' your'
  0.0503  ' the'
  0.0371  ','
  0.0126  ' a'
  0.0092  ' and'
PROMPT 'Where you go I go'
  0.2810  ' to'
  0.0892  ','
  0.0857  ' on'
  0.0781  '.'
  0.0534  ' for'
PROMPT 'Fahim is a'
  0.0174  ' very'
  0.0156  ' member'
  0.0135  ' special'
  0.0107  ' great'
  0.0099  ' major'
PROMPT 'Grammar correction should'
  0.7617  ' be'
  0.0603  ' not'
  0.0364  ' have'
  0.0214  ' include'
  0.0199  ' also'

MODEL stage1 RUN checkpoints/lm_stage1
PROMPT 'The server listens'
  0.5625  ' to'
  0.1100  ' and'
  0.0788  ' in'
  0.0365  '.'
  0.0314  ' for'
PROMPT 'The user should be able'
  0.9927  ' to'
  0.0040  ' and'
  0.0008  ','
  0.0004  ' for'
  0.0001  ' at'
PROMPT 'Please check the configuration'
  0.1490  ' file'
  0.1338  ' for'
  0.0649  ' and'
  0.0523  ' first'
  0.0431  ' before'
PROMPT 'I would like to'
  0.1944  ' play'
  0.0730  ' help'
  0.0650  ' share'
  0.0424  ' be'
  0.0375  ' learn'
PROMPT 'When the model finishes'
  0.4796  ','
  0.2457  ' the'
  0.0771  ' its'
  0.0427  ' and'
  0.0139  ' a'
PROMPT 'The most important thing is'
  0.4931  ' that'
  0.2271  ' to'
  0.0822  ' when'
  0.0623  ' not'
  0.0103  ' the'
PROMPT 'If you want to improve'
  0.8249  ' your'
  0.0416  ','
  0.0282  ' a'
  0.0087  ' and'
  0.0080  ' the'
PROMPT 'Where you go I go'
  0.2642  '.'
  0.2166  ' to'
  0.0752  ','
  0.0616  ' and'
  0.0517  '!"'
PROMPT 'Fahim is a'
  0.0659  ' big'
  0.0459  ' small'
  0.0276  ' little'
  0.0247  ' girl'
  0.0238  ' very'
PROMPT 'Grammar correction should'
  0.5387  ' be'
  0.1213  ' have'
  0.0767  ' include'
  0.0645  ' always'
  0.0269  ' not'
```

Finding:

Stage 1 is stronger on several common continuations: `server listens -> to`, `configuration -> file`, `improve -> your`, and `user should be able -> to`. It is also more probability-concentrated on these prompts. Some prompts regress or become less semantically appropriate, such as `I would like to -> play` and `Where you go I go -> .`. Both models still expose raw BPE pieces rather than complete word suggestions. Stage 1 is a useful improvement candidate, but not yet validated as a general autocomplete improvement.

## Step 2: Stage 1 GEC training

Configuration: `configs/train/gec_stage1_6h.yaml`

The run initializes the Stage 1 GEC model from `checkpoints/lm_stage1`, uses the existing 50,000 synthetic GEC pairs, and keeps the six-hour hard cap.

Initial terminal output:

```text
device=cuda cuda=NVIDIA GeForce RTX 5070
params=39,104,961  task=gec  run=gec_stage1
gec: 0%| | 0/12000
step 20: loss=29.4275, tok_s=23092
step 40: loss=9.9397, tok_s=38245
step 60: loss=4.2732, tok_s=38247
step 80: loss=3.0375, tok_s=33916
step 100: loss=2.8158, tok_s=38351
```

The `triton not found` message is a warning from PyTorch's FLOP counter; training itself is running on CUDA. The early loss drop is expected when the edit head starts learning the synthetic label distribution. It is not a validation-quality result.

The first baseline run was paused at step 1,200 to improve throughput. Its
visible rate was approximately 24,500–34,000 tokens/s with
`micro_batch_size=16`, `grad_accum=4`, and `num_workers=0`. GPU memory was not
the limiting factor. The checkpoint saved at
`checkpoints/gec_stage1/step_00001200`, and the process was stopped cleanly.
The configuration now uses `micro_batch_size=64` and `grad_accum=1`, preserving
the same effective batch size while reducing per-update launch and
accumulation overhead. `num_workers` remains `0` because the current iterable
dataset is not worker-sharded; increasing it without a dataset fix could
duplicate or reorder examples.

Visible stop/throughput evidence:

```text
gec:  11%|▋ | 1320/12000 [05:31, ... loss=1.4310, tok_s=27307]
checkpoints/gec_stage1/latest.json -> {"step": 1200, ...}
nvidia-smi after stop: 0 %, 0 MiB
```

The initial loss fell from about `29.43` at step 20 to about `1.43–1.64`
around steps 1,200–1,320. This is training loss only; it is not evidence of
useful correction quality.

After the restart from step 1,200 with the larger micro-batch, the visible
terminal output was:

```text
[resume] checkpoints\\gec_stage1\\step_00001200 step=1200 samples_seen=76800
step 1220: loss=0.4129, tok_s=8672       # compile/warm-up window
step 1240: loss=0.3849, tok_s=72613
step 1280: loss=0.4085, tok_s=71760
step 1460: loss=0.3700, tok_s=62753
```

A five-sample `nvidia-smi` observation during the resumed run reported GPU
utilization of `30%, 71%, 62%, 54%, 50%` and stable memory usage of about
`4216 MiB / 12227 MiB`. The improvement is therefore approximately 2x in
tokens/s while preserving the effective batch size. The model is relatively
small, so intermittent utilization below 100% is expected; forcing a larger
batch would change optimization behavior and is not justified yet.

The baseline GEC run then completed successfully:

```text
gec: 100%|█████| 12000/12000 [25:54<00:00, 7.70it/s, loss=0.3042, tok_s=57889]
[done] saved checkpoints\\gec_stage1\\step_00012000
latest step: 12000
samples_seen: 768000
```

The GPU was idle after the process exited: `0 %, 0 MiB`.

## Step 3: GEC evaluation

The fixed smoke evaluator is `scripts/evaluate_gec.py`. It uses the same 16
cases for both models: 10 typo/error cases and 6 already-correct sentences.
The baseline comparison was run with:

```powershell
.\\.venv\\Scripts\\python.exe scripts/evaluate_gec.py `
  --stage0 checkpoints/gec_stage0 `
  --stage1 checkpoints/gec_stage1 `
  --maps data/processed/gec/edit_maps.json `
  --device cuda
```

The important baseline findings were:

```text
Stage 0: exact_match_all=4/16, typo_recovery=0/10,
         clean_unchanged=4/6, clean_false_edits=2/6,
         mean_latency_ms=17.03, peak_cuda_memory_mb=105.8
Stage 1: exact_match_all=4/16, typo_recovery=0/10,
         clean_unchanged=4/6, clean_false_edits=2/6,
         mean_latency_ms=23.07, peak_cuda_memory_mb=161.5
```

Both checkpoints produced severe token deletion or corruption on several
examples, including clean sentences. Strict edit precision, recall, and F0.5
were all `0.000` on this smoke set. Therefore the original Stage 1 GEC model
did not pass the quality gate despite completing training.

## Step 4: Repairs

The following repairs were implemented before retraining:

1. Synthetic corruption now preserves the original whitespace, separators, and
   newlines. The previous implementation could rebuild corrupted sentences
   with artificial spaces, creating labels that were not real corrections.
2. Edit alignment now receives source and target token pieces and recognizes
   case-only changes separately.
3. Correction decoding supports `--min-edit-prob` and `--edit-margin`, which
   can force low-confidence edits back to KEEP.
4. The trainer now supports a configurable `keep_downweight` value; the
   repaired run used `1.0` so the result isolates the data repair.
5. Repaired data was written separately to
   `data/processed/gec/repaired/train.jsonl`, leaving the original dataset
   intact.

The repaired dataset check showed:

```text
original: keep=9,641,604 / 10,701,344 labels = 90.10%
repaired: keep=10,958,025 / 11,036,698 labels = 99.29%
original delete rate = 1.12%
repaired delete rate = 0.30%
```

The repaired dataset also passed a whitespace-preservation smoke test. A
confidence-gating test on the old Stage 1 checkpoint did not rescue its bad
outputs, confirming that the training labels/model behavior were the root
problem rather than decoding thresholds alone.

## Step 5: Retraining and final decision

The repaired model was trained with
`configs/train/gec_stage1_repaired_6h.yaml`, using the Stage 1 LM
initialization, repaired data, `micro_batch_size=64`, `grad_accum=1`, and the
same effective batch size as the earlier baseline. It completed at:

```text
[done] saved checkpoints\\gec_stage1_repaired\\step_00012000
latest step: 12000
samples_seen: 768000
final logged loss: 0.0278
```

The final fixed-set evaluation was:

```text
Stage 0: exact_match_all=4/16, typo_recovery=0/10,
         clean_unchanged=4/6, clean_false_edits=2/6,
         typo_changed=4/10, mean_latency_ms=29.35,
         peak_cuda_memory_mb=105.9
Repaired Stage 1: exact_match_all=6/16, typo_recovery=0/10,
         clean_unchanged=6/6, clean_false_edits=0/6,
         typo_changed=1/10, mean_latency_ms=29.21,
         peak_cuda_memory_mb=161.6
```

Representative repaired outputs were unchanged clean sentences, including
`The server listens on port 8000.` and
`Can you check the configuration?`. The repaired model still failed to fix
the tested subject-verb and typo examples; one article example became
`theh`, so typo correction is not yet reliable.

A stricter decode test with `--min-edit-prob 0.90 --edit-margin 0.10` removed
the remaining incorrect typo edit and kept all 6 clean cases unchanged, but
also produced `typo_recovery=0/10`. This confirms that thresholds improve
safety, not correction recall.

Decision: the repaired model passes the no-false-edit part of the gate but
fails the correction-recall part. Do not proceed to ONNX/INT8 packaging,
desktop integration, or Stage 2 yet. The next engineering step is to build a
larger held-out GEC development set with realistic typo and grammar edits,
verify source/target alignment on that set, and tune the edit objective and
decoding against correction precision and F0.5. Resume behavior for the
streaming LM dataset also remains a separate engineering item; it has not
been claimed as fixed here.
