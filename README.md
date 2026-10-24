# TyPoTaKer

TyPoTaKer is an offline-first language-model framework for lightweight typo correction, grammar correction, and next-word suggestions.

It is designed for a Windows desktop assistant, but the current repository focuses on the model, data, training, checkpoint, and export layers. The desktop overlay is intentionally a later layer.

The model is trained from scratch on a single RTX 5070 and is designed around a strict six-hour training budget. Every stage is checkpointed so training can be paused, moved to another day, resumed, or expanded into a larger stage.

## Current model status

| Component | Current state |
|---|---|
| Stage 0 LM | 25,369,601 parameters; baseline trained checkpoint available |
| Stage 1 LM & GEC | 39,104,961 parameters; Stage 1 Quality Gate PASSED |
| Stage 2 LM & GEC | 93,888,001 parameters; Stage 2 Quality Gate PASSED (99.2% clean preservation) |
| Tokenizer | 16k byte-level BPE with special tokens and character CNN embeddings |
| Export & Quantization | ONNX FP32 and INT8 quantized models (~125–138 MB, sub-10ms CPU latency) |
| Desktop App | .NET 8 WPF system tray & caret overlay with right-click selection grammar check |

The current architecture should be treated as an experimental research and engineering design. The model is not intended to compete with large general-purpose LLMs.

## Architecture at a glance

TyPoTaKer uses one character-aware transformer trunk with two specialized output heads:

```text
                                  +-----------------------------+
                                  | Next-token LM head          |
                                  | tied to token embeddings    |
                                  +-----------------------------+
                                                ^
                                                |
Input text                                      |
    |                                           |
    +--> 16k BPE tokens --> token embeddings --+--> gated fusion
    |                                           |
    +--> token bytes --> character CNN --------+
                                                |
                         +----------------------+----------------------+
                         |                                             |
                  Stage-aware transformer trunk                 Edit-tag head
                  8 or 10 pre-norm blocks                       for GEC
                  GQA + RoPE + RMSNorm + SwiGLU                  KEEP / DELETE /
                  scaled dot-product attention                   CASE / REPLACE /
                         |                                       APPEND
                         +----------------------+----------------------+
                                                |
                                  corrected token actions
```

The same representation is useful for both tasks, but the attention mode changes:

| Task | Attention mode | Output | Why |
|---|---|---|---|
| Next-word suggestion | Causal | Vocabulary logits | The model can only see the left context |
| Grammar correction | Bidirectional | One edit action per input token | The whole sentence is available when correcting |

The GEC model does not generate an unrestricted replacement sentence. It predicts a constrained edit action for each input token. This makes it smaller, faster, and less likely to hallucinate new content.

## Detailed data path

For an input sequence of tokens `t_0 ... t_(T-1)`, TyPoTaKer builds two synchronized representations.

### 1. Subword representation

The tokenizer is a 16,000-entry byte-level BPE tokenizer. It handles normal words, punctuation, spaces, numbers, and unusual strings without requiring a huge word vocabulary.

Special tokens are:

```text
<pad>  <unk>  <bos>  <eos>
```

The BPE embedding table is:

```text
E_tok[t_i] in R^d
```

where `d` is 384 in Stage 0 and 512 in Stage 1.

### 2. Character representation

Subword tokenization is efficient, but a typo can change a whole token. For example, `server`, `servre`, and `ser ver` may receive very different BPE pieces.

To preserve letter-level information, every BPE piece is also converted into up to 16 byte/character IDs. The character path is:

```text
character IDs
    -> character embedding
    -> 1D convolution + GELU
    -> 1D convolution + GELU
    -> max pooling over characters
    -> projection to model width
```

For token `t_i`, the fused input is:

```text
x_i = E_tok[t_i] + alpha * E_char(t_i)
```

`alpha` is a learned scalar initialized to `0.1`. This lets the model begin with a stable language representation and gradually learn how much character evidence helps.

This hybrid representation is one of the central TyPoTaKer ideas: BPE supplies efficient language modeling while the character path keeps local spelling evidence visible.

## Transformer trunk

Each block is a pre-normalized residual transformer block:

```text
h_1 = h + GQA(RMSNorm(h))
h_2 = h_1 + SwiGLU(RMSNorm(h_1))
```

### Grouped-query attention

Queries use more heads than keys and values. Multiple query heads share a smaller number of key/value heads:

```text
Stage 0: 6 query heads, 2 key/value heads
Stage 1: 8 query heads, 2 key/value heads
```

This reduces key/value memory and computation while preserving useful query capacity. Attention is implemented with PyTorch scaled dot-product attention so the runtime can select an efficient CUDA kernel when available.

### RoPE positions

Rotary position embeddings are applied to query and key vectors. RoPE gives the model relative position information without adding a large learned position table.

### RMSNorm

RMSNorm is used before attention and the feed-forward network. It removes the need for mean-centering and is inexpensive for small inference models.

### SwiGLU feed-forward network

The feed-forward part uses a gated activation:

```text
SwiGLU(x) = W_2( SiLU(W_1(x)) * W_3(x) )
```

The gate gives the small model additional nonlinear capacity without requiring a larger vocabulary or a much deeper network.

## Output head A: next-word language modeling

The language-model head is tied to the token embedding matrix:

```text
logits_i = h_i E_tok^T
```

Weight tying reduces parameters and makes the input and output vocabulary representations consistent.

During normal inference, the model predicts the next BPE token from the left context. The CLI currently exposes the top-k raw BPE pieces; a future desktop layer should group these pieces into user-facing word suggestions.

### Training-only multi-token prediction

In addition to the normal next-token loss, Stage 0 and Stage 1 use two small projection adapters to predict tokens at offsets `t+2` and `t+3`:

```text
L_LM = L(t+1) + 0.30 L(t+2) + 0.15 L(t+3)
```

The auxiliary heads share the tied vocabulary head and are discarded from the inference path conceptually. Their purpose is to encourage representations that contain slightly more future information during training without turning inference into a multi-head production system.

## Output head B: constrained grammar correction

The correction head assigns one edit label to each input token. The Stage 0/Stage 1 label space contains:

```text
KEEP
DELETE
CASE_CAP
CASE_LOWER
REPLACE_0 ... REPLACE_2047
APPEND_0  ... APPEND_2047
```

The replace and append classes refer to frequent tokenizer IDs collected from the clean training corpus. The decoder maps the predicted action back to a token ID using `edit_maps.json`.

The inference path is therefore:

```text
wrong sentence
    -> BPE + character features
    -> bidirectional TyPoFormer trunk
    -> edit label for every token
    -> constrained token edits
    -> corrected sentence
```

This design has an important safety property: the corrector cannot freely invent an arbitrary paragraph. It can keep, delete, change case, replace with a known token, or append a known token.

The current implementation performs one edit pass. Multiple conservative passes and confidence thresholds are planned after evaluation proves that they improve correction quality without increasing false edits.

## Stage growth and resumable training

The model is intentionally staged instead of forcing the largest architecture to train from the beginning.

| Stage | Width | Layers | Parameters | Purpose |
|---|---:|---:|---:|---|
| Stage 0 | 384 | 8 | 25,369,601 | Fast baseline and stable first training run |
| Stage 1 | 512 | 10 | 39,104,961 | More capacity after Stage 0 has been validated |

Stage growth copies the compatible part of the old model into the new configuration. Newly added residual blocks are initialized as near-identity blocks by zeroing their residual output projections. This allows a later stage to begin near the behavior of the previous stage instead of starting from random weights.

The width expansion is a warm-start strategy, not an exact function-preserving transformation: the old weights occupy the compatible prefix of the wider tensors and new dimensions are initialized separately.

Every checkpoint contains:

```text
checkpoints/<run>/step_XXXXXXXX/
    model.pt
    optimizer.pt
    scheduler.pt
    rng.pt
    trainer_state.json
```

Checkpoint folders are written through a temporary directory and atomic rename. This protects a completed checkpoint if the process is interrupted while saving.

## Training objectives

### Language model objective

The LM uses causal cross-entropy over the next token, plus the two training-only future-token losses described above.

### Correction objective

The GEC model uses weighted cross-entropy over edit labels. `KEEP` is downweighted because most tokens in a sentence normally remain unchanged. This weighting must be calibrated on a validation set: too little KEEP weight creates unsafe overcorrection, while too much makes the model ignore real errors.

### Transfer workflow

The intended workflow is sequential:

```text
random initialization
        |
        v
Stage 0 LM training
        |
        v
Stage 1 LM growth and continuation
        |
        v
GEC head/trunk training initialized from the best LM checkpoint
```

The language model learns general English structure first. The GEC objective then specializes the shared trunk and edit head for correction.

## Data pipeline

The repository currently uses:

- TinyStories and filtered FineWeb-Edu text for language modeling.
- A 16k BPE tokenizer trained from the clean corpus.
- Synthetic noisy/clean pairs for initial GEC training.
- Character-level typos, confusions, article changes, preposition changes, case changes, and simple subject/verb suffix changes.

Synthetic data is useful for bootstrapping, but it is not automatically realistic. Before calling the model production-ready, the data pipeline needs a held-out validation set, more realistic corruption distributions, and evaluation on licensed real-error data.

## Why this design is distinctive

The individual building blocks are established techniques. The distinctive part of TyPoTaKer is their combination under one practical constraint: a small offline assistant that can be trained, paused, resumed, and grown on consumer hardware.

### 1. Character-aware subword modeling for a tiny model

Many small language models rely only on subword IDs. TyPoTaKer adds a compact character CNN to each BPE piece, giving the same trunk access to both language context and the internal letters of a typo.

This is especially relevant to the target use case because desktop typing errors are often local character mistakes rather than unknown words.

### 2. One trunk, two complementary behaviors

The same compact transformer supports:

- causal next-token prediction for autocomplete;
- bidirectional constrained edit prediction for correction.

This avoids shipping two completely unrelated large models while still allowing each task to use its appropriate attention mode and output space.

### 3. Constrained correction instead of unrestricted rewriting

The GEC head predicts edit actions rather than generating a free-form replacement sentence. This is a deliberate safety and latency choice for an always-on typing assistant.

The corrector is therefore designed to prefer precision and predictable edits over expressive rewriting.

### 4. Training-only future prediction

Multi-token prediction adds a small auxiliary training signal without requiring extra inference-time decoding heads. It is an efficiency-oriented training idea for extracting more useful representations from a fixed training budget.

### 5. Progressive, identity-aware model growth

The Stage 0 to Stage 1 path treats checkpoints as reusable research artifacts. A developer can train a smaller model within one session, inspect it, and expand it later without discarding the learned representation.

This is particularly useful when the available training window is limited to six hours and experimentation must survive interruptions.

## What must be proven before calling it novel research

The architecture is a novel integrated design for this repository, but architectural novelty should not be claimed as a scientific result until it is measured.

The required ablations are:

| Experiment | Question |
|---|---|
| Remove character CNN | Does character input improve typo recovery? |
| Remove multi-token prediction | Does MTP improve validation perplexity or top-k accuracy? |
| Replace GQA with full attention | What is the quality/latency tradeoff? |
| Remove staged growth | Does warm-start growth improve quality per GPU-hour? |
| Free-form correction baseline | Does constrained editing reduce false corrections? |
| Stage 0 vs Stage 1 | Does the larger model justify its memory and latency cost? |

The project should report validation quality, correction precision, recall/F0.5, no-change accuracy, latency, memory, and training throughput for these comparisons. Until then, “novel” means a documented design hypothesis, not a proven research contribution.

## Six-hour training recipes

### Stage 0 LM

```powershell
python -m typotaker train --config configs/train/lm_6h.yaml
```

### Grow Stage 0 into Stage 1

```powershell
python -m typotaker grow `
  --src checkpoints/lm_stage0 `
  --config configs/model/stage1_40m.yaml `
  --run-name lm_stage1
```

### Stage 1 LM continuation

```powershell
python -m typotaker train --config configs/train/lm_stage1_6h.yaml
```

The Stage 1 recipe uses a smaller micro-batch and gradient accumulation to preserve a similar effective token batch while fitting the wider network.

### Smoke test

```powershell
python -m typotaker prepare-smoke
python -m typotaker train-tokenizer `
  --inputs data/processed/lm/smoke.jsonl `
  --out data/tokenizer/tokenizer.json `
  --vocab-size 16000
python -m typotaker train `
  --config configs/train/smoke.yaml `
  --no-compile
```

## Inference

```powershell
python -m typotaker suggest `
  --run checkpoints/lm_stage1 `
  --text "The server listens"

python -m typotaker correct `
  --run checkpoints/gec_stage0 `
  --text "Its important to chekc their config"
```

The current `suggest` command returns raw BPE pieces. A production UI should combine compatible pieces into word-level suggestions, apply a confidence threshold, and avoid suggesting inside password fields or blocked applications.

## Export status

The repository contains an ONNX export path:

```powershell
python -m typotaker export `
  --run checkpoints/lm_stage1 `
  --task lm `
  --out export/lm_stage1.onnx
```

The current exporter produces a float ONNX graph. INT8 quantization, ONNX Runtime verification, CPU latency measurement, and KV-cache support are follow-up engineering tasks.

## Repository layout

```text
TyPoTaKer/
  configs/model/          Stage 0 and Stage 1 model definitions
  configs/train/          Smoke, Stage 0, Stage 1, and GEC recipes
  data/                   Local ignored corpora and tokenizer artifacts
  checkpoints/            Local ignored resumable checkpoints
  src/typotaker/models/   TyPoFormer and stage growth
  src/typotaker/data/     Corpus preparation, corruption, and alignment
  src/typotaker/train/    CUDA training and checkpoint handling
  src/typotaker/infer.py  PyTorch inference helpers
  src/typotaker/export.py ONNX export path
  tests/                  CPU unit tests
  app/                    Future desktop overlay integration
```

## Verification

```powershell
pytest
python -m typotaker count-params --config configs/model/stage0_25m.yaml
python -m typotaker count-params --config configs/model/stage1_40m.yaml
```

## Limitations and next milestones

The current GEC system is an early synthetic-data baseline. It can overcorrect and should not yet be installed as an always-on typing service.

The next milestones are:

1. Create fixed LM and GEC validation sets.
2. Measure Stage 0 versus Stage 1 quality.
3. Repair synthetic corruption and edit alignment based on validation results.
4. Add confidence-aware and precision-first decoding.
5. Fix resumable data cursors so long LM jobs do not replay the entire corpus.
6. Export, quantize, and benchmark the best checkpoint.
7. Build the offline Windows overlay only after model quality and latency are acceptable.

## License

The code is MIT licensed. Training corpora, generated datasets, tokenizer files, checkpoints, and exported weights may have separate licensing and redistribution requirements. Do not commit large training artifacts without documenting their provenance and license.
