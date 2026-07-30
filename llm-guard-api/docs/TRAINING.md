# Training the intent classifier

The guard ships with two classifier backends. Pick one with `CLASSIFIER_BACKEND`
in `.env`, or `--backend` on the CLI.

| | `baseline` (default) | `transformer` |
|---|---|---|
| Model | TF-IDF (word 1-2gram + char 3-5gram) -> logistic regression | DeBERTa-v3-small fine-tune |
| Training time | ~10 s on a laptop CPU | ~30 min on a Colab GPU, hours on CPU |
| Extra dependencies | none beyond `requirements.txt` | `requirements-transformer.txt` (torch, transformers, sentencepiece) |
| Artifact | `guard/models/baseline_classifier.joblib` (~5 MB) | `guard/models/intent_classifier/` (~570 MB) |
| Inference | ~1 ms | ~50 ms CPU |

Start with the baseline. It scores 0.99 held-out accuracy on the bundled corpus,
which is hard to beat with a fine-tune on the same data.

## The dataset

`data/prompts.csv` holds 8,123 deduplicated prompts with a `prompt` and a
`label` column, derived from
[`xTRam1/safe-guard-prompt-injection`](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection):

| label | count |
|---|---|
| `benign` | 5,683 |
| `malicious` | 2,440 |

Re-download and re-normalise it with:

```bash
python train.py --download-only --force-download
```

`train.py` accepts several common column namings (`prompt`/`text`/`input`,
`label`/`target`/`is_injection`) and maps numeric or string labels onto
`benign` / `suspicious` / `malicious`, so you can point `HF_DATASET_NAME` at a
different dataset without editing code.

### Note on the `suspicious` class

The public dataset is binary. The pipeline still exposes three intents because
the decision engine needs a middle band that maps to SANITIZE rather than BLOCK.

The baseline backend produces that band by thresholding a single malicious
probability:

```
p >= MALICIOUS_THRESHOLD  (0.70) -> malicious  -> BLOCK
p >= SUSPICIOUS_THRESHOLD (0.40) -> suspicious -> SANITIZE
otherwise                        -> benign     -> ALLOW
```

Widening that band trades false negatives for sanitization; narrowing it lets
more borderline prompts through untouched.

The transformer backend cannot invent the class: trained on binary data it will
only ever emit `benign` or `malicious`, and `train.py` logs a warning saying so.
To get a genuine three-way head, add `suspicious` rows to the CSV first.

## Baseline

```bash
python train.py --backend baseline          # download if needed, then train
python train.py --backend baseline --train-only
```

Writes `guard/models/baseline_classifier.joblib` and
`guard/models/baseline_metrics.json`, and prints a full classification report.

If no artifact exists, `BaselineIntentClassifier` trains one automatically on
first use from `data/prompts.csv`, so a clean checkout works with no setup step.

Held-out metrics on the bundled corpus (80/20 stratified split, seed 42):

```
accuracy 0.9932   precision 0.9979   recall 0.9795   f1 0.9886   roc_auc 0.9998
```

## Transformer

```bash
pip install -r requirements.txt -r requirements-transformer.txt
python train.py --backend transformer --epochs 3
USE_GPU=1 python train.py --backend transformer --epochs 3   # force CUDA
```

Writes a standard `transformers` checkpoint to
`guard/models/intent_classifier/` plus `training_metrics.json`.

### Google Colab

`train_classifier.ipynb` runs the same fine-tune on a free Colab GPU. After it
finishes, download the output directory and drop it at
`guard/models/intent_classifier/`, or point `CLASSIFIER_MODEL_PATH` at wherever
you put it.

Verify with:

```bash
python setup_model.py --check
```

### Fallback behaviour

`guard.build_classifier()` falls back to the baseline when the transformer
backend is requested but torch is missing or no checkpoint exists. This is
deliberate: loading the pre-trained backbone with a randomly initialised
classification head yields random verdicts, which would silently disable the ML
layer while still looking healthy.
