# Integration Guide: Using Notebook-Trained Models

This guide explains how the LLM Guard system integrates with models trained via the Jupyter notebook.

## 🎯 Overview

```
┌─────────────────────────┐
│  train_classifier.ipynb │  (Colab GPU Training)
│  - Fine-tunes model     │
│  - Saves to Drive/local │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  guard/models/          │  (Model Directory)
│  intent_classifier/     │
│  ├─ pytorch_model.bin   │
│  ├─ config.json         │
│  ├─ vocab.txt           │
│  └─ training_metrics.json
└────────────┬────────────┘
             │
             ▼
┌──────────────────────────────┐
│  config.get_trained_model_   │  (Auto-Detection)
│  path()                      │
│  - Finds model location      │
│  - Returns path to Guard     │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  IntentClassifier            │  (Loads Model)
│  - Loads from notebook path  │
│  - Falls back to pre-trained │
│  - Ready for inference       │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  LLMGuard.guard()            │  (Inference)
│  - Uses classifier           │
│  - Returns predictions       │
└──────────────────────────────┘
```

## 📋 Step-by-Step Workflow

### Step 1: Train Model in Colab

**File:** `train_classifier.ipynb`

1. Open in [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime > Change runtime type > GPU**
3. Run all cells sequentially
4. Model automatically saves to Google Drive at:
   ```
   /content/drive/My Drive/llm-guard/intent_classifier/
   ```

**Outputs:**
- `pytorch_model.bin` - Trained weights
- `config.json` - Model config
- `vocab.txt` - Tokenizer vocab
- `training_metrics.json` - Training results

### Step 2: Download Model

Option A: Download from Google Drive
- Navigate to `/content/drive/My Drive/llm-guard/intent_classifier/`
- Download as ZIP

Option B: Direct from Colab
- Use Colab's download feature on the output files

### Step 3: Place Model Locally

Extract to your project:
```
llm-guard-api/
├── guard/
│   └── models/
│       └── intent_classifier/
│           ├── pytorch_model.bin      ← Download this
│           ├── config.json            ← Download this
│           ├── vocab.txt              ← Download this
│           └── training_metrics.json  ← Download this
```

### Step 4: Verify Installation

```bash
python setup_model.py --check
```

Expected output:
```
✅ Fine-tuned model found and valid!

Location: guard/models/intent_classifier

Model: distilbert-base-uncased
Intent classes: ['benign', 'suspicious', 'malicious']

Training Config:
  - Epochs: 3
  - Batch size: 16
  - Learning rate: 2e-05

Dataset Stats:
  - Total examples: 150
  - Train examples: 120
  - Validation examples: 30

Final Metrics:
  - Training loss: 0.1234
  - Validation accuracy: 0.9067
  - Validation F1: 0.9055
```

### Step 5: Test Inference

```bash
python setup_model.py --test
```

### Step 6: Use in Your Code

The guard automatically loads the trained model:

```python
from app import LLMGuard

# Automatically detects and loads trained model
guard = LLMGuard()

# Use it
result = guard.guard("What is the capital of France?")
print(result["decision"])  # "allow"
print(result["metadata"]["intent_analysis"])
```

## 🔍 Auto-Detection Logic

The system automatically finds your trained model:

```python
# In config.py
def get_trained_model_path() -> str:
    """
    Checks in this order:
    1. CLASSIFIER_MODEL_PATH env variable
    2. guard/models/intent_classifier/
    3. ./intent_classifier/
    
    Returns first valid path, or default if none found
    """
```

Model is valid if it contains:
- `pytorch_model.bin` (required)
- `config.json` (required)
- `vocab.txt` (required)

## 📊 Model Components

### From Notebook Training

The notebook automatically generates these files:

**pytorch_model.bin** (150-200 MB)
- DistilBERT weights fine-tuned on your dataset
- Used during inference

**config.json**
```json
{
  "model_name": "distilbert-base-uncased",
  "num_labels": 3,
  "intent_classes": ["benign", "suspicious", "malicious"],
  "training_config": {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "max_length": 128
  },
  "dataset_stats": {
    "total_examples": 150,
    "train_examples": 120,
    "val_examples": 30
  },
  "final_metrics": {
    "train_loss": 0.1234,
    "val_accuracy": 0.9067,
    "val_f1": 0.9055
  }
}
```

**vocab.txt**
- DistilBERT tokenizer vocabulary (30,522 tokens)
- Maps words to token IDs

**training_metrics.json**
```json
{
  "train_loss": [0.5, 0.3, 0.15],
  "val_accuracy": [0.85, 0.88, 0.91],
  "val_f1": [0.84, 0.87, 0.91]
}
```

## 🛠️ Troubleshooting

### "No trained model found"

**Check 1: Model exists?**
```bash
ls guard/models/intent_classifier/
```

Should show: `pytorch_model.bin`, `config.json`, `vocab.txt`

**Check 2: Use setup utility**
```bash
python setup_model.py --check
```

**Fix:** Download model from Colab/Drive and place in `guard/models/intent_classifier/`

### "Failed to load model"

**Check 1: Model path correct?**
```python
import config
print(config.get_trained_model_path())
```

**Check 2: File permissions?**
```bash
chmod -R 755 guard/models/
```

**Check 3: Pytorch version mismatch?**
```bash
pip install -r requirements.txt --upgrade
```

### "Falling back to pre-trained"

This is normal! Means:
- Trained model not found (use pre-trained instead)
- Model directory exists but weights missing

Solution: Train via notebook or place trained model in correct location

## 📈 Performance

### After Notebook Training

Expected metrics (on 150 examples):
- **Validation Accuracy:** 85-92%
- **Validation F1-Score:** 0.85-0.91
- **Training Time:** ~30 min (GPU), ~2 hours (CPU)
- **Inference Latency:** ~50ms per prompt

### Pre-trained Fallback

If no trained model:
- Uses `distilbert-base-uncased` (pre-trained)
- Accuracy: ~60-70% (lower than fine-tuned)
- No training required
- Same latency: ~50ms

## 🔄 Retraining

To train a new version:

1. Update training data in notebook (if desired)
2. Re-run all cells in `train_classifier.ipynb`
3. Download new model
4. Replace old model in `guard/models/intent_classifier/`
5. Guard automatically uses new model on restart

## 🚀 Production Deployment

### Local Deployment

1. Train model in Colab
2. Download to local machine
3. Place in `guard/models/intent_classifier/`
4. Deploy application

### Cloud Deployment

1. Train model in Colab
2. Upload to cloud storage (S3, GCS, etc)
3. App downloads during startup:
   ```python
   # In app initialization
   download_model_if_needed()
   guard = LLMGuard()
   ```

### Docker Deployment

Include trained model in Docker image:
```dockerfile
COPY guard/models/intent_classifier/ /app/guard/models/intent_classifier/
```

## 📝 Summary

The notebook-based training workflow:
1. ✅ Trains on GPU for speed
2. ✅ Auto-saves to Google Drive
3. ✅ Easy to retrain
4. ✅ Auto-detected by guard
5. ✅ Falls back to pre-trained if needed
6. ✅ Production-ready deployment

**Next:** Train your model using `train_classifier.ipynb`!
