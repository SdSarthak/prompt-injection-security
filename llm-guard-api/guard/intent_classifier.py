"""Transformer-based intent classifier for detecting prompt injection attempts.

Importing this module requires torch and transformers. Use
``guard.build_classifier()`` if you want automatic fallback to the CPU-only
baseline backend when those are unavailable or no checkpoint has been trained.
"""

import json
import logging
import os
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import config
from .results import ClassificationResult

logger = logging.getLogger(__name__)

BACKEND_NAME = "transformer"
PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME", "microsoft/deberta-v3-small")

__all__ = ["IntentClassifier", "PromptDataset", "ClassificationResult"]


class PromptDataset(Dataset):
    """PyTorch Dataset for prompt classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class IntentClassifier:
    """Fine-tuned DeBERTa classifier for prompt injection intent detection."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize classifier with fine-tuned or pre-trained model.
        
        Tries to load fine-tuned model first, falls back to pre-trained DeBERTa-v3-small.
        
        Args:
            model_path: Path to trained model directory. If None, auto-detects using config.
            device: Device to use ('cpu' or 'cuda'). Auto-detects GPU if None.
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Use intent classes from config
        self.intent_to_id = config.INTENT_TO_ID
        self.id_to_intent = config.ID_TO_INTENT
        
        # Determine model path
        if model_path is None:
            model_path = config.get_trained_model_path()
        
        self.model_path = model_path
        self.is_fine_tuned = False

        # `config.has_transformer_weights` accepts either pytorch_model.bin or
        # model.safetensors, which is what recent transformers versions write.
        if config.has_transformer_weights(model_path):
            logger.info("Loading fine-tuned model from %s", model_path)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.is_fine_tuned = True
                logger.info("Model and tokenizer loaded successfully")
            except Exception as exc:
                logger.warning("Failed to load %s (%s); falling back to pre-trained", model_path, exc)
                self._load_pretrained()
        else:
            logger.warning(
                "No fine-tuned checkpoint at %s. Falling back to pre-trained %s, whose "
                "classification head is randomly initialised - predictions are NOT meaningful "
                "until you run `python train.py --backend transformer`.",
                model_path,
                PRETRAINED_MODEL_NAME,
            )
            self._load_pretrained()

        self.model.to(self.device)
        self.model.eval()

    def _load_pretrained(self):
        """Load the pre-trained backbone with a fresh classification head."""
        logger.info("Loading pre-trained %s", PRETRAINED_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_NAME, num_labels=len(config.INTENT_CLASSES)
        )
        self.is_fine_tuned = False

    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify a prompt's intent.
        
        Args:
            prompt: Prompt to classify
            
        Returns:
            ClassificationResult with intent, confidence, and class scores
        """
        return self.batch_classify([prompt])[0]

    def batch_classify(self, prompts: List[str], batch_size: int = 32) -> List[ClassificationResult]:
        """
        Classify multiple prompts in batched forward passes.

        Args:
            prompts: List of prompts to classify
            batch_size: Number of prompts per forward pass

        Returns:
            List of ClassificationResult objects, aligned with `prompts`
        """
        cleaned = [("" if p is None else str(p)) for p in prompts]
        if not cleaned:
            return []

        results: List[ClassificationResult] = []

        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            inputs = self.tokenizer(
                chunk,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for row in probabilities:
                # `id_to_intent` is keyed by Python ints, so cast off numpy's int64.
                predicted_id = int(np.argmax(row))
                results.append(
                    ClassificationResult(
                        intent=self.id_to_intent[predicted_id],
                        confidence=float(row[predicted_id]),
                        class_scores={
                            self.id_to_intent[i]: float(row[i]) for i in range(len(row))
                        },
                        backend=BACKEND_NAME,
                    )
                )

        return results

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Fine-tune the model on labeled prompt data.
        
        Args:
            train_texts: Training prompt texts
            train_labels: Training labels ("benign", "suspicious", "malicious")
            val_texts: Validation prompt texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            output_dir: Directory to save fine-tuned model
            
        Returns:
            Dictionary with training metrics
        """
        from sklearn.metrics import f1_score

        unknown = {label for label in list(train_labels) + list(val_labels)} - set(self.intent_to_id)
        if unknown:
            raise ValueError(
                f"Unknown intent label(s) {sorted(unknown)}; expected one of {config.INTENT_CLASSES}"
            )

        # Convert labels to ids
        train_label_ids = [self.intent_to_id[label] for label in train_labels]
        val_label_ids = [self.intent_to_id[label] for label in val_labels]

        # Create datasets and dataloaders
        train_dataset = PromptDataset(train_texts, train_label_ids, self.tokenizer)
        val_dataset = PromptDataset(val_texts, val_label_ids, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Training loop
        self.model.train()
        metrics = {"train_loss": [], "val_accuracy": [], "val_f1": []}

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            metrics["train_loss"].append(avg_loss)
            print(f"Training loss: {avg_loss:.4f}")

            # Validation
            self.model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)

                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            accuracy = float((np.array(val_preds) == np.array(val_true)).mean())
            f1 = float(f1_score(val_true, val_preds, average="weighted", zero_division=0))

            metrics["val_accuracy"].append(accuracy)
            metrics["val_f1"].append(f1)

            print(f"Validation accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            self.model.train()

        self.model.eval()

        # Save model if output dir specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "training_metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            self.is_fine_tuned = True
            self.model_path = output_dir
            logger.info("Model saved to %s", output_dir)

        return metrics
