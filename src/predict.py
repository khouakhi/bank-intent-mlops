"""Load the intent model from the Hugging Face Hub and classify customer messages."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "model_config.json"
    return json.loads(path.read_text())


class IntentClassifier:
    def __init__(self, model_repo, revision="main", max_length=64):
        # revision = a tag such as "v1": this pins EXACTLY which model version is used
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_repo, revision=revision).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_length = max_length

    @classmethod
    def from_config(cls, path=None):
        cfg = load_config(path)
        return cls(cfg["model_repo"], cfg.get("revision", "main"))

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, dim=-1))
        return torch.cat(out)

    def predict(self, texts):
        probs = self.predict_proba(texts)
        conf, idx = probs.max(dim=-1)
        return [{"intent": self.id2label[int(i)], "confidence": float(c)} for c, i in zip(conf, idx)]

    def top_k(self, text, k=5):
        probs = self.predict_proba([text])[0]
        vals, idx = probs.topk(min(k, len(probs)))
        return {self.id2label[int(i)]: float(v) for v, i in zip(vals, idx)}
