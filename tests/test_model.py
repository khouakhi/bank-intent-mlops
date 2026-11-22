"""Tests that every model version must pass before it is deployed."""
import json
import time
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.predict import IntentClassifier, ROOT

GATE = json.loads((ROOT / "quality_gate.json").read_text())


@pytest.fixture(scope="session")
def clf():
    return IntentClassifier.from_config()


@pytest.fixture(scope="session")
def golden():
    return pd.read_csv(ROOT / "data" / "golden_set.csv")


def test_model_loads(clf):
    assert len(clf.id2label) > 0


def test_prediction_contract(clf):
    """The app relies on this output format: never change it silently."""
    out = clf.predict(["my card has not arrived", "how do I top up?"])
    assert len(out) == 2
    for o in out:
        assert set(o) == {"intent", "confidence"}
        assert o["intent"] in clf.id2label.values()
        assert 0.0 <= o["confidence"] <= 1.0


def test_handles_odd_inputs(clf):
    out = clf.predict(["", "card " * 500, "???"])
    assert len(out) == 3


def test_latency_budget(clf):
    clf.predict("warm up")
    msgs = ["where is my new card"] * 20
    t0 = time.perf_counter()
    for m in msgs:
        clf.predict(m)
    avg_ms = (time.perf_counter() - t0) / len(msgs) * 1000
    print(f"average latency: {avg_ms:.1f} ms")
    assert avg_ms <= GATE["max_latency_ms"], f"Too slow: {avg_ms:.0f} ms per message"


def test_model_knows_all_golden_intents(clf, golden):
    unknown = sorted(set(golden["category"]) - set(clf.id2label.values()))
    assert not unknown, f"The model cannot predict these intents: {unknown}"


def test_quality_gate(clf, golden):
    preds = [p["intent"] for p in clf.predict(golden["text"].tolist())]
    acc = accuracy_score(golden["category"], preds)
    f1 = f1_score(golden["category"], preds, average="macro")
    Path(ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "gate_report.json").write_text(json.dumps(
        {"accuracy": acc, "macro_f1": f1, "n_examples": len(golden), "gate": GATE}, indent=2))
    print(f"golden set: accuracy={acc:.3f} macro_F1={f1:.3f}")
    assert acc >= GATE["min_accuracy"], f"Accuracy {acc:.3f} is below the gate {GATE['min_accuracy']}"
    assert f1 >= GATE["min_macro_f1"], f"Macro F1 {f1:.3f} is below the gate {GATE['min_macro_f1']}"
