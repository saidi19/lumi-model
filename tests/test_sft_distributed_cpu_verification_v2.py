from pathlib import Path

import pytest
import torch

from lumi import posttraining_distributed_cpu_preflight_v2 as run
from lumi import posttraining_distributed_cpu_verification_v2 as verify


def test_model_verifier_reloads_tensors_instead_of_trusting_published_digest(monkeypatch):
    calls = []

    def state(config, checkpoint):
        calls.append(checkpoint)
        return {"weight": torch.tensor([1.0, 2.0])}

    monkeypatch.setattr(verify, "_model_state", state)
    monkeypatch.setattr(run, "_final_model_digest", lambda path: {"sha256": "a" * 64})
    with pytest.raises(ValueError, match="published model digest"):
        verify._verify_model_pairs(config={}, reference=Path("reference"), resumed=Path("resumed"))
    assert len(calls) == 4


def test_model_verifier_rejects_tensor_mismatch(monkeypatch):
    def state(config, checkpoint):
        return {"weight": torch.tensor([1.0 if checkpoint.parts[0] == "reference" else 2.0])}

    monkeypatch.setattr(verify, "_model_state", state)
    with pytest.raises(ValueError, match="model tensors differ"):
        verify._verify_model_pairs(config={}, reference=Path("reference"), resumed=Path("resumed"))


def test_private_snapshot_rejects_public_file(tmp_path):
    tmp_path.chmod(0o700)
    file = tmp_path / "receipt.json"
    file.write_text("{}")
    file.chmod(0o644)
    with pytest.raises(ValueError, match="private file metadata"):
        verify._tree_fingerprint(tmp_path)


def test_verifier_refuses_to_overwrite_existing_verification(tmp_path, monkeypatch):
    target = tmp_path / "verified.json"
    target.write_text("preserve")
    monkeypatch.setattr(
        run,
        "load_plan",
        lambda *args, **kwargs: {
            "outputs": {"verification": "verified.json"},
        },
    )
    with pytest.raises(FileExistsError):
        verify.verify_distributed_cpu_preflight(
            plan_path=tmp_path / "plan.json", repository_root=tmp_path
        )
    assert target.read_text() == "preserve"
