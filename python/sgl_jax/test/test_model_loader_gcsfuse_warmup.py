import logging
from types import SimpleNamespace


def test_gcsfuse_warmup_can_be_skipped_before_mount_or_weight_reads(monkeypatch, caplog):
    from sgl_jax.srt.model_loader.loader import JAXModelLoader

    monkeypatch.setenv("SGLANG_JAX_SKIP_GCSFUSE_WARMUP", "1")

    def fail_open(*args, **kwargs):
        raise AssertionError(f"warm-up touched the filesystem: args={args}, kwargs={kwargs}")

    monkeypatch.setattr("builtins.open", fail_open)
    with caplog.at_level(logging.INFO):
        JAXModelLoader._warmup_safetensors_cache(
            SimpleNamespace(model_path="/models/GLM-5.2")
        )

    assert "Skipping GCSFuse cache warm-up by request" in caplog.text
