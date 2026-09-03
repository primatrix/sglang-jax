import logging
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.model_runner import ModelRunner, _SimDevice


def _simulation_runner(*, vocab_size: int = 100) -> ModelRunner:
    runner = object.__new__(ModelRunner)
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    runner.mesh = Mesh(devices, ("data", "tensor"))
    runner.model_config = SimpleNamespace(vocab_size=vocab_size)
    return runner


def test_simulation_logits_are_lightweight_unless_full_vocab_is_requested():
    runner = _simulation_runner(vocab_size=97)
    batch = SimpleNamespace(batch_size=3)

    lightweight = runner.simulation_logits_output(batch)
    full_vocab = runner.simulation_logits_output(batch, full_vocab=True)

    assert lightweight.next_token_logits.shape == (3, 1)
    assert full_vocab.next_token_logits.shape == (3, 97)


def test_simulation_sample_fills_only_valid_rows():
    runner = _simulation_runner(vocab_size=100)

    next_token_ids = runner.simulation_sample(np.asarray([4, 0, 9], dtype=np.int32))

    np.testing.assert_array_equal(jax.device_get(next_token_ids), [32, 0, 32])


def test_simulation_sampling_falls_back_for_logprob_requests():
    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace(simulate_compute=True)

    assert worker._use_simulated_sampling(
        SimpleNamespace(
            return_logprob=False,
            return_output_logprob_only=False,
            sampling_info=SimpleNamespace(grammars=None),
        )
    )
    assert not worker._use_simulated_sampling(
        SimpleNamespace(
            return_logprob=True,
            return_output_logprob_only=False,
            sampling_info=SimpleNamespace(grammars=None),
        )
    )
    assert not worker._use_simulated_sampling(
        SimpleNamespace(
            return_logprob=False,
            return_output_logprob_only=True,
            sampling_info=SimpleNamespace(grammars=None),
        )
    )
    assert not worker._use_simulated_sampling(
        SimpleNamespace(
            return_logprob=False,
            return_output_logprob_only=False,
            sampling_info=SimpleNamespace(grammars=[object()]),
        )
    )


def test_sim_device_logs_actual_compute_interval(caplog):
    caplog.set_level(logging.INFO)
    done = _SimDevice().dispatch(0, kind="decode", bid=7, batch_size=3)

    assert done.wait(timeout=1)
    assert "SIM-DEVICE-COMPUTE" in caplog.text
    assert "kind=decode bid=7 batch_size=3" in caplog.text
