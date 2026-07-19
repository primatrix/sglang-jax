from sgl_jax.srt.managers.scheduler_profiler_mixing import _StageBasedTrigger


def test_stage_trigger_captures_exact_configured_step_count_per_stage():
    captured = {"prefill": 0, "decode": 0}
    active_stage = None

    def on_start(*, stage):
        nonlocal active_stage
        active_stage = stage

    def on_stop():
        nonlocal active_stage
        active_stage = None

    trigger = _StageBasedTrigger(on_start=on_start, on_stop=on_stop)
    trigger.configure(num_steps=4, interesting_stages=["prefill", "decode"])

    for stage in ["prefill"] * 6 + ["decode"] * 6:
        trigger.step(stage)
        if active_stage is not None:
            captured[active_stage] += 1

    assert captured == {"prefill": 4, "decode": 4}
    assert trigger.is_configured is False


def test_stage_trigger_starts_decode_on_early_prefill_transition():
    captured = {"prefill": 0, "decode": 0}
    active_stage = None

    def on_start(*, stage):
        nonlocal active_stage
        active_stage = stage

    def on_stop():
        nonlocal active_stage
        active_stage = None

    trigger = _StageBasedTrigger(on_start=on_start, on_stop=on_stop)
    trigger.configure(num_steps=4, interesting_stages=["prefill", "decode"])

    for stage in ["prefill"] * 2 + ["decode"] * 5:
        trigger.step(stage)
        if active_stage is not None:
            captured[active_stage] += 1

    assert captured == {"prefill": 2, "decode": 4}
    assert trigger.is_configured is False
