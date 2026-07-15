from sgl_jax.srt.disaggregation import mini_lb_helpers as helpers


def test_inject_bootstrap_fields_aligns_room_to_prefill_index(monkeypatch):
    monkeypatch.setattr(helpers, "generate_bootstrap_room", lambda: 10)

    req = helpers.inject_bootstrap_fields(
        {"prompt": "hello"},
        prefill_server="http://prefill-1:10000",
        bootstrap_port=8998,
        prefill_index=2,
        prefill_count=3,
    )

    assert req["bootstrap_room"] == 11
    assert req["bootstrap_room"] % 3 == 2


def test_inject_bootstrap_fields_keeps_batched_rooms_on_same_prefill(monkeypatch):
    monkeypatch.setattr(helpers, "generate_bootstrap_room", lambda: 10)

    req = helpers.inject_bootstrap_fields(
        {"text": ["a", "b", "c"]},
        prefill_server="http://prefill-1:10000",
        bootstrap_port=8998,
        prefill_index=2,
        prefill_count=3,
    )

    assert req["bootstrap_room"] == [11, 14, 17]
    assert [room % 3 for room in req["bootstrap_room"]] == [2, 2, 2]
