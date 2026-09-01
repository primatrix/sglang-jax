from sgl_jax.bench_serving import _is_multi_turn_prompt, gen_mm_prompt


def test_multi_turn_prompt_distinguishes_text_turns_from_token_ids():
    assert _is_multi_turn_prompt(["first question", "follow-up question"])
    assert not _is_multi_turn_prompt([101, 102, 103])
    assert not _is_multi_turn_prompt("single-turn prompt")


def test_multimodal_random_prompt_excludes_special_tokens():
    class Tokenizer:
        all_special_ids = [2, 3]

        @staticmethod
        def get_vocab():
            return {"text": 1, "image": 2, "video": 3, "extra_image_pad": 4}

        @staticmethod
        def decode(token_ids):
            assert set(token_ids) == {1}
            return "text"

    assert gen_mm_prompt(Tokenizer(), image_pad_id=4, token_num=16) == "text"
