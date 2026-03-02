# Task: Fix ModuleNotFoundError: No module named 'datasets'

## Status
- [x] Identified missing dependency `datasets` in `python/pyproject.toml`.
- [ ] Add `datasets` to `python/pyproject.toml`.
- [ ] Verify fix by running `test_flashattention.py` in a skypilot cluster.

## Investigation
- Error occurred when running `python/sgl_jax/test/test_flashattention.py`.
- `sgl_jax/bench_serving.py` imports `load_dataset` from `datasets`.
- `python/pyproject.toml` does not list `datasets` in its dependencies.

## Plan
1. Update `python/pyproject.toml` to include `datasets`.
2. Sync the environment (or rely on the CI/remote environment to have it installed).
3. Run the test on a remote cluster using `exec-remote`.
