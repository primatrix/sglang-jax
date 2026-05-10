FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1

ARG SOURCE_COMMIT=""

WORKDIR /opt/sglang-jax

COPY . .

ENV PYTHONPATH=/opt/sglang-jax:/opt/sglang-jax/python
ENV JAX_COMPILATION_CACHE_DIR=/tmp/jax_compilation_cache
ENV SGLANG_JAX_SOURCE_COMMIT=${SOURCE_COMMIT}

RUN mkdir -p /tmp/jax_compilation_cache /tmp/operator-artifact \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e "./python[tpu]" \
    && python3 -c 'import importlib.metadata as md; import benchmark.moe.calibration.bench_calibration as bc; print("bench_calibration_import=ok"); print("scenarios=" + ",".join(bc.SCENARIOS)); [print(f"{package}={md.version(package)}") if package in {dist.metadata["Name"] for dist in md.distributions()} else print(f"{package}=MISSING") for package in ("jax", "jaxlib", "libtpu", "sglang-jax")]'

CMD ["/bin/bash"]
