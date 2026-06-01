"""PD end-to-end test matrix (Stage 4 acceptance).

Each test in this package is a standalone Python script that:
  1. Parses a shared topology spec from CLI / env (P URLs, D URLs,
     bootstrap URL, optional shared secret).
  2. Exercises one capability area (multi-pd topology, byte-equal
     correctness, long-prompt KV transfer, concurrency, chaos,
     orthogonal-feature combinations).
  3. Prints a single-line `RESULT: PASS|FAIL ...` to stdout and exits
     0 on PASS / 1 on FAIL, so a shell driver can chain them.
  4. Writes a JSON summary to ``--out`` (default
     ``/tmp/pd_e2e_<test>.json``) so subsequent CI dashboards can
     pick it up.

These are operator scripts — they require a deployed PD topology and
do NOT run in unit-test CI. The single shell driver in
``scripts/run_pd_e2e_matrix.sh`` runs them in order against a
configured deployment, and the table in
``docs/operations/pd_e2e_matrix.md`` lists the last green commit per
test.

Why not pytest? These tests bind to a live multi-pod deployment and
their failure modes (5xx storms, KV-corrupt outputs, registry
drift) are not test-failure shapes pytest is good at surfacing —
the per-script JSON + decision-tree-in-runbook idiom is what an
SRE actually wants on call.
"""
