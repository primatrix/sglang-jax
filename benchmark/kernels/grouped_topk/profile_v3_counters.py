"""HW perf-counter profile of grouped_topk_pallas_v3: VPU/Scalar/XLU utilization + per-scope cost.
Uses v7 periodic counter sampling; prints discovery + breakdown to stdout (GCS not needed)."""
import os
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + " --xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
).strip()
import collections, glob, gzip, json, subprocess  # noqa: E402
import jax, jax.numpy as jnp  # noqa: E402
from sgl_jax.srt.kernels.grouped_topk.v2.kernel3 import grouped_topk_pallas_v3  # noqa: E402

T, E, G, Gtop, k = 16384, 256, 8, 4, 8
TR = os.path.join(os.environ.get("OUT", "/tmp/cnt"), "xprof")
SCOPES = ["bias_add", "group_top2", "group_select", "expert_mask", "final_select"]

lg = jax.device_put(jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(0), (T, E), jnp.float32)))
b = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (E,), jnp.float32) * 0.1)
fn = jax.jit(lambda l, bb: grouped_topk_pallas_v3(l, bb, num_expert_group=G, topk_group=Gtop, topk=k))
for _ in range(5):
    jax.block_until_ready(fn(lg, b))

opts = jax.profiler.ProfileOptions()
try:
    opts.advanced_configuration = {
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": (
            "interval_us:1 scaling:0 counter_size_bits:1 "
            "indices:1 indices:3 indices:4 indices:10 indices:11 indices:31 indices:32 "
            "indices:33 indices:34 indices:35 indices:37 indices:38 indices:56 indices:57 "
            "indices:58 indices:73 indices:74 indices:75 indices:105"
        ),
        "num_tensor_cores_to_trace_per_device": 1,
    }
    print("counter sampling configured")
except Exception as e:  # noqa: BLE001
    print("counter config FAILED:", e)

os.makedirs(TR, exist_ok=True)
with jax.profiler.trace(TR, profiler_options=opts):
    for i in range(10):
        with jax.profiler.StepTraceAnnotation("s", step_num=i):
            jax.block_until_ready(fn(lg, b))

latest = max(glob.glob(os.path.join(TR, "plugins", "profile", "*")), key=os.path.getmtime)
print("trace files:", [os.path.basename(x) for x in glob.glob(latest + "/*")])
evs = []
for tf in sorted(glob.glob(latest + "/*.trace.json.gz")):
    evs += json.load(gzip.open(tf)).get("traceEvents", [])
pn, tn = {}, {}
for e in evs:
    if e.get("ph") == "M":
        a = e.get("args", {})
        if e["name"] == "process_name": pn[e["pid"]] = a.get("name", "")
        if e["name"] == "thread_name": tn[(e["pid"], e["tid"])] = a.get("name", "")

print("\n=== TPU-device thread/track names ===")
for (p, t), nm in sorted(tn.items(), key=lambda x: x[1]):
    if "TPU" in pn.get(p, ""):
        print(f"  [{pn[p]}] {nm}")

print("\n=== phase counts ===", dict(collections.Counter(e.get("ph") for e in evs)))
cser = collections.Counter(e.get("name", "") for e in evs if e.get("ph") == "C")
print("=== counter(C) series ===")
for nm, c in cser.most_common(40):
    print(f"  {c:5d}  {nm}")
print("=== sample C events (name,args) ===")
shown = 0
for e in evs:
    if e.get("ph") == "C" and shown < 25:
        print("  ", e.get("name"), e.get("args")); shown += 1

# per-scope device time (region trace)
byname = collections.Counter()
for e in evs:
    if e.get("ph") == "X" and "dur" in e and pn.get(e.get("pid")) == "/device:TPU:0":
        byname[e.get("name", "")] += e["dur"]
kern = sum(v for n, v in byname.items() if "grouped-topk-v3" in n)
print(f"\n=== per-scope device time (kernel custom-call = {kern/10:.2f}us/iter) ===")
for sc in SCOPES:
    t = sum(v for n, v in byname.items() if sc in n)
    print(f"  {sc:14s} {t/10:8.3f} us  {100*t/kern:5.1f}%")

# grep xplane for readable counter/util names
xp = glob.glob(latest + "/*.xplane.pb")
if xp:
    try:
        out = subprocess.run(["strings", xp[0]], capture_output=True, text=True, timeout=60).stdout
        hits = sorted({l for l in out.splitlines()
                       if any(x.lower() in l.lower() for x in
                              ("vpu", "scalar", "xlu", "vector", "utiliz", "flop", "occup", "duty",
                               "active", "cycles", "perf_counter", "mxu"))
                       and 3 < len(l) < 60})
        print(f"\n=== xplane readable counter-ish strings ({len(hits)}) ===")
        for h in hits[:60]:
            print("  ", h)
    except Exception as e:  # noqa: BLE001
        print("strings failed:", e)
print("=== counters profile exit: 0 ===")
