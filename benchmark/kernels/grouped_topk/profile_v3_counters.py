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

print("\n=== all process names ===", sorted(set(pn.values())))
print("=== all thread/track names ===")
for (p, t), nm in sorted(tn.items(), key=lambda x: x[1]):
    print(f"  [{pn.get(p,'')}] {nm}")

print("\n=== phase counts ===", dict(collections.Counter(e.get("ph") for e in evs)))
cser = collections.Counter(e.get("name", "") for e in evs if e.get("ph") == "C")
print("=== counter(C) series ===")
for nm, c in cser.most_common(40):
    print(f"  {c:5d}  {nm}")
shown = 0
for e in evs:
    if e.get("ph") == "C" and shown < 25:
        print("   C:", e.get("name"), e.get("args")); shown += 1

# per-scope device time (region trace), crash-safe
byname = collections.Counter()
for e in evs:
    if e.get("ph") == "X" and "dur" in e and pn.get(e.get("pid")) == "/device:TPU:0":
        byname[e.get("name", "")] += e["dur"]
kern = sum(v for n, v in byname.items() if "grouped-topk-v3" in n)
print(f"\n=== per-scope device time (kernel custom-call = {kern/10:.2f}us/iter) ===")
if kern > 0:
    for sc in SCOPES:
        t = sum(v for n, v in byname.items() if sc in n)
        print(f"  {sc:14s} {t/10:8.3f} us  {100*t/kern:5.1f}%")
else:
    print("  (no device events in trace.json.gz for this counter run)")

# --- duty cycle: merged busy-time per track vs wall (XLA Modules) ---
def _merged_busy(track):
    ivs = sorted(((e["ts"], e["ts"] + e["dur"]) for e in evs
                  if e.get("ph") == "X" and "dur" in e
                  and pn.get(e.get("pid")) == "/device:TPU:0"
                  and tn.get((e.get("pid"), e.get("tid"))) == track), key=lambda x: x[0])
    busy = 0; cs = ce = None
    for s, en in ivs:
        if ce is None or s > ce:
            if ce is not None: busy += ce - cs
            cs, ce = s, en
        else:
            ce = max(ce, en)
    if ce is not None: busy += ce - cs
    return busy

wall = _merged_busy("XLA Modules")
print(f"\n=== track merged-busy vs wall (per iter; wall=XLA Modules={wall/10:.2f}us) ===")
for tk in ["XLA Modules", "XLA Ops", "Tensor Core", "TC Overlay", "Async XLA Ops"]:
    bd = _merged_busy(tk)
    duty = 100 * bd / wall if wall else float("nan")
    print(f"  {tk:16s} busy={bd/10:8.2f}us  duty={duty:6.1f}%")

# --- dump counter-track events (values sampled by periodic counters) ---
for tk in ["_counters_", "counters_0"]:
    cvs = [e for e in evs if tn.get((e.get("pid"), e.get("tid"))) == tk]
    print(f"\n=== track '{tk}': {len(cvs)} events ===")
    seen = {}
    for e in cvs:
        nm = e.get("name", "")
        if nm not in seen:
            seen[nm] = (e.get("ph"), e.get("args"))
    for nm, (ph, args) in list(seen.items())[:30]:
        print(f"   ph={ph} name={nm} args={args}")

# grep xplane for readable counter/util names
for xp in glob.glob(latest + "/*.xplane.pb"):
    print(f"\n=== xplane {os.path.basename(xp)} ({os.path.getsize(xp)/1e6:.1f} MB) readable strings ===")
    try:
        out = subprocess.run(["strings", "-n", "4", xp], capture_output=True, text=True, timeout=120).stdout
        hits = sorted({l.strip() for l in out.splitlines()
                       if any(x.lower() in l.lower() for x in
                              ("vpu", "scalar core", "xlu", "vector unit", "utiliz", "flop", "occup",
                               "duty", "vpu active", "mxu", "vector active"))
                       and 3 < len(l) < 70})
        for h in hits[:60]:
            print("   ", h)
    except Exception as e:  # noqa: BLE001
        print("   strings failed:", e)
print("=== counters profile exit: 0 ===")
