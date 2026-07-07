"""Sweep v3 `unroll` (final-select fori_loop factor): device time, and (with --counters) the
Vector-ALU utilization + register spills/fills that motivate it."""
import os
if os.environ.get("V3_COUNTERS") == "1":
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "")
        + " --xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
    ).strip()
import argparse, glob, gzip, json, statistics, time  # noqa: E402
import jax, jax.numpy as jnp  # noqa: E402
from sgl_jax.srt.kernels.grouped_topk.v2.kernel3 import grouped_topk_pallas_v3  # noqa: E402

E, G, Gtop, k = 256, 8, 4, 8
TR = os.path.join(os.environ.get("OUT", "/tmp/sweep"), "xprof")

def _evs(rt):
    d = max(glob.glob(os.path.join(rt, "plugins", "profile", "*")), key=os.path.getmtime)
    evs = []
    for tf in sorted(glob.glob(d + "/*.trace.json.gz")):
        evs += json.load(gzip.open(tf)).get("traceEvents", [])
    return evs

def _maps(evs):
    pn, tn = {}, {}
    for e in evs:
        if e.get("ph") == "M":
            a = e.get("args", {})
            if e["name"] == "process_name": pn[e["pid"]] = a.get("name", "")
            if e["name"] == "thread_name": tn[(e["pid"], e["tid"])] = a.get("name", "")
    return pn, tn

def _module_us(evs):
    pn, tn = _maps(evs)
    d = [e["dur"] for e in evs if e.get("ph") == "X" and "dur" in e
         and pn.get(e["pid"]) == "/device:TPU:0" and tn.get((e["pid"], e["tid"])) == "XLA Modules"]
    return statistics.median(d) if d else float("nan")

def _counters(evs):
    _, tn = _maps(evs)
    out = {}
    for e in evs:
        if e.get("ph") == "X" and tn.get((e.get("pid"), e.get("tid"))) == "_counters_":
            a = e.get("args", {}); nm = e.get("name", "")
            for key in ("% util", "fills", "spills"):
                if key in a:
                    out[nm] = float(a[key]); break
    return out

def _trace(fn, lg, b, tag, opts=None, iters=20):
    for _ in range(5):
        jax.block_until_ready(fn(lg, b))
    rt = os.path.join(TR, f"{tag}_{int(time.time()*1000)}"); os.makedirs(rt, exist_ok=True)
    ctx = jax.profiler.trace(rt, profiler_options=opts) if opts else jax.profiler.trace(rt)
    with ctx:
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation("s", step_num=i):
                jax.block_until_ready(fn(lg, b))
    return _evs(rt)

def _counter_opts():
    opts = jax.profiler.ProfileOptions()
    opts.advanced_configuration = {
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": "interval_us:1 scaling:0 counter_size_bits:1",
        "num_tensor_cores_to_trace_per_device": 1,
    }
    return opts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="4096,8192,16384")
    ap.add_argument("--unroll", default="1,2,4,8")
    ap.add_argument("--counters", action="store_true")
    a = ap.parse_args()
    Ts = [int(x) for x in a.T.split(",")]
    Us = [int(x) for x in a.unroll.split(",")]
    print(f"device={jax.devices()[0].device_kind} counters={a.counters}")
    if not a.counters:
        print(f"\n{'T':>6} " + " ".join(f"u={u:>2}(us)" for u in Us))
        for T in Ts:
            lg = jax.device_put(jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(T), (T, E), jnp.float32)))
            b = jax.device_put(jax.random.normal(jax.random.PRNGKey(9), (E,), jnp.float32) * 0.1)
            row = []
            for u in Us:
                try:
                    fn = jax.jit(lambda l, bb, u=u: grouped_topk_pallas_v3(
                        l, bb, num_expert_group=G, topk_group=Gtop, topk=k, unroll=u))
                    row.append(f"{_module_us(_trace(fn, lg, b, f't{T}u{u}')):8.2f}")
                except Exception as e:  # noqa: BLE001
                    row.append(f"{('ERR:'+type(e).__name__):>8}")
            print(f"{T:>6} " + " ".join(row))
    else:
        T = 16384
        lg = jax.device_put(jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(T), (T, E), jnp.float32)))
        b = jax.device_put(jax.random.normal(jax.random.PRNGKey(9), (E,), jnp.float32) * 0.1)
        print(f"\ncounters @ T={T}: {'unroll':>6} {'VecALU%':>8} {'VecLd%':>7} {'VecSt%':>7} {'fills':>8} {'spills':>8}")
        for u in Us:
            fn = jax.jit(lambda l, bb, u=u: grouped_topk_pallas_v3(
                l, bb, num_expert_group=G, topk_group=Gtop, topk=k, unroll=u))
            c = _counters(_trace(fn, lg, b, f"cnt_u{u}", opts=_counter_opts(), iters=10))
            print(f"       {u:>6} {c.get('Vector ALU',float('nan')):8.1f} {c.get('Vector Load',float('nan')):7.1f} "
                  f"{c.get('Vector Store',float('nan')):7.1f} {c.get('Vector Fills',float('nan')):8.0f} {c.get('Vector Spills',float('nan')):8.0f}")

if __name__ == "__main__":
    main()
