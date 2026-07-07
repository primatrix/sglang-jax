"""Probe: (A) does jax.lax.top_k lower inside a Pallas kernel? (B) device-time the XLA-level
jax.lax.top_k grouped-topk path vs v3, on the same TPU."""
import functools, glob, gzip, json, os, statistics, time
import jax, jax.numpy as jnp
import jax.experimental.pallas as pl
from sgl_jax.srt.kernels.grouped_topk.v2.kernel3 import grouped_topk_pallas_v3

G, Gtop, k, E = 8, 4, 8, 256
TR = os.environ.get("TR", "/tmp/probe/xprof")

# ---------- Part A: top_k INSIDE a Pallas kernel ----------
def _k_lastaxis(x_ref, o_ref):
    _, idx = jax.lax.top_k(x_ref[...], k)   # [bt,E] -> last-axis top_k
    o_ref[...] = idx.astype(jnp.int32)
def _k_axis0(x_ref, o_ref):
    xt = x_ref[...].T                        # [E,bt] -> [bt,E]
    _, idx = jax.lax.top_k(xt, k)
    o_ref[...] = idx.astype(jnp.int32)
def probeA():
    bt = 256
    x = jax.random.normal(jax.random.PRNGKey(0), (bt, E), jnp.float32)
    try:
        f = pl.pallas_call(_k_lastaxis, out_shape=jax.ShapeDtypeStruct((bt, k), jnp.int32))
        jax.block_until_ready(f(x)); print("A1) top_k inside Pallas [BT,E] last-axis: COMPILES")
    except Exception as e:
        print("A1) top_k inside Pallas [BT,E]: FAILS ->", type(e).__name__, str(e)[:140])
    xe = jax.random.normal(jax.random.PRNGKey(1), (E, bt), jnp.float32)
    try:
        f = pl.pallas_call(_k_axis0, out_shape=jax.ShapeDtypeStruct((bt, k), jnp.int32))
        jax.block_until_ready(f(xe)); print("A2) top_k inside Pallas [E,BT]+.T: COMPILES")
    except Exception as e:
        print("A2) top_k inside Pallas [E,BT]+.T: FAILS ->", type(e).__name__, str(e)[:140])

# ---------- Part B: XLA-level jax.lax.top_k grouped-topk (pure JAX) ----------
def jax_topk_grouped(rl, cb):
    rl = rl.astype(jnp.float32); n = rl.shape[0]
    s = rl + cb[None, :]; sg = s.reshape(n, G, -1)
    gs = jnp.sum(jax.lax.top_k(sg, k=2)[0], axis=-1)
    gi = jax.lax.top_k(gs, k=Gtop)[1]
    gm = jnp.clip(jax.nn.one_hot(gi, G).sum(axis=1), 0, 1)
    epg = rl.shape[-1] // G
    sm = jnp.broadcast_to(gm[..., None], (n, G, epg)).reshape(n, -1)
    tmp = jnp.where(sm, s, float("-inf"))
    w, ids = jax.lax.top_k(tmp, k=k)
    return jnp.take_along_axis(rl, ids, axis=1), ids

def _dev_ms(evs):
    pn, tn = {}, {}
    for e in evs:
        if e.get("ph") == "M":
            a = e.get("args", {})
            if e["name"] == "process_name": pn[e["pid"]] = a.get("name", "")
            if e["name"] == "thread_name": tn[(e["pid"], e["tid"])] = a.get("name", "")
    d = [e["dur"] for e in evs if e.get("ph")=="X" and "dur" in e
         and pn.get(e["pid"])=="/device:TPU:0" and tn.get((e["pid"],e["tid"]))=="XLA Modules"]
    return statistics.median(d)/1e3 if d else float("nan")

def timed(fn, lg, b, tag):
    for _ in range(5): jax.block_until_ready(fn(lg, b))
    rt = os.path.join(TR, f"{tag}_{int(time.time()*1000)}"); os.makedirs(rt, exist_ok=True)
    with jax.profiler.trace(rt):
        for i in range(30):
            with jax.profiler.StepTraceAnnotation("s", step_num=i):
                jax.block_until_ready(fn(lg, b))
    dirs = glob.glob(os.path.join(rt, "plugins", "profile", "*"))
    evs = []
    for tf in sorted(glob.glob(os.path.join(max(dirs, key=os.path.getmtime), "*.trace.json.gz"))):
        evs += json.load(gzip.open(tf)).get("traceEvents", [])
    return _dev_ms(evs) * 1e3  # us

def probeB():
    jtk = jax.jit(jax_topk_grouped)
    v3 = jax.jit(lambda l, b: grouped_topk_pallas_v3(l, b, num_expert_group=G, topk_group=Gtop, topk=k))
    print(f"\n{'T':>6} {'jax_topk(us)':>13} {'v3(us)':>9} {'jax/v3':>7}")
    for T in [4096, 8192, 16384]:
        lg = jax.device_put(jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(T), (T, E), jnp.float32)))
        b = jax.device_put(jax.random.normal(jax.random.PRNGKey(9), (E,), jnp.float32) * 0.1)
        a = timed(jtk, lg, b, f"jtk{T}"); c = timed(v3, lg, b, f"v3_{T}")
        print(f"{T:>6} {a:>13.2f} {c:>9.2f} {a/c:>6.2f}x")

print(f"device={jax.devices()[0].device_kind}")
probeA(); probeB()
print("=== probe exit: 0 ===")
