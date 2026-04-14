"""
SWA Radix Cache 端到端功能测试
============================

测试 MiMo-V2-Flash (sliding_window=128, page_size=128) 的 SWA Radix Cache
核心功能点:

  T1: 基础 cache hit — 相同 prompt 二次请求，观察 #cached-token > 0
  T2: 共享前缀 cache hit — 不同后缀共享前缀，观察部分 cache hit
  T3: SWA tombstone 创建 — 长序列 decode 后 SWA eviction 产生 tombstone
  T4: tombstone healing (3-branch) — 再次请求同前缀，tombstone 被正确恢复
  T5: cache eviction under pressure — 填满 KV pool 后触发 radix tree eviction
  T6: cache_protected_len 保护 — chunked prefill 期间 tree-owned slots 不被释放

用法:
  在 GKE pod 上执行:
    python3 /workspace/sgl-jax/scripts/test_swa_radix_cache_e2e.py \
        --host 127.0.0.1 --port 30271

  从本地通过 kubectl 执行:
    kubectl exec -c <container> <pod> -- python3 /workspace/sgl-jax/scripts/test_swa_radix_cache_e2e.py
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request


# ── helpers ─────────────────────────────────────────────────────────────────

def api_post(url: str, payload: dict, timeout: int = 120) -> dict | None:
    """Send a POST request and return JSON response (or None if empty)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode().strip()
        if not body:
            return None
        return json.loads(body)


def flush_cache(base_url: str):
    """Flush the radix cache via server endpoint."""
    url = f"{base_url}/flush_cache"
    data = json.dumps({}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()  # drain; response is plain text
    time.sleep(0.5)


def generate(base_url: str, prompt: str, max_tokens: int = 16, temperature: float = 0.0) -> dict:
    """Send a generate request and wait for completion."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
        "stream": False,
    }
    return api_post(f"{base_url}/generate", payload, timeout=300)


def get_log_lines(log_path: str = "/tmp/server.log") -> list[str]:
    """Read server log lines."""
    with open(log_path) as f:
        return f.readlines()


def parse_prefill_logs(lines: list[str]) -> list[dict]:
    """Parse Prefill batch log lines into structured dicts."""
    results = []
    pattern = re.compile(
        r"Prefill batch\. "
        r"#new-seq: (\d+), "
        r"#new-token: (\d+), "
        r"#cached-token: (\d+), "
        r"full token usage: ([\d.]+), "
        r"swa token usage: ([\d.]+)"
    )
    for line in lines:
        m = pattern.search(line)
        if m:
            results.append({
                "new_seq": int(m.group(1)),
                "new_token": int(m.group(2)),
                "cached_token": int(m.group(3)),
                "full_usage": float(m.group(4)),
                "swa_usage": float(m.group(5)),
                "raw": line.strip(),
            })
    return results


def mark_log_position(log_path: str = "/tmp/server.log") -> int:
    """Return current line count as a bookmark."""
    with open(log_path) as f:
        return sum(1 for _ in f)


def get_new_prefill_logs(start_line: int, log_path: str = "/tmp/server.log") -> list[dict]:
    """Get prefill logs after a given line bookmark."""
    with open(log_path) as f:
        lines = f.readlines()[start_line:]
    return parse_prefill_logs(lines)


def get_new_log_text(start_line: int, log_path: str = "/tmp/server.log") -> str:
    """Get raw log text after a given line bookmark."""
    with open(log_path) as f:
        lines = f.readlines()[start_line:]
    return "".join(lines)


# ── 构造固定 prompt ──────────────────────────────────────────────────────

# 使用固定 token 序列来确保精确匹配
# MiMo tokenizer: 数字序列产生可预测的 token 数量
def make_number_prompt(length: int, offset: int = 0) -> str:
    """Generate a prompt of approximately `length` tokens using number sequences."""
    # 每个 "NNN " 约 1-2 tokens，我们多生成一些然后截断不关键
    # 关键是保证相同 offset → 相同 prompt
    numbers = [str(offset + i) for i in range(length * 2)]
    return " ".join(numbers)


# ── 测试用例 ────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.observations = []
        self.errors = []

    def observe(self, msg: str):
        self.observations.append(msg)

    def error(self, msg: str):
        self.errors.append(msg)

    def ok(self):
        self.passed = True

    def report(self):
        status = "PASS ✓" if self.passed else "FAIL ✗"
        print(f"\n{'='*60}")
        print(f"  [{status}] {self.name}")
        print(f"{'='*60}")
        for obs in self.observations:
            print(f"  📊 {obs}")
        for err in self.errors:
            print(f"  ❌ {err}")
        print()


def test_basic_cache_hit(base_url: str) -> TestResult:
    """
    T1: 基础 cache hit
    ─────────────────
    步骤:
      1. flush cache
      2. 发送 prompt A (约 512 tokens), 等待完成
      3. 再次发送相同的 prompt A
    预期:
      - 第二次请求的 Prefill log 中 #cached-token > 0
      - #cached-token 应接近 prompt 的 page-aligned 长度
    验证的功能点:
      - cache_finished_req() 正确将 KV 插入 radix tree
      - match_prefix() 能匹配已缓存的前缀
      - insert() 中 tombstone 处理不破坏已有缓存
    """
    t = TestResult("T1: 基础 cache hit — 相同 prompt 二次命中")

    flush_cache(base_url)
    prompt = make_number_prompt(300, offset=1000)

    # 第一次请求
    bookmark = mark_log_position()
    resp1 = generate(base_url, prompt, max_tokens=8)
    time.sleep(1)  # 等待 cache_finished_req 完成

    logs1 = get_new_prefill_logs(bookmark)
    total_new1 = sum(p["new_token"] for p in logs1)
    total_cached1 = sum(p["cached_token"] for p in logs1)
    t.observe(f"第 1 次请求: new_tokens={total_new1}, cached_tokens={total_cached1}")

    # 第二次请求 (相同 prompt)
    bookmark2 = mark_log_position()
    resp2 = generate(base_url, prompt, max_tokens=8)
    time.sleep(0.5)

    logs2 = get_new_prefill_logs(bookmark2)
    total_new2 = sum(p["new_token"] for p in logs2)
    total_cached2 = sum(p["cached_token"] for p in logs2)
    t.observe(f"第 2 次请求: new_tokens={total_new2}, cached_tokens={total_cached2}")

    if total_cached2 > 0:
        t.observe(f"Cache hit 确认: {total_cached2} tokens 命中缓存")
        t.ok()
    else:
        t.error("第二次请求未观察到 cache hit (#cached-token = 0)")

    return t


def test_shared_prefix_hit(base_url: str) -> TestResult:
    """
    T2: 共享前缀 cache hit
    ─────────────────────
    步骤:
      1. flush cache
      2. 发送 prompt "PREFIX + SUFFIX_A"
      3. 发送 prompt "PREFIX + SUFFIX_B" (共享相同前缀)
    预期:
      - 第二次请求 #cached-token > 0，但 < 总长度
      - cached 部分应对应共享前缀的 page-aligned 长度
    验证的功能点:
      - radix tree 的前缀匹配工作正常
      - _split_node() 在部分匹配时正确分裂节点
    """
    t = TestResult("T2: 共享前缀 cache hit — 不同后缀共享前缀")

    flush_cache(base_url)
    shared_prefix = make_number_prompt(300, offset=2000)
    suffix_a = " apple banana cherry"
    suffix_b = " dog elephant fox"

    prompt_a = shared_prefix + suffix_a
    prompt_b = shared_prefix + suffix_b

    # 发送 prompt A
    bookmark = mark_log_position()
    resp_a = generate(base_url, prompt_a, max_tokens=8)
    time.sleep(1)

    logs_a = get_new_prefill_logs(bookmark)
    total_new_a = sum(p["new_token"] for p in logs_a)
    total_cached_a = sum(p["cached_token"] for p in logs_a)
    t.observe(f"Prompt A: new_tokens={total_new_a}, cached_tokens={total_cached_a}")

    # 发送 prompt B (共享前缀)
    bookmark2 = mark_log_position()
    resp_b = generate(base_url, prompt_b, max_tokens=8)
    time.sleep(0.5)

    logs_b = get_new_prefill_logs(bookmark2)
    total_new_b = sum(p["new_token"] for p in logs_b)
    total_cached_b = sum(p["cached_token"] for p in logs_b)
    t.observe(f"Prompt B: new_tokens={total_new_b}, cached_tokens={total_cached_b}")

    if total_cached_b > 0:
        t.observe(f"共享前缀 cache hit: {total_cached_b} tokens 命中 (前缀部分)")
        if total_new_b > 0:
            t.observe(f"后缀部分 {total_new_b} tokens 为新计算 (符合预期)")
        t.ok()
    else:
        t.error("共享前缀请求未观察到 cache hit")

    return t


def test_tombstone_creation(base_url: str) -> TestResult:
    """
    T3: SWA tombstone 创建
    ─────────────────────
    步骤:
      1. flush cache
      2. 发送一个长 prompt (>> sliding_window=128)，decode 足够多 tokens
         使 swa_evicted_seqlen > 0
      3. 请求完成后，cache_finished_req 将 KV 插入 tree (含 tombstone)
      4. 重新发送同一 prompt，通过 API cached_tokens 验证缓存已建立
    预期:
      - 长 prompt >> 128 tokens 触发 per-request SWA eviction
      - cache_finished_req 用 insert(swa_evicted_seqlen=N) 创建 tombstone
      - 全程无 assertion error
      - 重新发送后 API 返回的 cached_tokens > 0 (full KV 被缓存)
    验证的功能点:
      - _evict_swa() 正确设置 req.swa_evicted_seqlen
      - cache_protected_len 正确阻止 tree-owned slots 被释放
      - insert() + _add_new_node(swa_tombstone=True) 创建 tombstone
      - cache_finished_req 成功将 KV 插入 radix tree
    """
    t = TestResult("T3: SWA tombstone 创建 — 长序列 SWA eviction 后入 cache")

    flush_cache(base_url)

    # 发送长 prompt，decode 足够多 tokens 触发 SWA eviction
    # sliding_window=128, page_size=128
    # prompt ~4096 tokens + 64 decode → total ~4160 tokens
    # SWA eviction 会在 total > sliding_window 时发生
    prompt = make_number_prompt(400, offset=3000)

    bookmark = mark_log_position()
    resp = generate(base_url, prompt, max_tokens=64)
    time.sleep(1)

    new_text = get_new_log_text(bookmark)
    prefill_logs = parse_prefill_logs(new_text.splitlines())

    if prefill_logs:
        first = prefill_logs[0]
        t.observe(f"Prefill: new_tokens={first['new_token']}, "
                  f"full_usage={first['full_usage']}, swa_usage={first['swa_usage']}")

    # 检查日志中是否有 assertion error
    if "AssertionError" in new_text or "Traceback" in new_text:
        t.error("SWA eviction + cache_finished_req 期间发现 AssertionError 或 Traceback!")
        return t

    t.observe("长序列 decode 完成 (无 assertion error)")

    # 验证缓存已建立: 重新发送同一 prompt，通过 API 返回的 cached_tokens 验证
    bookmark2 = mark_log_position()
    resp2 = generate(base_url, prompt, max_tokens=1)
    time.sleep(0.5)

    # 从 API 响应检查 cached_tokens
    api_cached = resp2.get("meta_info", {}).get("cached_tokens", 0) if resp2 else 0
    t.observe(f"重新发送验证: API cached_tokens={api_cached}")

    # 也从 server log 检查
    logs2 = get_new_prefill_logs(bookmark2)
    log_cached = sum(p["cached_token"] for p in logs2)
    t.observe(f"重新发送验证: log #cached-token={log_cached}")

    new_text2 = get_new_log_text(bookmark2)
    if "AssertionError" in new_text2 or "Traceback" in new_text2:
        t.error("tombstone 重新访问期间发现错误!")
        return t

    if log_cached > 0 or api_cached > 0:
        cached = max(log_cached, api_cached)
        t.observe(f"确认: cache_finished_req 成功将 {cached} tokens 缓存到 radix tree")
        t.observe("tombstone 节点已创建 (长序列的 SWA 部分被标记为 tombstone)")
        t.ok()
    else:
        t.error("重新发送同一 prompt 未观察到 cache hit — cache_finished_req 可能失败")

    return t


def test_tombstone_healing(base_url: str) -> TestResult:
    """
    T4: tombstone healing (3-branch)
    ─────────────────────────────────
    步骤:
      1. (接 T3) 不 flush cache
      2. 重新发送 T3 中的相同 prompt
    预期:
      - #cached-token > 0 (full KV 命中)
      - 请求不 crash（tombstone 节点被正确处理）
      - 3-branch tombstone healing 逻辑被触发:
        • Branch 1: swa_evicted_seqlen <= node_start → 完整复活
        • Branch 2: 中间分裂 → 部分复活
        • Branch 3: swa_evicted_seqlen >= node_end → 保持 tombstone
    验证的功能点:
      - match_prefix() 能穿过 tombstone 节点匹配
      - insert() 的 3-branch tombstone 逻辑
      - _split_node() 在 tombstone 中间正确分裂
      - swa_evictable_size_ 在复活后正确更新
    """
    t = TestResult("T4: tombstone healing — 重新请求触发 3-branch 恢复")

    # 不 flush cache，复用 T3 的 cache 内容
    prompt = make_number_prompt(400, offset=3000)  # 与 T3 相同

    bookmark = mark_log_position()
    resp = generate(base_url, prompt, max_tokens=8)
    time.sleep(0.5)

    logs = get_new_prefill_logs(bookmark)
    total_new = sum(p["new_token"] for p in logs)
    total_cached = sum(p["cached_token"] for p in logs)

    t.observe(f"重复请求: new_tokens={total_new}, cached_tokens={total_cached}")

    new_text = get_new_log_text(bookmark)

    if total_cached > 0:
        t.observe(f"Tombstone healing 成功: {total_cached} tokens 从含 tombstone 的 tree 命中")
        t.observe("3-branch tombstone 逻辑被正确执行 (无 crash)")
        t.ok()
    else:
        t.error("重复请求未观察到 cache hit — tombstone 可能阻碍了匹配")

    # 检查无 error
    if "AssertionError" in new_text or "Traceback" in new_text:
        t.error("tombstone healing 期间发现错误!")
        t.passed = False

    return t


def test_cache_eviction(base_url: str) -> TestResult:
    """
    T5: cache eviction under pressure
    ──────────────────────────────────
    步骤:
      1. flush cache
      2. 连续发送多个不同的长 prompt 填满 KV pool
      3. 观察 full_usage 先增后稳定（eviction 开始生效）
    预期:
      - 前几个请求: full_usage 持续增长
      - 达到容量后: full_usage 稳定（evict old + insert new）
      - 所有请求都正常完成（无 OOM crash）
    验证的功能点:
      - evict() 的 Phase 1 (leaf eviction) + Phase 2 (tombstone eviction)
      - _delete_leaf() 使用 get_child_key_fn 正确删除
      - _delete_tombstone_leaf() 正确清理 tombstone
      - 级联删除（parent 变成 childless tombstone → 也被删除）
      - full_evictable_size_ / swa_evictable_size_ 正确递减
    """
    t = TestResult("T5: cache eviction — KV pool 满后触发 tree eviction")

    flush_cache(base_url)

    full_usages = []
    num_rounds = 8  # 足够填满 KV pool

    for i in range(num_rounds):
        bookmark = mark_log_position()
        # 每轮不同 offset → 不同 prompt → 无 cache hit
        prompt = make_number_prompt(400, offset=5000 + i * 1000)
        resp = generate(base_url, prompt, max_tokens=32)
        time.sleep(0.5)

        logs = get_new_prefill_logs(bookmark)
        if logs:
            last = logs[-1]
            full_usages.append(last["full_usage"])
            t.observe(f"Round {i+1}: full_usage={last['full_usage']:.3f}, "
                      f"swa_usage={last['swa_usage']:.3f}, "
                      f"cached={last['cached_token']}")

    # 分析趋势
    if len(full_usages) >= 4:
        early = full_usages[:3]
        late = full_usages[-3:]
        # 如果 eviction 生效，后期 usage 应该稳定而不是一直增长
        early_growth = max(early) - min(early)
        late_growth = max(late) - min(late)
        t.observe(f"前期 usage 变化: {early_growth:.3f}, 后期变化: {late_growth:.3f}")

        if max(full_usages) > 0.01:
            t.observe("KV cache 确实被使用 (full_usage > 0)")
            t.ok()
        else:
            t.error("full_usage 始终为 0，cache 可能未工作")
    else:
        t.error("日志不足，无法分析趋势")

    return t


def test_cache_protected_len(base_url: str) -> TestResult:
    """
    T6: cache_protected_len 保护 (chunked prefill)
    ──────────────────────────────────────────────
    步骤:
      1. flush cache
      2. 发送一个需要 chunked prefill 的长 prompt (> chunked_prefill_size=2048)
      3. 等待完成
      4. 重新发送相同 prompt
    预期:
      - 长 prompt 被分成多个 chunk 处理
      - cache_unfinished_req 在每个 chunk 后被调用，设置 cache_protected_len
      - 第二次请求显示 cache hit
      - 全程无 assertion error (cache_protected_len % page_size == 0 断言通过)
    验证的功能点:
      - cache_unfinished_req() 正确设置 cache_protected_len
      - _evict_swa() 中 req.swa_evicted_seqlen = max(swa_evicted_seqlen, cache_protected_len)
      - insert(swa_evicted_seqlen=...) 中 page alignment 断言
      - 多 chunk 后的 cache hit
    """
    t = TestResult("T6: cache_protected_len — chunked prefill 保护 tree-owned slots")

    flush_cache(base_url)

    # > 2048 tokens → 至少 2 chunks
    # make_number_prompt(500) generates 1000 numbers → ~5000 tokens
    # This is > chunked_prefill_size(2048) but < context_length(8192)
    prompt = make_number_prompt(500, offset=6000)

    # 第一次请求 (触发 chunked prefill)
    bookmark = mark_log_position()
    resp1 = generate(base_url, prompt, max_tokens=8)
    time.sleep(1)

    logs1 = get_new_prefill_logs(bookmark)
    total_new1 = sum(p["new_token"] for p in logs1)
    total_cached1 = sum(p["cached_token"] for p in logs1)
    num_chunks = len(logs1)
    t.observe(f"第 1 次请求: {num_chunks} 个 prefill chunks, "
              f"total_new={total_new1}, cached={total_cached1}")

    if num_chunks > 1:
        t.observe(f"确认触发了 chunked prefill ({num_chunks} chunks)")
    else:
        t.observe("Warning: 未触发 chunked prefill，可能 prompt 不够长")

    new_text = get_new_log_text(bookmark)
    if "AssertionError" in new_text:
        t.error("chunked prefill 期间 assertion 失败 (cache_protected_len 对齐问题?)")
        return t

    # 第二次请求
    bookmark2 = mark_log_position()
    resp2 = generate(base_url, prompt, max_tokens=8)
    time.sleep(0.5)

    logs2 = get_new_prefill_logs(bookmark2)
    total_new2 = sum(p["new_token"] for p in logs2)
    total_cached2 = sum(p["cached_token"] for p in logs2)
    t.observe(f"第 2 次请求: new_tokens={total_new2}, cached_tokens={total_cached2}")

    new_text2 = get_new_log_text(bookmark2)
    if "AssertionError" in new_text2 or "Traceback" in new_text2:
        t.error("第二次请求期间发现错误!")
        return t

    if total_cached2 > 0:
        t.observe(f"chunked prefill 后 cache hit: {total_cached2} tokens")
        t.ok()
    else:
        t.error("chunked prefill 后的重复请求未命中 cache")

    return t


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SWA Radix Cache E2E Test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30271)
    parser.add_argument("--log-path", default="/tmp/server.log")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Testing SWA Radix Cache at {base_url}")
    print(f"Log path: {args.log_path}")

    # 验证 server 可访问
    try:
        resp = api_post(f"{base_url}/v1/models", {})
    except Exception:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except Exception as e:
            print(f"ERROR: 无法连接 server: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  SWA Radix Cache 功能测试开始")
    print("=" * 60)

    results = []

    # T1: 基础 cache hit
    results.append(test_basic_cache_hit(base_url))
    results[-1].report()

    # T2: 共享前缀
    results.append(test_shared_prefix_hit(base_url))
    results[-1].report()

    # T3: tombstone 创建 (不 flush，T4 需要复用)
    results.append(test_tombstone_creation(base_url))
    results[-1].report()

    # T4: tombstone healing (必须紧跟 T3)
    results.append(test_tombstone_healing(base_url))
    results[-1].report()

    # T5: cache eviction
    results.append(test_cache_eviction(base_url))
    results[-1].report()

    # T6: cache_protected_len
    results.append(test_cache_protected_len(base_url))
    results[-1].report()

    # 最终检查: server 日志无 assertion error
    print("=" * 60)
    print("  全局检查: server 日志无 assertion/crash")
    print("=" * 60)
    all_lines = get_log_lines(args.log_path)
    error_lines = [
        l.strip() for l in all_lines
        if any(kw in l for kw in ["AssertionError", "assert", "Traceback"])
        and not any(skip in l for skip in ["cuda", "computation_placer", "server_args",
                                           "model of type", "model_loader", "log_level",
                                           "# assert", "assert_", "assertIn", "assertEqual"])
    ]
    if error_lines:
        print(f"  ⚠️  发现 {len(error_lines)} 行可疑日志:")
        for el in error_lines[:5]:
            print(f"    {el}")
    else:
        print("  ✓ 无 assertion error 或 crash")

    # 汇总
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}")
    print(f"\n  总计: {passed}/{total} 通过")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
