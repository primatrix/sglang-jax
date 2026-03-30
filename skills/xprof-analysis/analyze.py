#!/usr/bin/env python3
"""xprof Profiling Analyzer for Pallas Kernels on TPU v7.

Extracts actionable metrics from .xplane.pb profile data using the xprof Python API.

Usage:
    # Analyze a single profile
    python -m skills.xprof_analysis.analyze ./profile_r8_current

    # Compare two profiles
    python -m skills.xprof_analysis.analyze --compare ./profile_r6 ./profile_r8_current

    # Only show the kernel time breakdown
    python -m skills.xprof_analysis.analyze --ops-only ./profile_r8_current
"""

import argparse
import glob
import json
import os
import sys


def _get_xprof():
    """Import xprof with error handling."""
    try:
        from xprof.convert._pywrap_profiler_plugin import xspace_to_tools_data
        return xspace_to_tools_data
    except ImportError:
        print("ERROR: xprof not installed. pip install xprof", file=sys.stderr)
        sys.exit(1)


def find_xplane(profile_dir: str) -> str | None:
    """Find the .xplane.pb file in a profile directory."""
    if profile_dir.endswith('.xplane.pb') and os.path.isfile(profile_dir):
        return profile_dir
    for pattern in [
        f"{profile_dir}/plugins/profile/*/*.xplane.pb",
        f"{profile_dir}/**/*.xplane.pb",
    ]:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    return None


def _tool(xplane: str, tool_name: str, options: dict | None = None) -> dict | list | None:
    """Call xprof tool and return parsed JSON, or None on failure."""
    xspace_to_tools_data = _get_xprof()
    try:
        raw, ok = xspace_to_tools_data([xplane], tool_name, options or {})
        if ok and raw:
            return json.loads(raw)
    except Exception as e:
        print(f"  [warn] {tool_name}: {e}", file=sys.stderr)
    return None


def _parse_datatable(data) -> tuple[list[str], list[dict], dict]:
    """Parse a Google DataTable JSON into (columns, rows_as_dicts, properties)."""
    table = data[0] if isinstance(data, list) else data
    cols = [c['label'] for c in table['cols']]
    props = table.get('p', {})
    rows = []
    for row in table.get('rows', []):
        vals = {}
        for j, col in enumerate(cols):
            cell = row['c'][j] if j < len(row['c']) else {}
            vals[col] = cell.get('v') if cell else None
        rows.append(vals)
    return cols, rows, props


# ── Roofline Analysis ──────────────────────────────────────────────────────

def analyze_roofline(xplane: str) -> dict:
    """Extract per-op roofline data. Returns {meta, ops, kernel_time_us, total_time_us}."""
    data = _tool(xplane, 'roofline_model')
    if not data:
        return {}

    cols, rows, meta = _parse_datatable(data)

    # De-duplicate ops (roofline often has 2 steps, same op appears twice)
    seen = {}
    for r in rows:
        name = r.get('Operation', '')
        if name not in seen:
            seen[name] = r

    ops = sorted(seen.values(),
                 key=lambda x: x.get('Total self time (us)', 0),
                 reverse=True)

    # Find the Pallas kernel op (custom-call with "fused-moe" or similar)
    kernel_op = None
    for op in ops:
        name = op.get('Operation', '')
        if 'fused-moe' in name or 'pallas' in name.lower():
            kernel_op = op
            break

    kernel_time = kernel_op.get('Total self time (us)', 0) if kernel_op else 0
    total_time = sum(op.get('Total self time (us)', 0) for op in ops)

    return {
        'meta': meta,
        'ops': ops,
        'kernel_op': kernel_op,
        'kernel_time_us': kernel_time,
        'total_time_us': total_time,
    }


# ── Op Profile Tree ────────────────────────────────────────────────────────

def analyze_op_profile(xplane: str) -> dict:
    """Extract op profile tree data."""
    data = _tool(xplane, 'op_profile')
    if not data:
        return {}

    # Flatten the tree for the main program
    flat_ops = []
    by_prog = data.get('byProgramExcludeIdle', data.get('byProgram', {}))

    def walk(node, depth=0):
        metrics = node.get('metrics', {})
        raw_time = metrics.get('rawTime', 0)
        if raw_time > 0:
            flat_ops.append({
                'name': node.get('name', ''),
                'time_ps': raw_time,
                'time_us': raw_time / 1e6,
                'flops': metrics.get('rawFlops', 0),
                'bf16_flops': metrics.get('bf16Flops', 0),
                'bytes': metrics.get('rawBytesAccessedArray', []),
                'occurrences': metrics.get('occurrences', 0),
                'category': node.get('xla', {}).get('category', ''),
                'depth': depth,
            })
        for child in node.get('children', []):
            walk(child, depth + 1)

    for child in by_prog.get('children', []):
        walk(child, 0)

    flat_ops.sort(key=lambda x: x['time_ps'], reverse=True)
    return {'ops': flat_ops, 'device_type': data.get('deviceType', '')}


# ── Framework Op Stats ──────────────────────────────────────────────────────

def analyze_framework_ops(xplane: str) -> dict:
    """Extract framework-level op statistics."""
    data = _tool(xplane, 'framework_op_stats')
    if not data:
        return {}
    cols, rows, props = _parse_datatable(data)
    return {'ops': rows, 'props': props}


# ── Printing ────────────────────────────────────────────────────────────────

def print_roofline(result: dict, label: str = ""):
    """Pretty-print roofline analysis results."""
    if not result:
        print("  No roofline data available.")
        return

    meta = result['meta']
    ops = result['ops']
    kernel_time = result['kernel_time_us']
    total_time = result['total_time_us']

    if label:
        print(f"\n{'=' * 78}")
        print(f"  {label}")
        print(f"{'=' * 78}")

    print(f"\n  Device: {meta.get('device_type', 'Unknown')}")
    print(f"  Peak FLOP: {float(meta.get('peak_flop_rate', 0)):,.0f} GFLOP/s "
          f"({float(meta.get('peak_flop_rate', 0))/1e3:,.1f} TFLOP/s)")
    print(f"  Peak HBM BW: {meta.get('peak_hbm_bw', '?')} GiB/s")
    print(f"  Peak VMEM Read/Write: {meta.get('peak_vmem_read_bw', '?')} / "
          f"{meta.get('peak_vmem_write_bw', '?')} GiB/s")

    # Ridge points
    hbm_rp = float(meta.get('hbm_ridge_point', 0))
    vmem_r_rp = float(meta.get('vmem_read_ridge_point', 0))
    print(f"  Ridge Points: HBM={hbm_rp:.1f}, VMEM_R={vmem_r_rp:.1f} FLOP/Byte")

    print(f"\n  Total profile time: {total_time:,.1f} us")
    if kernel_time > 0:
        print(f"  Pallas kernel time: {kernel_time:,.1f} us "
              f"({kernel_time/total_time*100:.1f}% of total)")

    # Top ops table
    print(f"\n  {'#':<3} {'Self Time':>10} {'%':>6} {'HBM BW':>9} {'VMEM_R':>9} "
          f"{'OI':>7} {'Bound':>12} {'DMA%':>5}  Operation")
    print(f"  {'─' * 100}")

    for i, op in enumerate(ops[:15]):
        name = op.get('Operation', '')
        self_time = op.get('Total self time (us)', 0)
        pct = self_time / total_time * 100 if total_time > 0 else 0
        if self_time < 0.01:
            continue
        # Truncate long names
        if len(name) > 45:
            name = name[:42] + '...'
        print(f"  {i+1:<3} {self_time:>9.1f}u {pct:>5.1f}% "
              f"{op.get('HBM BW (GiB/s)', 0):>8.1f}G "
              f"{op.get('VMEM Read BW (GiB/s)', 0):>8.1f}G "
              f"{op.get('Operational Intensity (FLOP/Byte)', 0):>7.1f} "
              f"{op.get('Bound by', ''):>12} "
              f"{op.get('%time stalled by DMA', 0) or 0:>4.1f}%"
              f"  {name}")

    # Bound-by summary
    bound_summary = {}
    for op in ops:
        bound = op.get('Bound by', 'Unknown')
        time = op.get('Total self time (us)', 0)
        if time > 0:
            bound_summary[bound] = bound_summary.get(bound, 0) + time

    print(f"\n  Bottleneck Distribution:")
    for bound, time in sorted(bound_summary.items(), key=lambda x: -x[1]):
        pct = time / total_time * 100 if total_time > 0 else 0
        bar = '█' * int(pct / 2.5) + '░' * (40 - int(pct / 2.5))
        print(f"    {bound:<15} {time:>8.1f}u ({pct:>5.1f}%) |{bar}|")


def print_op_profile(result: dict, label: str = ""):
    """Pretty-print op profile tree."""
    if not result or not result.get('ops'):
        return

    if label:
        print(f"\n  Op Profile ({label}):")
    else:
        print(f"\n  Op Profile:")

    print(f"  {'Time(us)':>10} {'FLOP':>15} {'BF16 FLOP':>15}  Category / Name")
    print(f"  {'─' * 80}")

    for op in result['ops'][:20]:
        name = op['name'][:50]
        cat = op.get('category', '')
        if cat:
            cat = f"[{cat}] "
        print(f"  {op['time_us']:>10.1f} {op['flops']:>15,} {op['bf16_flops']:>15,}  {cat}{name}")


def print_comparison(baseline: dict, optimized: dict, b_label="BASELINE", o_label="OPTIMIZED"):
    """Print comparison between two roofline results."""
    print(f"\n{'=' * 78}")
    print(f"  COMPARISON: {b_label} vs {o_label}")
    print(f"{'=' * 78}")

    b_ops = {op['Operation']: op for op in baseline.get('ops', []) if op.get('Operation')}
    o_ops = {op['Operation']: op for op in optimized.get('ops', []) if op.get('Operation')}

    # Kernel time comparison
    b_kt = baseline.get('kernel_time_us', 0)
    o_kt = optimized.get('kernel_time_us', 0)
    b_tt = baseline.get('total_time_us', 0)
    o_tt = optimized.get('total_time_us', 0)

    if b_kt > 0 and o_kt > 0:
        delta = o_kt - b_kt
        delta_pct = delta / b_kt * 100
        print(f"\n  Pallas Kernel Time:")
        print(f"    {b_label}: {b_kt:>8.1f} us")
        print(f"    {o_label}: {o_kt:>8.1f} us")
        print(f"    Delta:    {delta:>+8.1f} us ({delta_pct:>+.1f}%)")
        if delta < 0:
            print(f"    → Kernel is {-delta:.1f}us ({-delta_pct:.1f}%) FASTER")
        else:
            print(f"    → Kernel is {delta:.1f}us ({delta_pct:.1f}%) SLOWER")

    if b_tt > 0 and o_tt > 0:
        delta = o_tt - b_tt
        delta_pct = delta / b_tt * 100
        print(f"\n  Total Profile Time:")
        print(f"    {b_label}: {b_tt:>8.1f} us")
        print(f"    {o_label}: {o_tt:>8.1f} us")
        print(f"    Delta:    {delta:>+8.1f} us ({delta_pct:>+.1f}%)")

    # Per-op comparison for top ops
    all_names = []
    for name in b_ops:
        if name not in all_names:
            all_names.append(name)
    for name in o_ops:
        if name not in all_names:
            all_names.append(name)

    # Sort by baseline time
    all_names.sort(key=lambda n: b_ops.get(n, {}).get('Total self time (us)', 0), reverse=True)

    print(f"\n  Per-Op Comparison (top ops):")
    print(f"  {'Operation':<45} {b_label:>10} {o_label:>10} {'Delta':>10} {'%':>7}")
    print(f"  {'─' * 85}")

    for name in all_names[:12]:
        b_time = b_ops.get(name, {}).get('Total self time (us)', 0)
        o_time = o_ops.get(name, {}).get('Total self time (us)', 0)
        if b_time < 0.1 and o_time < 0.1:
            continue
        delta = o_time - b_time
        delta_pct = delta / b_time * 100 if b_time > 0 else float('inf')
        short_name = name[:45] if len(name) <= 45 else name[:42] + '...'
        marker = '✓' if delta < -0.5 else ('✗' if delta > 0.5 else ' ')
        print(f"  {short_name:<45} {b_time:>9.1f}u {o_time:>9.1f}u "
              f"{delta:>+9.1f}u {delta_pct:>+6.1f}% {marker}")


# ── Main ────────────────────────────────────────────────────────────────────

def analyze(profile_dir: str, ops_only: bool = False):
    """Full analysis of a single profile."""
    xplane = find_xplane(profile_dir)
    if not xplane:
        print(f"ERROR: No .xplane.pb found in {profile_dir}", file=sys.stderr)
        return None

    label = os.path.basename(profile_dir.rstrip('/'))
    print(f"\n  Analyzing: {profile_dir}")
    print(f"  XPlane: {xplane}")

    # Roofline (always)
    roofline = analyze_roofline(xplane)
    print_roofline(roofline, label)

    if not ops_only:
        # Op profile tree
        op_prof = analyze_op_profile(xplane)
        print_op_profile(op_prof, label)

    return roofline


def compare(baseline_dir: str, optimized_dir: str):
    """Compare two profiles."""
    b_xplane = find_xplane(baseline_dir)
    o_xplane = find_xplane(optimized_dir)

    if not b_xplane:
        print(f"ERROR: No .xplane.pb in {baseline_dir}", file=sys.stderr)
        return
    if not o_xplane:
        print(f"ERROR: No .xplane.pb in {optimized_dir}", file=sys.stderr)
        return

    b_label = os.path.basename(baseline_dir.rstrip('/'))
    o_label = os.path.basename(optimized_dir.rstrip('/'))

    # Analyze both
    print(f"\n  XPlane baseline: {b_xplane}")
    print(f"  XPlane optimized: {o_xplane}")

    b_roofline = analyze_roofline(b_xplane)
    print_roofline(b_roofline, b_label)

    o_roofline = analyze_roofline(o_xplane)
    print_roofline(o_roofline, o_label)

    # Comparison
    print_comparison(b_roofline, o_roofline, b_label, o_label)


def main():
    parser = argparse.ArgumentParser(
        description='xprof Profiling Analyzer for Pallas Kernels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m skills.xprof_analysis.analyze ./profile_r8_current
  python -m skills.xprof_analysis.analyze --compare ./profile_r6 ./profile_r8_current
  python -m skills.xprof_analysis.analyze --ops-only ./profile_r8_current
""")
    parser.add_argument('profile_dir', nargs='?', help='Profile directory to analyze')
    parser.add_argument('--compare', nargs=2, metavar=('BASELINE', 'OPTIMIZED'),
                        help='Compare two profiles')
    parser.add_argument('--ops-only', action='store_true',
                        help='Only show per-op time breakdown')
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    elif args.profile_dir:
        analyze(args.profile_dir, ops_only=args.ops_only)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
