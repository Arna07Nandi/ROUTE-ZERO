"""RouteZero CLI — simple 1-word commands for routing, auditing, models, and proxy server."""

import argparse
import json
import sys
from typing import Dict, List, Optional

from .auditor import CodeAuditor, render_ascii_mindmap
from .models import MODEL_CATALOG
from .proxy import RouteZeroEngine, run_server


def _cmd_audit(args: argparse.Namespace) -> int:
    auditor = CodeAuditor()
    targets = args.paths or ["."]
    paths, missing = auditor.collect_paths(targets)
    if missing:
        print(f"Warning: path(s) not found: {', '.join(missing)}", file=sys.stderr)
    if not paths:
        print(f"No supported code files found in: {', '.join(targets)}", file=sys.stderr)
        return 1

    file_map: Dict[str, str] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                file_map[p] = fh.read()
        except OSError as exc:
            print(f"Warning: could not read {p}: {exc}", file=sys.stderr)

    report = auditor.audit_files(file_map)
    if args.json:
        out = json.dumps(report, indent=2)
        if args.json == "-":
            print(out)
        else:
            with open(args.json, "w", encoding="utf-8") as fh:
                fh.write(out)
            print(f"Saved JSON report to {args.json}")
    else:
        print(render_ascii_mindmap(report))

    if args.min_score is not None and report["health_score"] < args.min_score:
        return 2
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    if args.baseline not in MODEL_CATALOG:
        valid = ", ".join(sorted(MODEL_CATALOG.keys()))
        print(f"Error: unknown baseline model '{args.baseline}'. Valid models: {valid}", file=sys.stderr)
        return 1
    engine = RouteZeroEngine(baseline_model=args.baseline)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("Error: prompt cannot be empty.", file=sys.stderr)
        return 1
    res = engine.handle_chat([{"role": "user", "content": prompt}], skip_cache=args.no_cache)
    print(f"Model:    {res.model_used} (Tier {res.tier}, score={res.complexity_score})")
    print(f"Cost:     ${res.actual_cost_usd:.6f}  (saved ${res.saved_usd:.6f} vs {res.baseline_model})")
    print(f"Scrubbed: {res.redacted_secrets} secret(s) | Latency: {res.overhead_ms:.2f} ms")
    return 0


def _cmd_models(_args: argparse.Namespace) -> int:
    print(f"{'MODEL':<24} {'PROVIDER':<11} {'TIER':<6} {'IN $/1M':>9} {'OUT $/1M':>10}")
    print("-" * 64)
    for spec in sorted(MODEL_CATALOG.values(), key=lambda s: (int(s.tier), s.input_per_1m_usd)):
        print(f"{spec.model_id:<24} {spec.provider:<11} T{int(spec.tier):<5} ${spec.input_per_1m_usd:>8.2f} ${spec.output_per_1m_usd:>9.2f}")
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    if args.baseline not in MODEL_CATALOG:
        valid = ", ".join(sorted(MODEL_CATALOG.keys()))
        print(f"Error: unknown baseline model '{args.baseline}'. Valid models: {valid}", file=sys.stderr)
        return 1
    try:
        run_server(host=args.host, port=args.port, baseline_model=args.baseline)
    except KeyboardInterrupt:
        print("\nRouteZero proxy stopped.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="routezero",
        description="RouteZero: Zero-dependency local LLM cost router & multi-file code auditor.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Audit files/folders for errors, system design, legal, and cost leaks.")
    p_audit.add_argument("paths", nargs="*", help="Files or directories to audit (defaults to '.')")
    p_audit.add_argument("--json", metavar="FILE", help="Write JSON report to FILE (or '-' for stdout)")
    p_audit.add_argument("--min-score", type=int, default=None, help="Exit with code 2 if health score is below N (CI gate)")
    p_audit.set_defaults(func=_cmd_audit)

    p_route = sub.add_parser("route", help="Route a single prompt and print tier, cost, and savings.")
    p_route.add_argument("prompt", nargs="+", help="Prompt text to route")
    p_route.add_argument("--baseline", default="gpt-6.1-sol", help="Baseline model for savings comparison")
    p_route.add_argument("--no-cache", action="store_true", help="Skip semantic cache")
    p_route.set_defaults(func=_cmd_route)

    p_models = sub.add_parser("models", help="List supported models and per-1M-token rates.")
    p_models.set_defaults(func=_cmd_models)

    p_start = sub.add_parser("start", help="Start the local OpenAI-compatible routing proxy server.")
    p_start.add_argument("--host", default="127.0.0.1")
    p_start.add_argument("--port", type=int, default=8088)
    p_start.add_argument("--baseline", default="gpt-6.1-sol")
    p_start.set_defaults(func=_cmd_start)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
