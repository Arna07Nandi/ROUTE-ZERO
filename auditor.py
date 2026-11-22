"""AST + state-machine static code auditor with cross-file relations and impact mind-map chains."""

import ast
from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Dict, List, Tuple

from .scrubber import SecretScrubber


CODE_EXTS = {
    "py", "js", "jsx", "ts", "tsx", "mjs", "cjs", "go", "rs",
    "java", "c", "cpp", "h", "hpp", "sql", "json", "yaml", "yml", "sh",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}


@dataclass
class AuditFinding:
    category: str       # ERROR | DESIGN | LEGAL | COST
    severity: str       # HIGH | MED | LOW
    file: str
    line: int
    title: str
    snippet: str
    why_it_affects: str
    impact_chain: List[str]
    fix: str


def _get_ext(path: str) -> str:
    _, ext = os.path.splitext(path.lower())
    return ext.lstrip(".")


def lex_and_check_delimiters(content: str, ext: str) -> Tuple[List[str], List[Tuple[int, int, str]], bool]:
    """Character-level lexer stripping comments/strings and verifying delimiter balance."""
    is_py = ext == "py"
    is_hash = ext in {"py", "sh", "bash", "yml", "yaml", "toml", "rb", "r"}
    is_slash = ext in {"js", "jsx", "ts", "tsx", "mjs", "cjs", "java", "c", "cpp", "h", "hpp", "go", "rs", "cs", "swift", "kt", "php", "css"}
    is_sql = ext == "sql"
    is_code = is_hash or is_slash or is_sql or ext == "json"

    code_lines: List[str] = [""]
    errors: List[Tuple[int, int, str]] = []
    stack: List[Tuple[str, int, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}

    st = "CODE"
    line = 1
    col = 0
    n = len(content)
    i = 0

    while i < n:
        ch = content[i]
        nxt = content[i + 1] if i + 1 < n else ""
        nxt2 = content[i + 2] if i + 2 < n else ""

        if ch == "\n":
            if st == "LINE_COMMENT":
                st = "CODE"
            elif st in ("STR_S", "STR_D"):
                if is_code and ext not in ("json", "py"):
                    errors.append((line, col, "Unterminated string literal before newline"))
                st = "CODE"
            code_lines.append("")
            line += 1
            col = 0
            i += 1
            continue

        col += 1
        if st == "LINE_COMMENT":
            i += 1
            continue
        if st == "BLOCK_COMMENT":
            if ch == "*" and nxt == "/":
                st = "CODE"
                i += 2
                col += 1
            else:
                i += 1
            continue
        if st == "STR_TD":
            if ch == "\\":
                i += 2
                col += 1
            elif ch == '"' and nxt == '"' and nxt2 == '"':
                st = "CODE"
                i += 3
                col += 2
            else:
                i += 1
            continue
        if st == "STR_TS":
            if ch == "\\":
                i += 2
                col += 1
            elif ch == "'" and nxt == "'" and nxt2 == "'":
                st = "CODE"
                i += 3
                col += 2
            else:
                i += 1
            continue
        if st == "STR_S":
            if ch == "\\":
                i += 2
                col += 1
            elif ch == "'":
                st = "CODE"
                i += 1
            else:
                i += 1
            continue
        if st == "STR_D":
            if ch == "\\":
                i += 2
                col += 1
            elif ch == '"':
                st = "CODE"
                i += 1
            else:
                i += 1
            continue
        if st == "STR_T":
            if ch == "\\":
                i += 2
                col += 1
            elif ch == "`":
                st = "CODE"
                i += 1
            else:
                i += 1
            continue

        # CODE state
        if is_hash and ch == "#":
            st = "LINE_COMMENT"
            i += 1
            continue
        if is_slash and ch == "/" and nxt == "/":
            st = "LINE_COMMENT"
            i += 2
            col += 1
            continue
        if is_sql and ch == "-" and nxt == "-":
            st = "LINE_COMMENT"
            i += 2
            col += 1
            continue
        if (is_slash or is_sql) and ch == "/" and nxt == "*":
            st = "BLOCK_COMMENT"
            i += 2
            col += 1
            continue

        if is_py and ch == '"' and nxt == '"' and nxt2 == '"':
            st = "STR_TD"
            i += 3
            col += 2
            continue
        if is_py and ch == "'" and nxt == "'" and nxt2 == "'":
            st = "STR_TS"
            i += 3
            col += 2
            continue
        if ch == '"':
            st = "STR_D"
            code_lines[-1] += '""'
            i += 1
            continue
        if ch == "'" and ext != "rs":
            st = "STR_S"
            code_lines[-1] += "''"
            i += 1
            continue
        if ch == "`" and is_slash:
            st = "STR_T"
            code_lines[-1] += "``"
            i += 1
            continue

        code_lines[-1] += ch
        if is_code and not is_py:
            if ch in "([{":
                stack.append((ch, line, col))
            elif ch in ")]}":
                if not stack:
                    errors.append((line, col, f"Unexpected closing '{ch}'"))
                elif stack[-1][0] != pairs[ch]:
                    top_ch, top_ln, _ = stack.pop()
                    errors.append((line, col, f"Mismatched '{ch}' (opened '{top_ch}' at line {top_ln})"))
                else:
                    stack.pop()
        i += 1

    if is_code and not is_py:
        if st == "BLOCK_COMMENT":
            errors.append((line, col, "Unclosed block comment /* ... */"))
        elif st == "STR_T":
            errors.append((line, col, "Unclosed template literal"))
        while stack:
            u_ch, u_ln, u_col = stack.pop()
            errors.append((u_ln, u_col, f"Unclosed '{u_ch}'"))

    return code_lines, errors, is_code


def extract_imports(content: str) -> List[str]:
    """Extract cross-file import module names for dependency graph edges."""
    refs: List[str] = []
    for m in re.finditer(r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+)|require\(['\"]([^'\"]+)['\"]\))", content):
        val = m.group(1) or m.group(2) or m.group(3)
        if val:
            refs.append(val.split("/")[-1].split(".")[-1])
    return sorted(set(refs))


class CodeAuditor:
    """Multi-file static analyzer for Errors, System Design, Legal/PII, and Cost Leaks."""

    def __init__(self) -> None:
        self.scrubber = SecretScrubber()

    def collect_paths(self, targets: List[str]) -> Tuple[List[str], List[str]]:
        """Collect unique supported code paths and return (collected, missing_targets)."""
        seen = set()
        collected: List[str] = []
        missing: List[str] = []
        for t in targets:
            if not os.path.exists(t):
                missing.append(t)
                continue
            if os.path.isfile(t):
                norm = os.path.normpath(t)
                if norm not in seen:
                    seen.add(norm)
                    collected.append(norm)
            elif os.path.isdir(t):
                for root, dirs, files in os.walk(t):
                    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                    for fn in sorted(files):
                        if _get_ext(fn) in CODE_EXTS:
                            norm = os.path.normpath(os.path.join(root, fn))
                            if norm not in seen:
                                seen.add(norm)
                                collected.append(norm)
        return collected, missing

    def audit_files(self, file_map: Dict[str, str]) -> Dict[str, object]:
        findings: List[AuditFinding] = []
        relations: List[Dict[str, str]] = []

        stem_to_file = {os.path.splitext(os.path.basename(p))[0]: p for p in file_map}
        for fname, content in file_map.items():
            for imp in extract_imports(content):
                if imp in stem_to_file and stem_to_file[imp] != fname:
                    relations.append({"from": fname, "to": stem_to_file[imp], "type": "imports"})

        def add(cat: str, sev: str, fname: str, ln: int, title: str, snip: str, why: str, chain: List[str], fix: str) -> None:
            findings.append(
                AuditFinding(
                    category=cat,
                    severity=sev,
                    file=fname,
                    line=ln,
                    title=title,
                    snippet=snip.strip()[:120],
                    why_it_affects=why,
                    impact_chain=chain,
                    fix=fix,
                )
            )

        for fname, content in file_map.items():
            ext = _get_ext(fname)
            lines = content.splitlines()
            code_lines, delim_errs, is_code = lex_and_check_delimiters(content, ext)

            # 1. Language-exact parsers (zero false positives for Python & JSON)
            if ext == "json":
                try:
                    json.loads(content)
                except Exception as exc:
                    add(
                        "ERROR", "HIGH", fname, 1, f"Invalid JSON: {exc}", lines[0] if lines else "",
                        "Malformed JSON crashes config loaders and API parsers immediately on startup.",
                        [fname, "JSON Parse Error", "Startup / Config Crash", "Service Downtime"],
                        "Fix JSON syntax (trailing commas, quotes, or braces).",
                    )
            elif ext == "py":
                try:
                    tree = ast.parse(content, filename=fname)
                    for node in ast.walk(tree):
                        # Mutable default args
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                                    ln = getattr(node, "lineno", 1)
                                    add(
                                        "ERROR", "HIGH", fname, ln, f"Mutable default argument in `{node.name}()`",
                                        lines[ln - 1] if ln <= len(lines) else "",
                                        "Default list/dict/set is created once at definition time and shared across calls, leaking state.",
                                        [f"{fname}:{ln}", "Mutable Default", "Cross-Call State Leak", "Heisenbugs / Tenant Leakage"],
                                        "Use default=None and initialize the container inside the function.",
                                    )
                        # Bare except:
                        elif isinstance(node, ast.ExceptHandler) and node.type is None:
                            ln = getattr(node, "lineno", 1)
                            add(
                                "ERROR", "MED", fname, ln, "Bare `except:` swallows all exceptions (including KeyboardInterrupt)",
                                lines[ln - 1] if ln <= len(lines) else "",
                                "Hides real crashes and system exits, turning fast failures into silent corruption.",
                                [f"{fname}:{ln}", "Bare except", "Hidden Runtime Fault", "Corrupt Downstream State"],
                                "Catch specific exception types (e.g. `except ValueError:`) and log or handle.",
                            )
                        # Direct eval() / exec() call
                        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                            ln = getattr(node, "lineno", 1)
                            add(
                                "ERROR", "HIGH", fname, ln, f"Unsafe dynamic `{node.func.id}()` call",
                                lines[ln - 1] if ln <= len(lines) else "",
                                "Allows arbitrary code execution if any untrusted string reaches this call.",
                                [f"{fname}:{ln}", f"{node.func.id}()", "Remote Code Execution", "Full Host Compromise"],
                                "Replace with json.loads() or an explicit dispatch table.",
                            )
                except SyntaxError as exc:
                    ln = exc.lineno or 1
                    col = exc.offset or 1
                    add(
                        "ERROR", "HIGH", fname, ln, f"Python {type(exc).__name__}: {exc.msg} (col {col})",
                        lines[ln - 1] if ln <= len(lines) else "",
                        "Syntax or indentation errors prevent Python from importing or running this module.",
                        [f"{fname}:{ln}", type(exc).__name__, "Import / Boot Failure", "Broken App & Dependents"],
                        "Fix the syntax/indentation error at the reported line and column.",
                    )
            else:
                for ln, col, msg in delim_errs[:5]:
                    add(
                        "ERROR", "HIGH", fname, ln, f"{msg} (col {col})", lines[ln - 1] if ln <= len(lines) else "",
                        "Unbalanced delimiters break parsing so the module fails to compile or run.",
                        [f"{fname}:{ln}", msg, "SyntaxError / Compile Fail", "Broken Build & Importers"],
                        "Balance all brackets, parentheses, and string literals.",
                    )

            in_async = False
            in_loop = False
            loop_indent = -1
            has_try = bool(re.search(r"\b(try\s*\{|try\s*:)", "\n".join(code_lines)))
            base_name = os.path.basename(fname).lower()

            for idx, raw in enumerate(lines):
                ln = idx + 1
                code = code_lines[idx] if idx < len(code_lines) else ""
                trimmed = code.strip()
                indent = len(raw) - len(raw.lstrip(" \t"))
                is_regex_line = bool(re.search(r"\bre\.(search|match|compile|finditer|sub)\s*\(", raw))

                if re.match(r"^\s*async\s+(def|function)\b", code):
                    in_async = True
                elif re.match(r"^\s*(def|function|class)\b", code) and indent == 0:
                    in_async = False

                if re.search(r"\b(for|while)\b", trimmed) or re.search(r"\.(map|forEach)\s*\(", trimmed):
                    in_loop = True
                    loop_indent = indent
                elif trimmed and indent <= loop_indent and not trimmed.startswith(("}", ")")):
                    in_loop = False

                # Non-Python code ERROR checks
                if is_code and ext not in ("json", "py"):
                    if re.search(r"\beval\s*\(", code):
                        add(
                            "ERROR", "HIGH", fname, ln, "Unsafe eval() execution", raw,
                            "Allows arbitrary code execution if any untrusted string reaches this call.",
                            [f"{fname}:{ln}", "eval()", "Remote Code Execution", "Full Host Compromise"],
                            "Replace with JSON.parse() or an explicit dispatch map.",
                        )
                    if re.search(r"\bcatch\s*(\([^)]*\))?\s*\{\s*\}", code):
                        add(
                            "ERROR", "MED", fname, ln, "Silent empty catch block", raw,
                            "Hides crashes and corrupt state, turning fast failures into silent data corruption.",
                            [f"{fname}:{ln}", "Silent Catch", "Hidden Runtime Fault", "Corrupt Downstream State"],
                            "Log, handle, or rethrow caught exceptions.",
                        )
                    if ext in {"js", "jsx", "ts", "tsx", "c", "cpp", "java", "go"} and re.search(r"\bif\s*\([^=<>!]*[^=<>!]=[^=][^)]*\)", code):
                        add(
                            "ERROR", "HIGH", fname, ln, "Assignment (=) inside if() condition", raw,
                            "Mutates the variable instead of comparing it, flipping branch behavior.",
                            [f"{fname}:{ln}", "Assignment in if()", "Inverted Control Flow", "Logic / Auth Bypass"],
                            "Use === or == for comparison.",
                        )
                    if ext in {"js", "jsx", "ts", "tsx"} and re.search(r"\bJSON\.parse\s*\(", code) and not has_try:
                        add(
                            "ERROR", "MED", fname, ln, "Unguarded JSON.parse() call", raw,
                            "A single malformed payload throws an uncaught SyntaxError and crashes the handler.",
                            [f"{fname}:{ln}", "Unguarded JSON.parse", "Uncaught Exception", "Worker / UI Crash"],
                            "Wrap JSON.parse() in try/catch.",
                        )

                # SQL injection check across languages
                if is_code and not is_regex_line and re.search(r"(?:SELECT|INSERT|UPDATE|DELETE)\b", raw, re.I) and not trimmed.startswith(("//", "#")):
                    if re.search(r'f["\'][^"\']*(?:SELECT|INSERT|UPDATE|DELETE)[^"\']*\{', raw, re.I) or re.search(r"`[^`]*(?:SELECT|INSERT|UPDATE|DELETE)[^`]*\$\{", raw, re.I):
                        add(
                            "ERROR", "HIGH", fname, ln, "SQL injection via string interpolation", raw,
                            "User input concatenated into SQL lets attackers read, modify, or drop tables.",
                            [f"{fname}:{ln}", "SQL Interpolation", "SQL Injection", "Database Breach"],
                            "Use parameterized queries.",
                        )

                # SYSTEM DESIGN checks
                if re.search(r"\b(requests\.(get|post|put|delete)|fetch|axios\.(get|post)|urlopen)\s*\(", code) and not re.search(r"timeout|AbortController|signal", raw, re.I):
                    add(
                        "DESIGN", "HIGH", fname, ln, "Network call without timeout", raw,
                        "A hung upstream holds worker threads forever until the entire service stops responding.",
                        [f"{fname}:{ln}", "No Timeout", "Thread Pool Exhaustion", "Cascading 504 Outage"],
                        "Set explicit connect/read timeouts and circuit breakers.",
                    )
                if re.search(r"while\s*(True|\(\s*true\s*\)|\(\s*1\s*\))", code):
                    add(
                        "DESIGN", "HIGH", fname, ln, "Unbounded loop (retry storm risk)", raw,
                        "Without max retries or backoff, transient errors turn into infinite CPU/API storms.",
                        [f"{fname}:{ln}", "Unbounded Loop", "Retry Storm", "CPU Spike & Rate-Limit Ban"],
                        "Add max_retries, exponential backoff, and a circuit breaker.",
                    )
                if in_async and re.search(r"\btime\.sleep\s*\(|fs\.readFileSync|execSync", code):
                    add(
                        "DESIGN", "HIGH", fname, ln, "Blocking synchronous I/O inside async handler", raw,
                        "Freezes the entire single-threaded event loop, stalling every concurrent user request.",
                        [f"{fname}:{ln}", "Blocking I/O in Async", "Event Loop Freeze", "Tail Latency Spike (p99)"],
                        "Use asyncio.sleep or non-blocking async I/O.",
                    )
                if in_loop and re.search(r"\b(fetch|requests\.|axios\.|cursor\.execute|db\.query)\b", code, re.I):
                    add(
                        "DESIGN", "MED", fname, ln, "Potential N+1 network or DB call in loop", raw,
                        "Latency and database load scale linearly with list size, causing timeouts under real traffic.",
                        [f"{fname}:{ln}", "N+1 Query Loop", "O(N) Round-Trips", "DB Saturation & Slow Pages"],
                        "Batch queries or use bounded concurrency.",
                    )

                # LEGAL checks (skip regex pattern lines & local host f-strings)
                if not is_regex_line and (raw.lstrip().startswith(("#", "//", "/*")) or base_name in ("package.json", "pyproject.toml", "setup.py", "license")):
                    if re.search(r"\b(AGPL|GPL-?3|GPL-?2|SSPL|CC-BY-SA)\b", raw, re.I):
                        add(
                            "LEGAL", "HIGH", fname, ln, "Copyleft license reference detected", raw,
                            "Strong copyleft terms (AGPL/GPL) can force you to open-source proprietary SaaS code if linked.",
                            [f"{fname}:{ln}", "Copyleft License", "IP Contamination Risk", "B2B Deal / Acquisition Blocker"],
                            "Audit dependency licenses before commercial distribution.",
                        )
                if not is_regex_line and re.search(r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|\{host\})[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw, re.I):
                    add(
                        "LEGAL", "MED", fname, ln, "Unencrypted HTTP transport URL", raw,
                        "Sends payloads in plaintext across networks, failing SOC 2, GDPR, and PCI compliance.",
                        [f"{fname}:{ln}", "Plaintext HTTP", "MITM Eavesdropping", "Compliance Violation"],
                        "Enforce HTTPS/TLS everywhere.",
                    )
                scrub = self.scrubber.scrub(raw)
                if scrub.redacted_count > 0:
                    add(
                        "LEGAL", "HIGH", fname, ln, f"Hardcoded secret or PII ({scrub.redacted_count} match)", scrub.scrubbed_text,
                        "Secrets committed to git are scraped within seconds by bots, leading to stolen bills and data breaches.",
                        [f"{fname}:{ln}", "Hardcoded Secret/PII", "Git Credential Leak", "Unauthorized Spend & Breach"],
                        "Move credentials to environment variables or a secret vault.",
                    )

                # COST checks (only flag actual model assignments/calls, not rate-card tables or regexes)
                if not is_regex_line and re.search(r"\bmodel\s*[:=]\s*['\"](gpt-6-astra|gpt-5\.5|gpt-4o(?!-mini)|claude-opus|o1-preview|o3(?!-mini))", raw, re.I):
                    add(
                        "COST", "HIGH", fname, ln, "Hardcoded frontier LLM model in API call", raw,
                        "Sends simple tasks to models costing up to 100x more than Tier-1 economy models.",
                        [f"{fname}:{ln}", "Hardcoded Frontier Model", "Over-Provisioned Inference", "60–90% Wasted API Bill"],
                        "Route via RouteZero tiers (Tier 1–2 for routine prompts, Tier 4 only for hard reasoning).",
                    )
                if re.search(r"\b(chat\.completions\.create|messages\.create|generate_content|generateContent)\b", code):
                    win = " ".join(lines[idx : min(len(lines), idx + 8)])
                    if not re.search(r"max_tokens|max_output_tokens|max_completion_tokens", win, re.I):
                        add(
                            "COST", "MED", fname, ln, "LLM call missing max_tokens output cap", raw,
                            "Output tokens cost 4–5x input tokens; uncapped generations can drain budgets on rambling replies.",
                            [f"{fname}:{ln}", "No max_tokens Cap", "Unbounded Output Tokens", "5x Output Cost Spike"],
                            "Set explicit max_tokens on every LLM call.",
                        )
                    if in_loop:
                        add(
                            "COST", "HIGH", fname, ln, "LLM API call executed inside a loop", raw,
                            "Multiplies token spend by N iterations and risks Denial-of-Wallet storms on large inputs.",
                            [f"{fname}:{ln}", "LLM Call in Loop", "O(N) Token Multiplier", "Denial-of-Wallet Bill Shock"],
                            "Add semantic caching, deduplication, or Batch API (50% off).",
                        )

        penalty = sum(12 if f.severity == "HIGH" else 6 if f.severity == "MED" else 2 for f in findings)
        return {
            "files": list(file_map.keys()),
            "health_score": max(0, 100 - penalty),
            "relations": relations,
            "findings": [asdict(f) for f in findings],
        }


def render_ascii_mindmap(report: Dict[str, object]) -> str:
    """Render a clean terminal mind map showing File -> Issue -> Impact Cascade."""
    lines = [f"RouteZero Impact Mind Map  [Health: {report['health_score']}/100]"]
    findings = report["findings"]
    relations = report["relations"]
    by_file: Dict[str, List[Dict[str, object]]] = {}
    for f in findings:
        by_file.setdefault(f["file"], []).append(f)

    if relations:
        lines.append("├── Cross-File Dependencies")
        for idx, rel in enumerate(relations):
            branch = "└──" if idx == len(relations) - 1 else "├──"
            lines.append(f"│   {branch} {rel['from']} ──imports──▶ {rel['to']}")

    files = list(by_file.keys())
    if not files:
        lines.append("└── Clean! Zero issues found across audited files.")
        return "\n".join(lines)

    for fi, fname in enumerate(files):
        f_branch = "└──" if fi == len(files) - 1 else "├──"
        f_pipe = "    " if fi == len(files) - 1 else "│   "
        lines.append(f"{f_branch} {fname}")
        items = by_file[fname]
        for ii, item in enumerate(items):
            i_branch = "└──" if ii == len(items) - 1 else "├──"
            i_pipe = "    " if ii == len(items) - 1 else "│   "
            chain = " ──▶ ".join(item["impact_chain"][1:])
            lines.append(f"{f_pipe}{i_branch} [{item['category']}/{item['severity']}] L{item['line']}: {item['title']}")
            lines.append(f"{f_pipe}{i_pipe}├── Why: {item['why_it_affects']}")
            lines.append(f"{f_pipe}{i_pipe}├── Impact: {chain}")
            lines.append(f"{f_pipe}{i_pipe}└── Fix: {item['fix']}")
    return "\n".join(lines)
