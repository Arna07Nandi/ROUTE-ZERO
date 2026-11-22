"""Unit & CLI integration tests + 160-request benchmark for RouteZero."""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routezero import (
    CircuitBreaker,
    CodeAuditor,
    ComplexityRouter,
    RouteZeroEngine,
    RoutingTier,
    SecretScrubber,
    SemanticCache,
)
from routezero.cli import main as cli_main


class TestRouteZero(unittest.TestCase):
    def test_secret_scrubber_catches_keys_emails_and_cards(self) -> None:
        scrubber = SecretScrubber()
        sample = (
            "Debug this with key sk-proj-9A8b7C6d5E4f3G2h1I0jK9L8m7N6o5P4 "
            "for user alice@acme.io and AWS AKIAIOSFODNN7EXAMPLE"
        )
        res = scrubber.scrub(sample)
        self.assertEqual(res.redacted_count, 3)
        self.assertNotIn("sk-proj-", res.scrubbed_text)
        self.assertNotIn("alice@acme.io", res.scrubbed_text)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", res.scrubbed_text)

    def test_semantic_cache_exact_and_near_duplicate(self) -> None:
        cache = SemanticCache(similarity_threshold=0.80)
        cache.put("Extract the invoice total and vendor name as JSON", '{"total": 420}', "gemini-2.5-flash-lite", 12, 8)

        exact = cache.lookup("extract the invoice total and vendor name as json")
        self.assertIsNotNone(exact)
        self.assertTrue(exact.exact)
        self.assertEqual(exact.similarity, 1.0)

        near = cache.lookup("Extract the invoice total and vendor name as JSON please")
        self.assertIsNotNone(near)
        self.assertFalse(near.exact)
        self.assertGreaterEqual(near.similarity, 0.80)

    def test_complexity_router_tiers(self) -> None:
        router = ComplexityRouter()
        easy = router.score([{"role": "user", "content": "Classify the sentiment of this review as positive or negative."}])
        self.assertEqual(easy.tier, RoutingTier.ECONOMY)

        hard = router.score(
            [
                {
                    "role": "user",
                    "content": (
                        "Prove that the Raft distributed consensus invariant holds under network partitions, "
                        "derive the formal Z3 SMT constraints to verify deadlock freedom, and optimize the "
                        "time complexity in O(log n) without using global locks:\n"
                        "```python\nasync def commit_log(state, entries):\n    return {k: [v for v in (entries)] for k in state}\n```"
                    ),
                }
            ]
        )
        self.assertIn(hard.tier, (RoutingTier.STRONG, RoutingTier.FRONTIER))
        self.assertGreater(hard.complexity_score, easy.complexity_score)

    def test_circuit_breaker_stops_runaway_loop(self) -> None:
        breaker = CircuitBreaker(max_identical_repeats=3, loop_window_sec=30.0)
        for i in range(3):
            st = breaker.check_and_record("agent-1", "Retry failed tool call #42", 0.002, now=100.0 + i)
            self.assertTrue(st.allowed)
        tripped = breaker.check_and_record("agent-1", "Retry failed tool call #42", 0.002, now=105.0)
        self.assertFalse(tripped.allowed)
        self.assertEqual(tripped.trip_type, "RUNAWAY_LOOP")

    def test_auditor_catches_real_bugs_without_false_positives(self) -> None:
        auditor = CodeAuditor()
        # Clean Python with multiline comprehension and smiley comment
        clean = auditor.audit_files({
            "clean.py": '"""Docstring :) ("""\n# comment eval()\nvals = [x for x in range(5)]\n'
        })
        self.assertEqual(clean["health_score"], 100)
        self.assertEqual(len(clean["findings"]), 0)

        # Real bugs across all 4 pillars
        buggy = auditor.audit_files({
            "bad.py": (
                "def leak(items=[]):\n"
                "    res = requests.get('http://api.example.com/v1')\n"
                "    client.chat.completions.create(model='gpt-6-astra', messages=[])\n"
            )
        })
        cats = {f["category"] for f in buggy["findings"]}
        self.assertEqual(cats, {"ERROR", "DESIGN", "LEGAL", "COST"})

    def test_cli_commands_and_error_handling(self) -> None:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            rc = cli_main(["models"])
        self.assertEqual(rc, 0)
        self.assertIn("gemini-2.5-flash-lite", buf_out.getvalue())

        # Invalid baseline returns clean exit code 1 instead of KeyError
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            rc = cli_main(["route", "hello", "--baseline", "fake-model"])
        self.assertEqual(rc, 1)
        self.assertIn("unknown baseline model", buf_err.getvalue())

        # Auditing RouteZero's own package should score 100/100 (zero self false positives)
        pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "routezero"))
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            rc = cli_main(["audit", pkg_dir, "--min-score", "95"])
        self.assertEqual(rc, 0, msg=buf_out.getvalue())
        self.assertIn("[Health: 100/100]", buf_out.getvalue())

    def test_end_to_end_benchmark_savings(self) -> None:
        engine = RouteZeroEngine(baseline_model="gpt-6.1-sol")
        workload = [
            ("Classify support ticket urgency: My invoice PDF won't download.", 200),
            ("Extract order_id and sku as JSON: Order #9921 for SKU-44B arrived damaged.", 180),
            ("Summarize these 3 release notes into one bullet point.", 160),
            ("Translate to Spanish: Your shipment has cleared customs.", 140),
            ("Classify support ticket urgency: My invoice PDF will not download!", 200),
            ("Extract order_id and sku as JSON: Order #9921 for SKU-44B arrived damaged", 180),
            (
                "Write a Python function to parse CSV rows and compute rolling 7-day averages.\n```python\ndef rolling_avg(rows):\n    pass\n```",
                450,
            ),
            (
                "Prove the linearizability invariant for a multi-shard distributed consensus log, "
                "verify deadlock freedom with Z3 SMT constraints, and optimize time complexity in O(log n) "
                "handling every edge case strictly step by step:\n```python\nasync def append_entries(raft_state, log):\n    pass\n```",
                900,
            ),
        ]

        t = 1000.0
        for i in range(160):
            prompt, out_tok = workload[i % len(workload)]
            t += 12.0
            res = engine.handle_chat(
                [{"role": "user", "content": prompt}],
                client_id=f"svc-{i % 4}",
                expected_output_tokens=out_tok,
                now=t,
            )
            self.assertEqual(res.status_code, 200)

        summary = engine.summary_metrics()
        self.assertGreater(summary["savings_pct"], 55.0)
        self.assertGreater(summary["cache_hits"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
