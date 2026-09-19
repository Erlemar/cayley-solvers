"""Regression checks: interrupted native searches cannot certify a path."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ladder", ROOT / "scripts/24_twsearch_ladder.py")
ladder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ladder)


class FakeProcess:
    def __init__(self, output, code):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(output)
        self.code = code

    def wait(self):
        return self.code


class NativeFailureTests(unittest.TestCase):
    def run_case(self, output, code):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/ihes_pdb") as td:
            folder = Path(td)
            args = ["ladder", "--baseline", str(ROOT / "submission_ihes.csv"),
                    "--journal", str(folder / "events.jsonl"), "--out", str(folder / "out.csv"),
                    "--window-length", "20", "--full-path-only", "--pids", "30"]
            with patch.object(sys, "argv", args), patch.object(ladder.subprocess, "Popen", return_value=FakeProcess(output, code)):
                result = ladder.main()
            rows = [json.loads(s) for s in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            return result, rows

    def test_abort_does_not_prove_none(self):
        code, rows = self.run_case("Solving\nterminate called\n", 9)
        self.assertEqual(code, 1)
        self.assertEqual(rows[0]["verdict"], "timeout")

    def test_silent_eof_does_not_prove_none(self):
        _, rows = self.run_case("Solving\n", 0)
        self.assertEqual(rows[0]["verdict"], "timeout")

    def test_explicit_timeout_remains_unresolved(self):
        _, rows = self.run_case("Solving\nSearch timed out at depth 18 after 2 seconds\n", 0)
        self.assertEqual(rows[0]["verdict"], "timeout")

    def test_explicit_exhaustion_is_a_proof(self):
        code, rows = self.run_case("Solving\nNo solution found in 18\n", 0)
        self.assertEqual(code, 0)
        self.assertEqual(rows[0]["verdict"], "none")


if __name__ == "__main__":
    unittest.main()
