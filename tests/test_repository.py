import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bilingual_retrieval.experiment import verify
from tests.test_contracts import ROOT


class RepositoryTests(unittest.TestCase):
    def test_readme_structure(self):
        headings = [
            line[3:]
            for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")
        ]
        self.assertEqual(
            headings,
            [
                "Problem",
                "Scope",
                "Quick start",
                "Usage",
                "Architecture",
                "Reproduction",
                "Results",
                "Testing",
                "Limitations",
                "Contributing",
                "License",
            ],
        )

    def test_tracked_text_has_no_forbidden_dash(self):
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        for name in filter(None, paths):
            with self.subTest(path=name):
                self.assertNotIn(chr(0x2014), (ROOT / name).read_text(encoding="utf-8"))

    def test_fixture_generation_is_byte_reproducible(self):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "data").mkdir()
            shutil.copy(ROOT / "scripts/build_dataset.py", root / "scripts/build_dataset.py")
            for name in ("scenarios.json", "distractors.json"):
                shutil.copy(ROOT / "data" / name, root / "data" / name)
            subprocess.run([sys.executable, str(root / "scripts/build_dataset.py")], check=True)
            self.assertEqual(
                (root / "data/dataset.json").read_bytes(), (ROOT / "data/dataset.json").read_bytes()
            )

    def test_reference_evidence(self):
        for split in ("dev", "eval"):
            path = ROOT / "results" / split / "report.json"
            verify(ROOT / "data/dataset.json", path)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(report["environment"]["dirty"])
            self.assertTrue(
                all(n <= 128 for group in report["token_lengths"].values() for n in group)
            )
            self.assertEqual(
                report["evaluation"]["breakdowns"],
                json.loads(path.with_name("summary.json").read_text(encoding="utf-8")),
            )
