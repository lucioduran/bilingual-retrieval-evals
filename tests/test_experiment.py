import contextlib
import copy
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bilingual_retrieval.cli import main
from bilingual_retrieval.contracts import parse_dataset, read_json
from bilingual_retrieval.experiment import environment, evaluate, run, selected, verify, write_json
from bilingual_retrieval.models import (
    FILES,
    MAX_TOKENS,
    MODEL_ID,
    REVISION,
    Encoder,
    digest,
    snapshot,
)
from tests.test_contracts import ROOT, fixture


class FakeEncoder:
    files = {name: "a" * 64 for name in FILES}

    def __init__(self, cache, threads):
        pass

    def encode(self, texts):
        return [[1.0, 0.0] if "Hot" in t or "hot" in t else [0.0, 1.0] for t in texts], [5] * len(
            texts
        )


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset.json"
        write_json(self.dataset, fixture())
        self.output = self.root / "output"

    def execute(self):
        with (
            patch("bilingual_retrieval.experiment.Encoder", FakeEncoder),
            patch("bilingual_retrieval.experiment.environment", return_value={"test": True}),
        ):
            run(self.dataset, self.output, self.root, "dev", 1, 2)
        return read_json(self.output / "report.json")

    def test_run_and_verify(self):
        report = self.execute()
        self.assertEqual(report["evaluation"]["traces"]["q"]["dense"]["ranking"], ["a", "b"])
        self.assertEqual(report["dataset_sha256"], digest(self.dataset))
        self.assertEqual(report["model"]["revision"], REVISION)
        self.assertEqual(
            read_json(self.output / "summary.json"), report["evaluation"]["breakdowns"]
        )
        verify(self.dataset, self.output / "report.json")
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_tampering(self):
        original = self.execute()
        mutations = [
            lambda r: r.update(version=2),
            lambda r: r.update(version=True),
            lambda r: r.update(dataset_sha256="wrong"),
            lambda r: r["model"].update(revision="main"),
            lambda r: r["model"].update(id="other"),
            lambda r: r["model"].update(max_tokens=256),
            lambda r: r["evaluation"]["traces"]["q"]["dense"].update(ranking=["b", "a"]),
            lambda r: r["vectors"]["queries"].update(q=[0, 1]),
            lambda r: r["vectors"]["documents"].pop("a"),
            lambda r: r["vectors"]["queries"].update(extra=[1, 0]),
        ]
        for mutate in mutations:
            report = copy.deepcopy(original)
            mutate(report)
            write_json(self.output / "report.json", report)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                verify(self.dataset, self.output / "report.json")

    def test_empty_unanswerable_and_bad_arguments(self):
        data = parse_dataset(fixture())
        for split in ("bad", "eval"):
            with self.assertRaises(ValueError):
                selected(data, split)
        with self.assertRaises(ValueError):
            run(self.dataset, self.output, self.root, "dev", 0, 2)
        raw = fixture()
        for j in raw["queries"][0]["judgments"]:
            j["grade"] = 0
        evaluated = evaluate(parse_dataset(raw), "dev", 1, {"a": [1], "b": [1]}, {"q": [1]})
        self.assertIsNone(evaluated["family_intervals"]["dense"]["ndcg"])

    def test_failed_run_leaves_no_success_report(self):
        with patch("bilingual_retrieval.experiment.Encoder", side_effect=RuntimeError("missing")):
            with self.assertRaises(RuntimeError):
                run(self.dataset, self.output, self.root, "dev", 1, 2)
        self.assertFalse((self.output / "report.json").exists())

    def test_environment(self):
        with (
            patch("bilingual_retrieval.experiment.platform.system", return_value="Darwin"),
            patch(
                "bilingual_retrieval.experiment.subprocess.check_output", return_value="test cpu\n"
            ),
            patch("bilingual_retrieval.experiment.importlib.metadata.version", return_value="1"),
        ):
            with patch("bilingual_retrieval.experiment.platform.platform", return_value="test OS"):
                got = environment(2)
            self.assertEqual(got["cpu"], "test cpu")
            self.assertTrue(got["dirty"])
        with (
            patch("bilingual_retrieval.experiment.platform.system", return_value="Linux"),
            patch("bilingual_retrieval.experiment.git_value", return_value=""),
            patch("bilingual_retrieval.experiment.importlib.metadata.version", return_value="1"),
        ):
            self.assertFalse(environment(2)["dirty"])

    def test_cli(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["validate", "--dataset", str(self.dataset)]), 0)
            self.assertEqual(main(["validate", "--dataset", str(self.root / "absent")]), 2)
            with patch("bilingual_retrieval.cli.snapshot") as download:
                self.assertEqual(main(["download"]), 0)
                download.assert_called_once_with(Path(".cache/models"), offline=False)
            with patch("bilingual_retrieval.cli.run") as execute:
                self.assertEqual(main(["run", "--output", "new", "--split", "dev"]), 0)
                self.assertEqual(execute.call_args.args[-2:], (3, 2))
            self.execute()
            self.assertEqual(
                main(
                    [
                        "verify",
                        "--dataset",
                        str(self.dataset),
                        "--report",
                        str(self.output / "report.json"),
                    ]
                ),
                0,
            )
            with patch("bilingual_retrieval.cli.snapshot", side_effect=ImportError("unavailable")):
                self.assertEqual(main(["download"]), 2)

    def test_module_entrypoint(self):
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "bilingual_retrieval", "validate"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class AdapterTests(unittest.TestCase):
    def test_snapshot_pins_revision_and_local_mode(self):
        hub = MagicMock()
        hub.snapshot_download.return_value = "cache"
        with patch("bilingual_retrieval.models.importlib.import_module", return_value=hub):
            self.assertEqual(snapshot(Path("cache"), True), Path("cache"))
            hub.snapshot_download.assert_called_once_with(
                MODEL_ID,
                revision=REVISION,
                cache_dir="cache",
                allow_patterns=FILES,
                local_files_only=True,
            )

    def test_encoder_loading_and_batch_pooling_contract(self):
        torch, transformers = MagicMock(), MagicMock()
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.side_effect = lambda batch, **kw: (
            {"input_ids": [[1, 2, 3] for _ in batch]}
            if not kw.get("return_tensors")
            else {"attention_mask": MagicMock()}
        )
        model = transformers.AutoModel.from_pretrained.return_value
        pooled = model.return_value.last_hidden_state.__mul__.return_value.sum.return_value.__truediv__.return_value
        pooled.tolist.side_effect = [[[1.0, 2.0]] * 8, [[1.0, 2.0]]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in FILES:
                (path / name).write_text("test", encoding="utf-8")
            with (
                patch("bilingual_retrieval.models.snapshot", return_value=path) as resolve,
                patch(
                    "bilingual_retrieval.models.importlib.import_module",
                    side_effect=[torch, transformers],
                ),
            ):
                encoder = Encoder(path, 2)
                vectors, lengths = encoder.encode(["text"] * 9)
                self.assertEqual(len(vectors), 9)
                self.assertEqual(lengths, [3] * 9)
                resolve.assert_called_once_with(path, offline=True)
                model.to.assert_called_once_with("cpu")
                model.eval.assert_called_once()
                torch.set_num_threads.assert_called_once_with(2)
                transformers.AutoModel.from_pretrained.assert_called_once_with(
                    path, local_files_only=True, trust_remote_code=False, use_safetensors=True
                )
                self.assertEqual(tokenizer.call_args.kwargs["max_length"], MAX_TOKENS)
                self.assertEqual(encoder.encode([]), ([], []))
        for threads in (0, 5, True):
            with self.assertRaises(ValueError):
                Encoder(Path("none"), threads)


class ReplayToleranceTests(unittest.TestCase):
    def test_numeric_roundoff_is_allowed_but_structure_remains_exact(self):
        from bilingual_retrieval.experiment import equivalent

        self.assertTrue(equivalent({"mean": 0.6875000000000001}, {"mean": 0.6875}))
        self.assertFalse(equivalent({"mean": 0.6875}, {"mean": 0.6876}))
        self.assertFalse(equivalent(0.0, float("nan")))
        self.assertFalse(equivalent(1.0, float("inf")))
        self.assertFalse(equivalent(1.0, True))
        self.assertFalse(equivalent(1, True))
        self.assertFalse(equivalent(["a", "b"], ["b", "a"]))
        self.assertFalse(equivalent(["a"], ["a", "b"]))
        self.assertFalse(equivalent(["a"], ("a",)))
        self.assertFalse(equivalent({"a": 1}, {"a": 1, "b": 2}))
        self.assertTrue(equivalent(None, None))
