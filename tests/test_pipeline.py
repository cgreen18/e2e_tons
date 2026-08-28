import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tons_sim.pipeline import _acceptance, run


REPOSITORY = Path(__file__).resolve().parents[1]


class ExperimentPipelineTest(unittest.TestCase):
    def test_missing_prepared_result_is_blocking(self) -> None:
        result = _acceptance([], {"expected-run"})
        self.assertFalse(result["passed"])
        self.assertIn("no structured result", result["failures"][0])

    def test_dry_run_records_exact_command_without_binary(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = {
                "repository_root": str(REPOSITORY),
                "output_directory": str(root / "output"),
                "binaries": {"congestion_unaware": str(root / "missing-binary")},
                "remote_memory": str(root / "remote.json"),
                "jobs": [
                    {
                        "run_id": "smoke",
                        "backend": "congestion_unaware",
                        "workload": str(root / "workload"),
                        "system": str(root / "system.json"),
                        "network": str(root / "network.yml"),
                    }
                ],
            }
            prepared_path = root / "prepared.json"
            prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
            run(prepared_path, dry_run=True)
            record = json.loads(
                (root / "output" / "runs" / "smoke" / "run.json").read_text(encoding="utf-8")
            )
            self.assertIn("--statistics-output=", record["command"][-1])
            self.assertEqual("smoke", record["run_id"])


if __name__ == "__main__":
    unittest.main()
