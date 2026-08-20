import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "docker" / "verify_submission.py"
MODULE_SPEC = importlib.util.spec_from_file_location("verify_submission", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
verifySubmission = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(verifySubmission)


class SubmissionVerifierTest(unittest.TestCase):
    def testSourceCheckoutIncludesCompleteRuntimeContext(self) -> None:
        verifySubmission.verifySourceCheckout()

    def testDockerIgnoreDirectoryRuleIsDetected(self) -> None:
        path = "src/instatarget/training/model.py"

        self.assertTrue(
            verifySubmission._isDockerIgnored(path, ["src/instatarget/training/"])
        )
        self.assertFalse(
            verifySubmission._isDockerIgnored(
                path,
                ["src/instatarget/training/", "!src/instatarget/training/model.py"],
            )
        )

    def testDockerIgnoreKeepsOnlyRuntimeTrainingModules(self) -> None:
        rules = [
            "src/instatarget/training/*",
            "!src/instatarget/training/__init__.py",
            "!src/instatarget/training/model.py",
        ]

        self.assertFalse(
            verifySubmission._isDockerIgnored(
                "src/instatarget/training/model.py", rules
            )
        )
        self.assertTrue(
            verifySubmission._isDockerIgnored(
                "src/instatarget/training/dataset.py", rules
            )
        )

    @patch.object(verifySubmission.subprocess, "run")
    def testBuiltImageVerificationRunsRuntimeImportProbe(self, run) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='["1", "2", "3", "4", "5", "6", "7"]',
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="runtime imports and 123 checkpoint parameters verified\n",
            ),
        ]

        verifySubmission.verifyBuiltImage("model:v3")

        self.assertEqual(run.call_count, 2)
        runtimeCommand = run.call_args_list[1].args[0]
        self.assertIn("docker", runtimeCommand)
        self.assertIn("run", runtimeCommand)
        self.assertIn("validateHiTCheckpoint", runtimeCommand[-1])
        self.assertIn("--network", runtimeCommand)
        self.assertIn("none", runtimeCommand)


if __name__ == "__main__":
    unittest.main()
