import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from instatarget.app.commands import REPOSITORY_ROOT, _formatInstanceIds, _resolveUserPath
from instatarget.app.track_airsim360 import EXIT_CONFIG
from instatarget.app.track_airsim360 import main as trackAirSim360Main


class AppCommandsTest(unittest.TestCase):
    def testAirSim360EntryWritesTimeArtifactWithoutChangingExitCode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resultRoot = Path(directory) / "result"

            code = trackAirSim360Main(
                [
                    "--dataset-root",
                    str(Path(directory) / "missing-dataset"),
                    "--target-instance",
                    "1",
                    "--output",
                    str(resultRoot / "tracking.txt"),
                    "--config",
                    str(Path(directory) / "missing-config.yaml"),
                ]
            )

            self.assertEqual(code, EXIT_CONFIG)
            payload = json.loads((resultRoot / "time.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "instatarget.time.v1")
            self.assertGreaterEqual(payload["elapsedNanoseconds"], 0)

    def testRepositoryStyleAbsolutePathIsResolvedInsideRepository(self) -> None:
        self.assertEqual(
            _resolveUserPath("/data/airsim360/nyc_sample"),
            (REPOSITORY_ROOT / "data/airsim360/nyc_sample").resolve(),
        )

    def testInstanceIdsAreGroupedByFirstFrameSemanticClass(self) -> None:
        instance = np.array([[0, 10, 10], [20, 20, 30]], dtype=np.int32)
        semantic = np.array([[0, 5, 5], [7, 7, 5]], dtype=np.int32)

        lines = _formatInstanceIds(instance, semantic, {5: "box", 7: "cone"})

        self.assertEqual(lines, ["box 1 10", "box 2 30", "", "cone 1 20"])


if __name__ == "__main__":
    unittest.main()
