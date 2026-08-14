import unittest
from pathlib import Path

import numpy as np

from instatarget.app.commands import REPOSITORY_ROOT, _formatInstanceIds, _resolveUserPath


class AppCommandsTest(unittest.TestCase):
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
