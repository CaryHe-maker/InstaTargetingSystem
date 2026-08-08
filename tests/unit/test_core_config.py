import tempfile
import unittest
from pathlib import Path

from instatarget.core.config import loadConfig
from instatarget.core.errors import ConfigError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CoreConfigTest(unittest.TestCase):
    def testLoadConfigConvertsAnglesAndResolvesWeights(self) -> None:
        config = loadConfig(REPOSITORY_ROOT / "configs" / "RGBD.yaml")

        self.assertAlmostEqual(config.geometry.minFovRad, 0.3490658503988659)
        self.assertEqual(config.geometry.boundarySamplesPerEdge, 65)
        self.assertEqual(config.tracking.windowLength, 5)
        self.assertTrue(config.depth.enabled)
        self.assertEqual(config.model.weights, REPOSITORY_ROOT / "models" / "hit_small.pth")
        self.assertFalse(config.visualization.enabled)
        self.assertEqual(
            config.visualization.outputRoot,
            REPOSITORY_ROOT / "outputs" / "visualization",
        )
        self.assertEqual(
            config.visualization.stages,
            frozenset({"local_rgb", "depth_rgb", "backend_box", "geometry_box"}),
        )

    def testLoadConfigRejectsUnknownFields(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "RGBonly.yaml").read_text(encoding="utf-8")
        source = source.replace("schemaVersion: 1", "schemaVersion: 1\nunknownField: true")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)

    def testLoadConfigRejectsInvertedThresholds(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "RGBonly.yaml").read_text(encoding="utf-8")
        source = source.replace("uncertainThreshold: 0.45", "uncertainThreshold: 0.80")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)

    def testLoadConfigRejectsUnknownVisualizationStage(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "RGBonly.yaml").read_text(encoding="utf-8")
        source = source.replace("    - geometry_box", "    - geometry_box\n    - unknown")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)


if __name__ == "__main__":
    unittest.main()
