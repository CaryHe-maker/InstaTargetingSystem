import unittest
from pathlib import Path

from instatarget.app.driver import buildRuntime, closeBackend
from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH
from instatarget.tracker.hit_backend import HiTPrediction

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RuntimeHiTWiringTest(unittest.TestCase):
    def testRgbOnlyCreatesOneHiTSession(self) -> None:
        factory = _SessionFactory()
        config = loadConfig(REPOSITORY_ROOT / "configs" / "RGBonly.yaml")

        runtime = buildRuntime(config, hitSessionFactory=factory)
        closeBackend(runtime.backend)

        self.assertEqual(factory.count, 1)

    def testRgbDepthCreatesIndependentRgbAndDepthHiTSessions(self) -> None:
        factory = _SessionFactory()
        config = loadConfig(REPOSITORY_ROOT / "configs" / "RGBD.yaml")

        runtime = buildRuntime(config, hitSessionFactory=factory)
        closeBackend(runtime.backend)

        self.assertEqual(factory.count, 2)


class _SessionFactory:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, _config):
        self.count += 1
        return _FakeHiTSession()


class _FakeHiTSession:
    supportsOnlineTemplates = True

    def encodeTemplate(self, _rgb, bbox: BBoxXYWH) -> object:
        return bbox

    def infer(self, _rgb, templateFeatures) -> HiTPrediction:
        return HiTPrediction(templateFeatures[-1], 0.9, 0.9)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
