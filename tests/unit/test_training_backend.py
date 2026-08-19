import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from instatarget.core.config import TrainingLossConfig, loadTrainingConfig
from instatarget.core.errors import ConfigError
from instatarget.core.types import BBoxXYWH
from instatarget.geometry.spherical_geometry import SphericalGeometryImpl
from instatarget.training.dataset import (
    ManifestPairDataset,
    _contextSpec,
    _erpBoxToNormalizedLocal,
)
from instatarget.training.losses import computeTrainingLoss
from instatarget.training.manifest_builder import SequenceFiles, assignSequenceSplits
from instatarget.training.model import HiTTrainingModel, configureTrainingStage

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _lossConfig() -> TrainingLossConfig:
    return TrainingLossConfig(
        presenceWeight=1.0,
        l1Weight=5.0,
        giouWeight=2.0,
        qualityWeight=1.0,
        qualityNegativeWeight=0.25,
        presenceFocalGamma=0.0,
    )


class TrainingLossTest(unittest.TestCase):
    def testNegativeBatchMasksAllBoxLosses(self) -> None:
        outputs = {
            "predBoxes": torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2]], [[0.4, 0.4, 0.3, 0.3]]],
                requires_grad=True,
            ),
            "presenceLogit": torch.tensor([0.3, -0.2], requires_grad=True),
            "qualityLogit": torch.tensor([0.4, 0.1], requires_grad=True),
        }
        targets = {
            "boxes": torch.zeros((2, 4)),
            "present": torch.tensor([False, False]),
            "labelQuality": torch.ones(2),
        }

        losses = computeTrainingLoss(outputs, targets, _lossConfig())
        losses["total"].backward()

        self.assertEqual(float(losses["bboxL1"]), 0.0)
        self.assertEqual(float(losses["bboxGiou"]), 0.0)
        self.assertEqual(float(losses["positiveCount"]), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))

    def testPerfectPositiveBoxHasZeroLocalizationLoss(self) -> None:
        box = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
        losses = computeTrainingLoss(
            {
                "predBoxes": box[:, None, :],
                "presenceLogit": torch.tensor([1.0]),
                "qualityLogit": torch.tensor([1.0]),
            },
            {
                "boxes": box,
                "present": torch.tensor([True]),
                "labelQuality": torch.ones(1),
            },
            _lossConfig(),
        )

        self.assertAlmostEqual(float(losses["bboxL1"]), 0.0)
        self.assertAlmostEqual(float(losses["bboxGiou"]), 0.0)
        self.assertAlmostEqual(float(losses["qualityTargetMean"]), 1.0)


class ManifestAndGeometryTest(unittest.TestCase):
    def testSequenceSplitAssignmentIsDeterministicAndComplete(self) -> None:
        sequences = tuple(
            SequenceFiles(
                group="train_sim" if index < 20 else "train_real",
                sequenceId=f"sequence-{index}",
                videoPath=Path(f"video-{index}.mp4"),
                groundtruthPath=Path(f"groundtruth-{index}.txt"),
            )
            for index in range(40)
        )

        first = assignSequenceSplits(sequences, seed=7)
        second = assignSequenceSplits(tuple(reversed(sequences)), seed=7)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {item.sequenceId for item in sequences})
        self.assertEqual(set(first.values()), {"train", "validation", "calibration", "holdout"})

    def testTrainingConfigIsIndependentAndStrict(self) -> None:
        config = loadTrainingConfig(REPOSITORY_ROOT / "configs" / "train_backend.yaml")

        self.assertEqual(config.model.stage, 1)
        self.assertEqual(config.data.templateSizePx, 128)
        self.assertEqual(config.optimization.gradientAccumulation, 4)
        self.assertEqual(config.runtime.resume, None)

        source = (REPOSITORY_ROOT / "configs" / "train_backend.yaml").read_text(
            encoding="utf-8"
        )
        source = source.replace("  decoderCacheSize: 4", "  decoderCacheSize: 4\n  typo: 1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadTrainingConfig(path)

    def testManifestRejectsSequenceSplitLeakage(self) -> None:
        config = loadTrainingConfig(REPOSITORY_ROOT / "configs" / "train_backend.yaml")
        records = []
        for index, split in enumerate(("train", "validation")):
            records.append(
                {
                    "sequenceId": "same-sequence",
                    "videoPath": "video.mp4",
                    "frameIndex": index,
                    "timestamp": index / 30.0,
                    "targetInstanceId": 7,
                    "bbox": [10, 10, 20, 20],
                    "visible": True,
                    "occluded": False,
                    "truncated": False,
                    "width": 360,
                    "height": 180,
                    "labelSource": "manual",
                    "labelQuality": 1.0,
                    "split": split,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            dataConfig = replace(config.data, manifest=path)
            with self.assertRaises(ConfigError):
                ManifestPairDataset(dataConfig, "train", seed=1)

    def testCrossSeamAndPolarBoxesProjectIntoLocalTrainingViews(self) -> None:
        geometry = SphericalGeometryImpl()
        for bbox in (
            BBoxXYWH(xPx=350.0, yPx=70.0, widthPx=20.0, heightPx=20.0),
            BBoxXYWH(xPx=170.0, yPx=1.0, widthPx=20.0, heightPx=12.0),
        ):
            bfov = geometry.bboxToBfov(bbox, 360, 180)
            spec = _contextSpec(0, bfov, 256, contextFactor=2.0, maxFovDeg=120.0)
            local = _erpBoxToNormalizedLocal(bbox, spec, 360, 180)

            self.assertIsNotNone(local)
            assert local is not None
            self.assertTrue(((local >= 0.0) & (local <= 1.0)).all())
            self.assertGreater(float(local[2]), 0.0)
            self.assertGreater(float(local[3]), 0.0)


class Residual(nn.Linear):
    pass


class AttentionSubsample(nn.Linear):
    pass


class _Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(8, 8)
        self.blocks = nn.Sequential(
            Residual(8, 8),
            AttentionSubsample(8, 8),
            Residual(8, 8),
            AttentionSubsample(8, 8),
            Residual(8, 8),
        )


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = _Body()


class _BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.bottleneck = nn.Linear(8, 8)
        self.box_head = nn.Linear(8, 4)


class FreezeStrategyTest(unittest.TestCase):
    def testStageThreeTrainsOnlyLastBackboneStageAndHeads(self) -> None:
        config = loadTrainingConfig(REPOSITORY_ROOT / "configs" / "train_backend.yaml")
        config = replace(config, model=replace(config.model, stage=3, hiddenDim=8))
        model = HiTTrainingModel(_BaseModel(), hiddenDim=8)

        _, reports = configureTrainingStage(model, config)
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}

        self.assertTrue(any(name.startswith("presenceHead.") for name in trainable))
        self.assertTrue(any(name.startswith("baseModel.bottleneck.") for name in trainable))
        self.assertTrue(
            any(name.startswith("baseModel.backbone.body.blocks.3.") for name in trainable)
        )
        self.assertFalse(
            any(name.startswith("baseModel.backbone.body.blocks.2.") for name in trainable)
        )
        self.assertFalse(
            any(name.startswith("baseModel.backbone.body.patch_embed.") for name in trainable)
        )
        self.assertEqual({report.name for report in reports}, {"heads", "neck_corner", "backbone"})


if __name__ == "__main__":
    unittest.main()
