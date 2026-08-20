"""Trainable HiT wrapper with explicit presence, quality, bbox, and heatmap outputs."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from instatarget.core.config import TrainingConfig
from instatarget.core.errors import ModelError


class HiTTrainingModel(nn.Module):
    """Add calibrated confidence heads without replacing HiT's spatial corner head."""

    def __init__(self, baseModel: nn.Module, hiddenDim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.baseModel = baseModel
        self.presenceHead = nn.Sequential(
            nn.LayerNorm(hiddenDim),
            nn.Linear(hiddenDim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.qualityHead = nn.Sequential(
            nn.LayerNorm(hiddenDim + 5),
            nn.Linear(hiddenDim + 5, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, template: Tensor, search: Tensor) -> dict[str, Tensor]:
        features = self.baseModel.forward_backbone(
            [search, template], first_score=None, threshold=0.9
        )
        output, _, outputEmbed = self.baseModel.forward_head(features)
        predBoxes = output["pred_boxes"]
        embedding = outputEmbed.squeeze(0).squeeze(1)
        if embedding.ndim != 2:
            raise ModelError(f"unexpected HiT output embedding shape: {tuple(outputEmbed.shape)}")
        cornerTl = output["corner_heatmap_tl"]
        cornerBr = output["corner_heatmap_br"]
        cornerStability = _cornerConcentration(cornerTl, cornerBr)
        presenceLogit = self.presenceHead(embedding).squeeze(-1)
        qualityFeatures = torch.cat(
            (embedding, predBoxes[:, 0, :].detach(), cornerStability.detach().unsqueeze(1)),
            dim=1,
        )
        qualityLogit = self.qualityHead(qualityFeatures).squeeze(-1)
        return {
            "predBoxes": predBoxes,
            "presenceLogit": presenceLogit,
            "qualityLogit": qualityLogit,
            "presenceProbability": presenceLogit.sigmoid(),
            "qualityProbability": qualityLogit.sigmoid(),
            "predictedIoU": qualityLogit.sigmoid(),
            "cornerHeatmapTl": cornerTl,
            "cornerHeatmapBr": cornerBr,
            "cornerStability": cornerStability,
            "outputEmbedding": embedding,
        }


def _cornerConcentration(cornerTl: Tensor, cornerBr: Tensor) -> Tensor:
    values = []
    for logits in (cornerTl, cornerBr):
        flat = logits.float().reshape(logits.shape[0], -1)
        probabilities = flat.softmax(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        normalizer = torch.log(
            torch.tensor(float(max(2, flat.shape[1])), device=flat.device)
        )
        values.append((1.0 - entropy / normalizer).clamp(0.0, 1.0))
    return torch.stack(values).mean(dim=0)


def buildTrainingModel(config: TrainingConfig) -> HiTTrainingModel:
    hitRoot = _resolveHitRoot()
    _activateVendorTree(hitRoot)
    try:
        import lib.models.HiT.backbone as backboneModule
        from lib.config.HiT.config import cfg, update_config_from_file
        from lib.models.HiT import build_hit
    except Exception as error:
        raise ModelError(f"cannot import bundled HiT training runtime: {error}") from error
    update_config_from_file(str(hitRoot / "configs" / "HiT_Small.yaml"))
    backboneModule.is_main_process = lambda: False
    baseModel = build_hit(cfg)
    model = HiTTrainingModel(
        baseModel,
        hiddenDim=config.model.hiddenDim,
        dropout=config.model.dropout,
    )
    checkpoint = _loadCheckpoint(config.model.initialWeights)
    if not isinstance(checkpoint.get("model"), dict):
        raise ModelError("training checkpoint must contain a Stage 3 'model' state")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


@dataclass(frozen=True, slots=True)
class ParameterGroupReport:
    name: str
    learningRate: float
    parameterNames: tuple[str, ...]
    parameterCount: int


def configureTrainingStage(
    model: HiTTrainingModel,
    config: TrainingConfig,
) -> tuple[list[dict[str, Any]], tuple[ParameterGroupReport, ...]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: dict[str, tuple[str, ...]] = {
        "heads": ("presenceHead.", "qualityHead."),
    }
    if config.model.stage >= 2:
        selected["neck_corner"] = ("baseModel.bottleneck.", "baseModel.box_head.")
    if config.model.stage >= 3:
        blockPrefix = "baseModel.backbone.body.blocks."
        starts = _backboneStageStarts(model.baseModel.backbone.body.blocks)
        firstStage = starts[-1] if config.model.stage == 3 else starts[-2]
        selected["backbone"] = tuple(
            f"{blockPrefix}{index}."
            for index in range(firstStage, len(model.baseModel.backbone.body.blocks))
        )

    rates = {
        "heads": config.optimization.headsLearningRate,
        "neck_corner": config.optimization.neckLearningRate,
        "backbone": config.optimization.backboneLearningRate,
    }
    groups: list[dict[str, Any]] = []
    reports: list[ParameterGroupReport] = []
    assigned: set[str] = set()
    named = dict(model.named_parameters())
    for groupName, prefixes in selected.items():
        names = tuple(
            name
            for name in named
            if name not in assigned and any(name.startswith(prefix) for prefix in prefixes)
        )
        if not names:
            raise ModelError(f"training parameter group is empty: {groupName}")
        for name in names:
            named[name].requires_grad_(True)
        assigned.update(names)
        parameters = [named[name] for name in names]
        groups.append(
            {
                "params": parameters,
                "lr": rates[groupName],
                "name": groupName,
            }
        )
        reports.append(
            ParameterGroupReport(
                name=groupName,
                learningRate=rates[groupName],
                parameterNames=names,
                parameterCount=sum(parameter.numel() for parameter in parameters),
            )
        )
    return groups, tuple(reports)


def keepFrozenNormalizationInEval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()


def _backboneStageStarts(blocks: nn.Sequential) -> tuple[int, ...]:
    boundaries = tuple(
        index
        for index, block in enumerate(blocks)
        if block.__class__.__name__ == "AttentionSubsample"
    )
    if not boundaries:
        raise ModelError("cannot identify HiT backbone stages: no AttentionSubsample blocks")
    starts = (0, *(index for index in boundaries))
    if len(starts) < 2:
        raise ModelError("HiT backbone must expose at least two stages")
    return starts


def _resolveHitRoot() -> Path:
    candidates = []
    if os.environ.get("HIT_ROOT"):
        candidates.append(Path(os.environ["HIT_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[1] / "vendor" / "hit")
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "lib" / "models" / "HiT").is_dir():
            return root
    raise ModelError(f"bundled HiT source is missing; checked={candidates}")


def _activateVendorTree(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _loadCheckpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelError(f"initial checkpoint does not exist: {path}")
    try:
        from lib.train.admin.local import EnvironmentSettings
        from lib.train.admin.settings import Settings
        from lib.train.admin.stats import AverageMeter, StatValue

        with torch.serialization.safe_globals(
            [AverageMeter, StatValue, Settings, EnvironmentSettings]
        ):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelError(f"cannot safely load training checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ModelError("training checkpoint root must be a mapping")
    return checkpoint


__all__ = [
    "HiTTrainingModel",
    "ParameterGroupReport",
    "buildTrainingModel",
    "configureTrainingStage",
    "keepFrozenNormalizationInEval",
]
