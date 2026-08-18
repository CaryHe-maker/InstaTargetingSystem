"""Project-wide exception hierarchy."""


class InstaTargetError(Exception):
    """Base class for expected InstaTargetingSystem failures."""


class ConfigError(InstaTargetError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class DecodeError(InstaTargetError):
    """Raised when an input frame cannot be decoded or validated."""


class GeometryError(InstaTargetError):
    """Raised when a spherical geometry operation cannot be completed."""


class ModelError(InstaTargetError):
    """Raised when a tracker model cannot load or infer."""


class ProtocolError(InstaTargetError):
    """Raised when frame ordering or a module protocol is violated."""


class OutputError(InstaTargetError):
    """Raised when results cannot be written or finalized."""
