"""Directory-based frame source re-export."""

from instatarget.io.video_source import FrameSource, VideoFrameSource

DirectoryFrameSource = VideoFrameSource

__all__ = ["DirectoryFrameSource", "FrameSource", "VideoFrameSource"]
