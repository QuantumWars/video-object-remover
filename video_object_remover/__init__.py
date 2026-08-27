"""video-object-remover: remove moving or static objects from video with SAM 2 + ProPainter."""
from .config import Box, PipelineConfig
from .pipeline import run_pipeline

__version__ = "0.3.0"
__all__ = ["Box", "PipelineConfig", "run_pipeline", "__version__"]
