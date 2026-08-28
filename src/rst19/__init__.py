"""rst19 星图识别基线实现。

第一阶段提供不依赖在线服务的 FITS 读取、星点检测、切平面 WCS 和
先验星表匹配能力。完整的盲解算、测光绝对标定和运动目标轨迹关联仍需
在本地数据实验后继续补充。
"""

from .catalog import CatalogSource, load_catalog_csv
from .detection import Detection, DetectionResult, detect_sources
from .fits import FitsError, decode_auxiliary, read_fits
from .matching import CatalogMatch, MatchResult, match_detections
from .models import AuxiliaryData, FitsFrame
from .photometry import FaintestSource, find_faintest_source, instrumental_magnitude
from .sequence import SequenceResult, SourceTrack, TrackPoint, analyze_sequence, track_detections
from .wcs import TangentPlaneWCS

__all__ = [
    "AuxiliaryData",
    "CatalogMatch",
    "CatalogSource",
    "Detection",
    "DetectionResult",
    "FitsError",
    "FitsFrame",
    "FaintestSource",
    "MatchResult",
    "TangentPlaneWCS",
    "SequenceResult",
    "SourceTrack",
    "TrackPoint",
    "analyze_sequence",
    "decode_auxiliary",
    "detect_sources",
    "find_faintest_source",
    "instrumental_magnitude",
    "load_catalog_csv",
    "match_detections",
    "read_fits",
    "track_detections",
]

__version__ = "0.1.0"
