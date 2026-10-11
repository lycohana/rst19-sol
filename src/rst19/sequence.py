"""15 帧星点稳定性和运动目标关联。

这里不把帧间最近邻直接叫作运动目标。先用高质量、高 SNR 源估计每一帧
相对于上一帧的全局平移，再用跨帧稳定源拟合小范围的全局仿射残差，最后
在配准坐标中关联检测源；只有具有足够帧数、并且相对于静态背景呈现可
重复位移的轨迹，才标为 ``moving``。仿射层只处理小旋转/尺度残差，模型
证据不足时会退回平移层，仍不替代带畸变项的完整图像配准。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.special import erf
from scipy.spatial import cKDTree

from .detection import (
    Detection,
    DetectionResult,
    _linear_artifact_mask,
    _working_mask,
    build_background_model,
    detect_sources,
    sigma_clipped_stats,
)
from .fits import auxiliary_mask, exposure_milliseconds, exposure_seconds, read_fits
from .models import FitsFrame
from .pipeline import FrameAnalysis, analyze_frame
from .photometry import instrumental_magnitude
from .relative_photometry import Observation, RelativePhotometryResult, fit_relative_photometry
from .single_frame_trails import (
    detect_local_pixel_trail_proposals,
    detect_single_frame_trail_proposals,
)


# 15 帧关联的默认工作集上限。候选总数仍然完整统计，只有进入跨帧
# 配准/点轨迹关联的源级测量数量受限；单图分析不使用这个上限。
DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT = 6000
# 叠加暗星是补充层，不应把参考图中数万条低优先级纹理全部带回每一帧
# 做强制测光。候选仍会在参考图检测结果中完整计数；只有进入逐帧复核的
# 工作集按参考 SNR 从高到低截取。需要研究极限暗星时，调用方可传 None。
DEFAULT_SEQUENCE_STACK_WORKING_LIMIT = 8000
# 当前 4096×4096 数据在开发机实测 4 个 worker 略快于 3 个；仍提供
# CLI ``--workers 1/2/3`` 给内存紧张或需要降低峰值带宽争用的机器。
DEFAULT_SEQUENCE_WORKERS = 4
DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT = 4_096
# 相邻帧的视场漂移不能假定永远小于轨迹关联半径。当前实测数据在
# F04→F05 等相邻帧出现了约 18 px 的平移，若仍把 8 px 同时当作“搜索
# 半径”，注册器会用密集星场里的错误近邻替代真正对应。这里把搜索和
# 最终轨迹关联分开：只放宽注册候选搜索，再用位移一致簇确认结果。
DEFAULT_REGISTRATION_SEARCH_RADIUS_PX = 64.0
REGISTRATION_CONSENSUS_RADIUS_PX = 2.5
# 平移注册完成后，再用跨帧稳定源拟合一个低自由度的全局仿射残差。
# 这不是把任意形变都吸收到“稳定星场”里：只有源数足够、空间上有
# 非共线分布、且 RANSAC 内点支持充分时才启用；否则继续使用平移模型。
REGISTRATION_AFFINE_MIN_PAIRS = 12
REGISTRATION_AFFINE_RANSAC_TRIALS = 160
REGISTRATION_AFFINE_INLIER_RADIUS_PX = 1.5
REGISTRATION_AFFINE_MAX_SCALE_DEVIATION = 0.05
REGISTRATION_AFFINE_MAX_SKEW = 0.05
# 经过仿射校正后，仍可能有少量亚像素质心噪声。运动判定的自适应
# 下限使用稳定多数的 robust median + 5*MAD；这是“不可由当前配准
# 解释的最小位移”，不是物理速度真值，也不会覆盖用户显式给出的更高门槛。
REGISTRATION_MOTION_FLOOR_MAD_FACTOR = 5.0
REGISTRATION_MOTION_FLOOR_MIN_TRACKS = 30
# 单张曝光的线状候选只承担“不要漏掉疑似拖影”的预检职责，因此比
# 15 帧序列的确认层略宽；它不会改变星点质量层，也不会把结果直接命名为
# moving。短线预检允许适度降低轴比；单帧默认不跨像素缺口桥接，避免把
# 两个相邻点源先合成一个连通域。残差阈值、面积和长度仍保持较高门槛，
# 避免把普通点源直接升级为线状候选。15 帧序列层有独立的缺口桥接和
# 跨帧证据门，不能把两个工作区的默认值混用。
DEFAULT_SINGLE_FRAME_TRAIL_RESIDUAL_SIGMA = 10.0
DEFAULT_SINGLE_FRAME_TRAIL_MIN_RESIDUAL_ADU = 60.0
DEFAULT_SINGLE_FRAME_TRAIL_MIN_AREA = 20
DEFAULT_SINGLE_FRAME_TRAIL_MIN_AXIS_RATIO = 2.5
DEFAULT_SINGLE_FRAME_TRAIL_MIN_LENGTH = 8.0
DEFAULT_SINGLE_FRAME_TRAIL_BRIDGE_GAPS = False
# 1.5 s 曝光内的短拖影可能只有约 3--5 个 PSF 宽度。序列层不能再用
# 18 px 的固定门把它们在逐帧提取阶段删掉；短线会在跨帧形态、位移和拟合
# 阶段继续接受更严格的检查，不能因为长度门降低就直接成为 moving。
DEFAULT_SEQUENCE_TRAIL_MIN_LENGTH = 8.0
# 15 帧宽筛也允许一次相邻像素缺口合并。合并只改变连通域的分组，不把
# 补出的像素计入面积、PCA 或亮度；逐帧形态门仍会检查原始残差的连续性。
DEFAULT_SEQUENCE_TRAIL_BRIDGE_GAPS = True
MOTION_FEATURE_SHORT_TRAIL_LENGTH = 16.0
MOTION_FEATURE_SHORT_TRAIL_MIN_AREA = 20
MOTION_FEATURE_SHORT_TRAIL_MIN_AXIS_RATIO = 2.5
MOTION_FEATURE_SHORT_TRAIL_MIN_PROFILE_FILL_RATIO = 0.80
MOTION_FEATURE_SHORT_TRAIL_MIN_CONTINUITY_RATIO = 0.80
# PCA 只能说明“这些高亮像素大致排成一条轴”，不能区分连续拖影和
# 两个相邻点源。下面的相对剖面填充度要求线段中段也有持续亮度支撑；
# 它只用于线形证据和 moving 的最后一道形态门，不改变星点总数。
MOTION_FEATURE_MIN_PROFILE_FILL_RATIO = 0.45
MOTION_FEATURE_MIN_CONTINUITY_RATIO = 0.45
# 8--15 px 的单帧结构容易被几个相邻点源拉成“线”。短线复核层只用于
# 单帧曝光候选的可见性，不改变 15 帧 moving 的严格形态门；相较普通线
# 候选，它要求线段中段和绝对连续支撑都更高。
SINGLE_FRAME_SHORT_TRAIL_MIN_LENGTH = 8.0
SINGLE_FRAME_SHORT_TRAIL_MIN_PROFILE_FILL_RATIO = 0.80
SINGLE_FRAME_SHORT_TRAIL_MIN_CONTINUITY_RATIO = 0.80
# 跨帧线段除了要逐帧有连续亮度支撑，还要保持同一条物理拖影的几何形态。
# 方向按 180° 周期处理（线没有正反方向）；长度门只防止“某一帧突然变成
# 很短的点状结构”，不要求边界截断的首帧与完整帧等长。
MOTION_FEATURE_MAX_ANGLE_DEVIATION_DEG = 15.0
MOTION_FEATURE_MAX_LENGTH_DEVIATION_FRACTION = 0.75
# 线段是曝光内运动的积分形状，因此其无向轴还应与跨帧中心位移
# 大致平行。这个门只用于最终 moving 确认，采用 25° 的宽容角和多数
# 过渡规则；不符合的结构保留为 candidate 供人工复核，不从候选池删除。
MOTION_FEATURE_MAX_MOTION_AXIS_DEVIATION_DEG = 25.0
MOTION_FEATURE_MIN_MOTION_AXIS_PASS_FRACTION = 0.60
# 对比赛的长序列，少数几帧共线不能直接叫作 moving。短测试序列仍保留
# 至少 3 帧的通用门；当输入达到 8 帧时，确认门随序列长度按至少一半
# 观测帧缩放。未通过确认门的结果仍保留为 candidate，供人工复核，
# 不会从审计中消失。
MOTION_FEATURE_LONG_SEQUENCE_MIN_FRAMES = 8
MOTION_FEATURE_CONFIRMATION_PRESENCE_FRACTION = 0.5
# 一条线从画面边缘进入或离开时，首/末帧的阈值连通域可能停在边界前
# 1--数个像素，不能只用 ``touches_edge`` 的精确相等判断。边界截断只
# 放宽一帧的覆盖门，仍必须满足连续帧、位移、拟合、形态和曝光一致性。
MOTION_FEATURE_EDGE_TRUNCATION_MARGIN_PX = 4.0
MOTION_FEATURE_EDGE_MIN_PRESENCE_RELAXATION = 1
# 内部缺帧不走边缘截断例外。只有两侧各有足够连续观测、总缺口很短，
# 且所有观测点都通过原图线模型/运动轴/曝光速度细筛时，才允许把同一
# 条物理拖影重新接起来；否则仍保持 candidate。
MOTION_FEATURE_INTERNAL_GAP_BRIDGE_MAX_FRAMES = 2
MOTION_FEATURE_INTERNAL_GAP_BRIDGE_MIN_RUN = 3
# 回接后的轨迹可能还带有少量被常速度预测支持的端点 singleton；它们
# 不应把两个强连续观测段误判成“三段互不相关的轨迹”。数量必须很小，
# 且最终仍要通过完整的形态、拟合、方向、曝光和伪影门。
MOTION_FEATURE_INTERNAL_GAP_BRIDGE_MAX_ISOLATED_POINTS = 2
# 有明确 ``DATE-OBS`` 和曝光时长时，线段长度还要和跨帧位移在同一
# 运动速度量级上。``length_px`` 是 PSF 卷积后的 PCA 尺度，不是裸的
# 目标位移；下面的比值门先扣除圆形 PSF 的二阶矩，再允许端点阈值、
# 边界截断和质心偏差带来的宽松误差。它是物理一致性证据，不是速度
# 真值；缺少头部时间/曝光时只记录为未知，不强行拒绝候选。
MOTION_FEATURE_EXPOSURE_RATIO_MIN = 0.35
MOTION_FEATURE_EXPOSURE_RATIO_MAX = 2.50
MOTION_FEATURE_EXPOSURE_MAX_LOG_MAD = 0.45
MOTION_FEATURE_EXPOSURE_MIN_PASS_FRACTION = 0.75
MOTION_FEATURE_EXPOSURE_MIN_TRANSITIONS = 2
# ``LINE_ARTIFACT`` 是已知的固定线状伪影提示。它不能在候选发现阶段
# 直接删掉真实目标，但如果一条跨帧轨迹的大多数可用观测都与该掩膜高度
# 重叠，就不能再把它升级为 moving；保留为 candidate 供人工复核。
MOTION_FEATURE_ARTIFACT_OVERLAP_FRACTION = 0.90
MOTION_FEATURE_MIN_ARTIFACT_CLEAR_FRACTION = 0.50
# 关联阶段的曝光速度门比最终 moving 判定略宽。它只排除“候选线段却
# 配到几乎静止的点”以及明显超出速度量级的近邻；剩下的结果仍要经过
# 轨迹拟合、形态多数和最终曝光一致性门。缺少头部元数据时不启用。
MOTION_FEATURE_LINK_EXPOSURE_RATIO_MIN = 0.25
MOTION_FEATURE_LINK_EXPOSURE_RATIO_MAX = 2.50
# 单帧短线若能被两个已通过点源质量门控的 PSF 解释，就不应继续显示为
# “曝光运动候选”。这个上限只作用于短形状；真正的长拖影交给线状/时序层。
DEFAULT_SINGLE_FRAME_BLEND_GUARD_MAX_LENGTH = 32.0
DEFAULT_SINGLE_FRAME_BLEND_GUARD_MARGIN = 2.0
# 单帧局部模型只对短/中等长度候选做精测。超长线需要更大的局部窗口，
# 会把 4096² 图上的预检拖慢；它们仍保留几何证据和伪影审计，不因跳过
# 局部模型而被误报成“单星”。
SINGLE_FRAME_MODEL_MAX_LENGTH_PX = 128.0
SINGLE_FRAME_MODEL_MIN_SNR = 5.0
SINGLE_FRAME_MODEL_MIN_DELTA_BIC = 6.0
# ΔBIC < 2 按照常用的模型比较解释只算“几乎没有证据”；这类线模型
# 即使略胜单星/双星/椭圆模型，也不应在宽筛复核层画成 EXP TRAIL?。
# 2 <= ΔBIC < 6 仍可作为人工复核候选，>= 6 才进入强线层。
SINGLE_FRAME_REVIEW_MIN_DELTA_BIC = 2.0
# 15 帧关联的入口不是最终结论。真实拖影在某些帧会被背景、饱和或
# 配准残差削弱，如果必须每帧都满足 6 的强模型差异，目标会在关联前
# 消失。这里允许“线模型是局部最佳、但证据尚未达到强线”的 review
# 点进入跨帧关联；最终 moving 仍要经过轨迹形态、位移、拟合、曝光
# 速度和伪影审计，且 compact-source winner 不得使用这条入口。
SEQUENCE_REVIEW_MIN_MODEL_SNR = SINGLE_FRAME_MODEL_MIN_SNR
SEQUENCE_REVIEW_MIN_SHAPE_RATIO = MOTION_FEATURE_MIN_PROFILE_FILL_RATIO
SEQUENCE_REVIEW_MIN_LENGTH_PX = 6.0
# 多尺度线段提案在序列中会先在时间残差上做一次宽筛模型竞争，之后还
# 必须回到当前帧原图做最终模型复核。限制前一轮每帧的模型评估预算，
# 避免几十个重复尺度提案把 4096² 序列拖慢；原图连通域候选和单帧
# ``detect_single_frame_long_trails`` 不使用这个预算。三类提案的预算在
# 调用点分别分给原生短线、伪影区复核和降采样长线，保留短线召回优先级。
DEFAULT_SEQUENCE_LINE_MODEL_PROPOSAL_BUDGET = 48
# 超过局部模型窗口的线段不再返回无解释的 ``not_fit``。这两个门只
# 作为整段剖面模型的内部证据，最终是否显示仍要经过连续性和伪影复核。
SINGLE_FRAME_LONG_PROFILE_MIN_SNR = 5.0
SINGLE_FRAME_LONG_PROFILE_MIN_DELTA_BIC = 6.0
# 有限线段模型只允许补回“几何门因阈值断裂”的候选，不能替代像素
# 连续性。短线的采样本来就少，但 0.35 的门会把两个相邻亮点的连线
# 也放成强橙线；把实线门提高到 0.60，低于它的结构仍留在候选审计层。
SINGLE_FRAME_MODEL_RESCUE_MIN_PROFILE_FILL_RATIO = 0.60
SINGLE_FRAME_MODEL_RESCUE_MIN_CONTINUITY_RATIO = 0.60
# 线检测器会把任何强直线（包括真实拖影）标进伪影候选图，因此不能
# 静默删除高重叠对象；但它也不能与未受该提示的对象共用“强线”状态。
# 高重叠对象仍保留在单帧候选层，交给人工或 15 帧独立证据复核。
SINGLE_FRAME_ARTIFACT_REVIEW_OVERLAP_FRACTION = 0.75
# 时间参考图只用于“宽筛提案”。当前数据的亮源实测 FWHM 约在 1.9 px
# 附近，而序列默认检测器参数是 3 px；只用一个尺度会让窄 PSF 源在参考图
# 中被匹配滤波稀释。用相对基准 FWHM 的小型尺度组覆盖窄/基准/宽三类
# 响应，后续仍必须通过逐帧原图支持、形状和持续性细筛。
TEMPORAL_PSF_FWHM_FACTORS = (0.65, 1.0, 1.35)

# 点状高速目标与固定星场使用不同的关联尺度。普通星场关联仍保持
# ``link_radius_px=4``，避免在密集星场中把相邻恒星串成轨迹；高速点源
# 先用相邻帧的速度种子宽筛，再用常速度拟合和逐帧质量源细筛。
DEFAULT_FAST_POINT_MIN_SNR = 7.0
DEFAULT_FAST_POINT_MAX_STEP_PX = 20.0
DEFAULT_FAST_POINT_GATE_PX = 4.5
DEFAULT_FAST_POINT_FIT_RMS_PX = 1.5
# 宽筛候选峰可能有数万条/帧。高速点源关联不需要把全部候选都做
# 源级测光；按匹配滤波响应取前一层工作集，再对形成轨迹的少量点回到
# 原图细筛。这个上限独立于单帧星点输出上限，也不改变 candidate_count。
DEFAULT_FAST_POINT_CANDIDATE_WORKING_LIMIT = 20_000
FAST_POINT_MOTION_EVIDENCE = "fast_point_motion"


@dataclass(frozen=True, slots=True)
class _FastCandidateRecord:
    """一个尚未做原图孔径测光的宽筛候选峰。"""

    row_index: int
    raw_x: float
    raw_y: float
    aligned_x: float
    aligned_y: float
    filter_snr: float


def _temporal_psf_fwhm_bank(psf_fwhm: float) -> tuple[float, ...]:
    """返回时间宽筛使用的 PSF FWHM 尺度组。"""

    if psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive")
    return tuple(
        sorted(
            {
                round(max(float(psf_fwhm) * factor, 0.8), 6)
                for factor in TEMPORAL_PSF_FWHM_FACTORS
            }
        )
    )


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """轨迹中的一次源观测；aligned 坐标已扣除全局帧间平移。"""

    frame_index: int
    detection_id: int
    x: float
    y: float
    aligned_x: float
    aligned_y: float
    flux_snr: float | None
    quality_passed: bool = True
    candidate_snr: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "aligned_x": self.aligned_x,
            "aligned_y": self.aligned_y,
            "flux_snr": self.flux_snr,
            "quality_passed": self.quality_passed,
            "candidate_snr": self.candidate_snr,
        }


@dataclass(frozen=True, slots=True)
class SourceTrack:
    """一条跨帧轨迹及其静态/持续/运动判定。"""

    track_id: int
    classification: str
    points: tuple[TrackPoint, ...]
    displacement_px: float
    speed_px_per_frame: float
    fit_rms_px: float | None
    evidence_level: str = "quality"

    @property
    def presence(self) -> int:
        return len(self.points)

    @property
    def quality_presence(self) -> int:
        """轨迹中通过单帧质量规则的观测次数。"""

        return sum(point.quality_passed for point in self.points)

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "classification": self.classification,
            "presence": self.presence,
            "displacement_px": self.displacement_px,
            "speed_px_per_frame": self.speed_px_per_frame,
            "fit_rms_px": self.fit_rms_px,
            "quality_presence": self.quality_presence,
            "evidence_level": self.evidence_level,
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class MotionFeaturePoint:
    """一帧中被时序/形状规则检出的线状运动候选。

    ``continuity_ratio`` 看绝对阈值上的最长连续段；``profile_fill_ratio``
    看相对峰值门槛上的整体填充。后者专门防止相邻点源被 PCA 误拟合成
    一条“线”。
    """

    frame_index: int
    x: float
    y: float
    aligned_x: float
    aligned_y: float
    residual_snr: float
    area_pixels: int
    length_px: float
    width_px: float
    angle_deg: float
    bbox: tuple[int, int, int, int]
    touches_edge: bool
    # 沿候选主轴逐点抽样后，连续超过线状残差阈值的最长比例。它是
    # 形态证据，不是运动确认；旧缓存没有该字段时按 None 处理。
    continuity_ratio: float | None = None
    # 沿主轴的采样剖面中，超过 max(绝对阈值, 峰值的 25%) 的比例。
    # 相邻双星通常只有两个峰，中段填充度很低；连续拖影则应有较高值。
    profile_fill_ratio: float | None = None
    # 当前线状候选的阈值支撑像素中，有多少比例落在点源检测层的
    # ``LINE_ARTIFACT`` 掩膜内。它是复核提示，不是硬删除条件：真实拖影
    # 也可能被通用线状伪影筛选命中，最终仍需跨帧位移证据区分。
    # ``None`` 表示该对象没有可用的线状伪影审计；0.0 才表示已经
    # 检查过且阈值支撑像素没有落入掩膜。两者不能混为一谈。
    line_artifact_overlap_fraction: float | None = None
    # 以下字段只由独立单帧路线的局部模型诊断填写。模型比较固定当前
    # 几何提案（中心、方向、PSF 宽度），因此是“模板证据”，不是已完成
    # 的全自由度物理拟合；旧序列缓存没有这些字段时按 None 处理。
    line_template_snr: float | None = None
    line_vs_point_delta_bic: float | None = None
    line_vs_double_delta_bic: float | None = None
    line_vs_background_delta_bic: float | None = None
    line_vs_elliptical_delta_bic: float | None = None
    single_frame_model: str | None = None
    single_frame_model_margin_bic: float | None = None
    # ``detected`` 是常规单帧观测；``recovered_by_predicted_roi`` 表示在
    # 已有运动轨迹预测的小窗口内重新测得，必须与纯外推位置区分。
    observation_source: str = "detected"

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "x": self.x,
            "y": self.y,
            "aligned_x": self.aligned_x,
            "aligned_y": self.aligned_y,
            "residual_snr": self.residual_snr,
            "area_pixels": self.area_pixels,
            "length_px": self.length_px,
            "width_px": self.width_px,
            "angle_deg": self.angle_deg,
            "bbox": list(self.bbox),
            "touches_edge": self.touches_edge,
            "continuity_ratio": self.continuity_ratio,
            "profile_fill_ratio": self.profile_fill_ratio,
            "line_artifact_overlap_fraction": self.line_artifact_overlap_fraction,
            "line_template_snr": self.line_template_snr,
            "line_vs_point_delta_bic": self.line_vs_point_delta_bic,
            "line_vs_double_delta_bic": self.line_vs_double_delta_bic,
            "line_vs_background_delta_bic": self.line_vs_background_delta_bic,
            "line_vs_elliptical_delta_bic": self.line_vs_elliptical_delta_bic,
            "single_frame_model": self.single_frame_model,
            "single_frame_model_margin_bic": self.single_frame_model_margin_bic,
            "observation_source": self.observation_source,
        }


def _line_profile_metrics(
    residual: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    width_px: float,
    angle_deg: float,
    threshold_adu: float,
) -> tuple[float | None, float | None]:
    """测量候选主轴上的绝对连续性和相对剖面填充度。

    ``detect_motion_features`` 的候选来自阈值残差连通域，但 PCA 的长度
    可能被两个相邻点源拉长。这里沿 PCA 主轴以约 0.5 px 间隔采样，在
    候选宽度附近取最大残差，再计算最长连续支撑段占线段长度的比例。
    取横截面最大值是为了不因亚像素中心偏移把真实细线打断；最终仍只
    记录诊断量，不把它当成“运动真值”。
    """

    values = np.asarray(residual)
    if values.ndim != 2 or length_px <= 0 or threshold_adu <= 0:
        return None, None
    sample_count = max(25, int(np.ceil(float(length_px) * 2.0)) + 1)
    along = np.linspace(-0.5 * float(length_px), 0.5 * float(length_px), sample_count)
    half_width = max(1.5, min(4.0, 0.5 * float(width_px) + 1.0))
    across = np.linspace(-half_width, half_width, 9)
    angle = np.deg2rad(float(angle_deg))
    unit_x = float(np.cos(angle))
    unit_y = float(np.sin(angle))
    normal_x = -unit_y
    normal_y = unit_x
    sample_x = float(center_x) + along[:, None] * unit_x + across[None, :] * normal_x
    sample_y = float(center_y) + along[:, None] * unit_y + across[None, :] * normal_y
    sampled = ndimage.map_coordinates(
        np.asarray(values, dtype=np.float32),
        [sample_y.T, sample_x.T],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    finite_sampled = np.where(np.isfinite(sampled), sampled, -np.inf)
    profile = np.max(finite_sampled, axis=0)
    valid = np.isfinite(profile)
    profile = np.where(valid, profile, np.nan)
    if not np.any(valid):
        return None, None
    support = valid & (profile >= float(threshold_adu))
    # 约 0.5 px 采样允许一个像素级缺口，但不会把两段相距较远的
    # 点源连接起来；后面的最长连续段仍以原始采样为单位计算。
    bridged = support.copy()
    bridged[1:-1] |= support[:-2] & support[2:]
    longest = 0
    current = 0
    for is_supported in bridged:
        current = current + 1 if is_supported else 0
        longest = max(longest, current)
    valid_count = int(np.count_nonzero(valid))
    continuity_ratio = float(longest / valid_count) if valid_count else None
    peak = float(np.max(profile[valid])) if np.any(valid) else 0.0
    relative_threshold = max(float(threshold_adu), 0.25 * peak)
    relative_support = valid & (profile >= relative_threshold)
    profile_fill_ratio = (
        float(np.count_nonzero(relative_support) / valid_count)
        if valid_count
        else None
    )
    return continuity_ratio, profile_fill_ratio


def _single_frame_model_bic(rss: float, sample_count: int, parameter_count: int) -> float:
    """Return the Gaussian-noise BIC used for relative local model ranking."""

    if sample_count <= 0 or parameter_count < 0 or not np.isfinite(rss):
        return float("inf")
    return float(
        sample_count * np.log(max(float(rss), np.finfo(np.float64).tiny) / sample_count)
        + parameter_count * np.log(sample_count)
    )


def _finite_line_template(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    angle_deg: float,
    psf_sigma: float,
) -> np.ndarray:
    """Build a normalized PSF convolved with a finite straight exposure line.

    Long segments use an analytic line integral in coordinates parallel and
    perpendicular to the proposed segment.  The previous implementation
    sampled up to 257 Gaussian PSFs along every segment and materialised a
    ``(pixels, 1, samples)`` temporary for every model comparison.  That was
    unnecessarily expensive for 4096² sequence data and could multiply the
    peak memory of the line-review workers.  Short segments retain the
    historical sampled form because the discrete-pixel approximation is more
    faithful for the 8--32 px candidates used by the existing negative-control
    tests; long candidates are already dominated by the continuous profile and
    benefit most from the bounded-memory analytic form.
    """

    safe_length = max(0.0, float(length_px))
    sigma = max(float(psf_sigma), 0.35)
    angle = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    if safe_length <= 32.0:
        sample_count = max(9, min(257, int(np.ceil(safe_length * 2.0)) + 1))
        offsets = np.linspace(-0.5 * safe_length, 0.5 * safe_length, sample_count)
        centers_x = float(center_x) + offsets * cosine
        centers_y = float(center_y) + offsets * sine
        distance_sq = (
            np.asarray(x_grid, dtype=np.float64)[..., None]
            - centers_x[None, None, :]
        ) ** 2 + (
            np.asarray(y_grid, dtype=np.float64)[..., None]
            - centers_y[None, None, :]
        ) ** 2
        template = np.mean(np.exp(-0.5 * distance_sq / (sigma * sigma)), axis=-1)
        peak = float(np.max(template))
        return template / peak if peak > 0.0 and np.isfinite(peak) else template
    along = (np.asarray(x_grid, dtype=np.float64) - float(center_x)) * cosine + (
        np.asarray(y_grid, dtype=np.float64) - float(center_y)
    ) * sine
    across = -(
        np.asarray(x_grid, dtype=np.float64) - float(center_x)
    ) * sine + (np.asarray(y_grid, dtype=np.float64) - float(center_y)) * cosine
    edge_scale = np.sqrt(2.0) * sigma
    template = 0.5 * (
        erf((along + 0.5 * safe_length) / edge_scale)
        - erf((along - 0.5 * safe_length) / edge_scale)
    ) * np.exp(-0.5 * (across / sigma) ** 2)
    peak = float(np.max(template))
    return template / peak if peak > 0.0 and np.isfinite(peak) else template


def _fit_single_frame_template(
    data: np.ndarray,
    templates: Sequence[np.ndarray],
    *,
    noise: float,
    background_terms: Sequence[np.ndarray] = (),
) -> tuple[float, float, float]:
    """Fit a local background basis plus non-negative template amplitudes.

    Geometry is held fixed at the proposal values.  The returned tuple is
    ``(weighted_rss, first_amplitude_snr, baseline)``.  Keeping the baseline
    free prevents a local background offset from being mistaken for a line;
    callers may provide centered linear terms to absorb a mild gradient.
    Clipping amplitudes prevents a template from winning only by subtracting
    signal.
    """

    values = np.asarray(data, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("inf"), 0.0, 0.0
    background_columns = [np.ones(values.size, dtype=np.float64)]
    for term in background_terms:
        background_columns.append(np.asarray(term, dtype=np.float64).reshape(-1))
    background_count = len(background_columns)
    safe_noise = max(float(noise), np.finfo(np.float64).eps)
    template_arrays = [np.asarray(template, dtype=np.float64).reshape(-1) for template in templates]
    if any(template.size != values.size for template in template_arrays):
        raise ValueError("template and data sizes must match")

    # 模板振幅物理上不能为负。简单地把无约束解的负振幅截成 0，
    # 却保留原来的背景/其它振幅，并不是非负最小二乘；它会让 double
    # 模型的 RSS 偏大，进而把静态双星推向 line。当前模板最多两个，
    # 枚举活动集既足够快，又能在每个活动集上重新拟合背景和其余振幅。
    best_fit: tuple[float, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    template_count = len(template_arrays)
    for active_mask in range(1 << template_count):
        active_indices = tuple(
            index for index in range(template_count) if active_mask & (1 << index)
        )
        design = np.column_stack(
            [*background_columns, *(template_arrays[index] for index in active_indices)]
        )
        gram = design.T @ design
        rhs = design.T @ values
        try:
            coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            try:
                coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            except np.linalg.LinAlgError:
                continue
        coefficients = np.asarray(coefficients, dtype=np.float64)
        amplitudes = coefficients[background_count:]
        tolerance = 1e-9 * max(1.0, float(np.max(np.abs(coefficients))))
        if amplitudes.size and np.any(amplitudes < -tolerance):
            continue
        # 只清除数值误差造成的微小负值；显著负振幅的活动集已被跳过。
        coefficients[background_count:] = np.maximum(amplitudes, 0.0)
        residual = values - design @ coefficients
        weighted_rss = float(np.sum((residual / safe_noise) ** 2))
        if best_fit is None or weighted_rss < best_fit[0]:
            best_fit = (weighted_rss, active_indices, coefficients, design)

    if best_fit is None:
        return float("inf"), 0.0, 0.0
    weighted_rss, active_indices, coefficients, design = best_fit
    baseline = float(coefficients[0]) if coefficients.size else 0.0
    if not template_count or 0 not in active_indices:
        return weighted_rss, 0.0, baseline

    # 用完整活动集设计矩阵的协方差估计第一个模板的振幅显著性，
    # 同时消除局部背景梯度以及其它活动模板带来的相关性。
    active_position = background_count + active_indices.index(0)
    try:
        gram_inverse = np.linalg.solve(design.T @ design, np.eye(design.shape[1], dtype=np.float64))
        first_variance = float(gram_inverse[active_position, active_position]) * safe_noise**2
    except np.linalg.LinAlgError:
        first_variance = float("inf")
    first_snr = (
        float(coefficients[active_position]) / np.sqrt(first_variance)
        if first_variance > 0.0 and np.isfinite(first_variance)
        else 0.0
    )
    return weighted_rss, first_snr, baseline


def _single_frame_line_model_evidence(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    width_px: float,
    angle_deg: float,
    psf_fwhm: float,
    noise: float,
) -> dict[str, float | str | None]:
    """Compare background, point, double-point and finite-line templates.

    This is deliberately a diagnostic layer for the single-frame route.  The
    proposal geometry is fixed, the line length is searched only over a small
    neighbourhood, and no result is promoted to cross-frame ``moving``.  A
    positive delta-BIC means the line template has a smaller local BIC than the
    named competitor; it is not a probability of physical reality.
    """

    values = np.asarray(residual)
    validity = np.asarray(valid, dtype=bool)
    if (
        values.ndim != 2
        or validity.shape != values.shape
        or not np.isfinite(length_px)
        or length_px <= 0.0
        or length_px > SINGLE_FRAME_MODEL_MAX_LENGTH_PX
        or psf_fwhm <= 0.0
        or noise <= 0.0
    ):
        return {
            "line_template_snr": None,
            "line_vs_point_delta_bic": None,
            "line_vs_double_delta_bic": None,
            "line_vs_background_delta_bic": None,
            "single_frame_model": "not_fit",
            "single_frame_model_margin_bic": None,
        }

    sigma = max(float(psf_fwhm) / 2.354820045, 0.45)
    # A line wider than a point source still needs enough context for the
    # competing point/double models, but cap the patch to keep 4096² scans
    # bounded when a noisy component has an inflated PCA length.
    radius = max(6.0, min(96.0, 0.5 * float(length_px) + 4.0 * sigma + 3.0))
    x0 = max(0, int(np.floor(float(center_x) - radius)))
    x1 = min(values.shape[1], int(np.ceil(float(center_x) + radius + 1.0)))
    y0 = max(0, int(np.floor(float(center_y) - radius)))
    y1 = min(values.shape[0], int(np.ceil(float(center_y) + radius + 1.0)))
    if x1 - x0 < 7 or y1 - y0 < 7:
        return {
            "line_template_snr": None,
            "line_vs_point_delta_bic": None,
            "line_vs_double_delta_bic": None,
            "line_vs_background_delta_bic": None,
            "single_frame_model": "not_fit",
            "single_frame_model_margin_bic": None,
        }

    patch_values = np.asarray(values[y0:y1, x0:x1], dtype=np.float64)
    patch_valid = validity[y0:y1, x0:x1] & np.isfinite(patch_values)
    if int(np.count_nonzero(patch_valid)) < 25:
        return {
            "line_template_snr": None,
            "line_vs_point_delta_bic": None,
            "line_vs_double_delta_bic": None,
            "line_vs_background_delta_bic": None,
            "single_frame_model": "not_fit",
            "single_frame_model_margin_bic": None,
        }

    yy, xx = np.mgrid[y0:y1, x0:x1]
    local_values = patch_values[patch_valid]
    local_x = np.asarray(xx[patch_valid], dtype=np.float64)
    local_y = np.asarray(yy[patch_valid], dtype=np.float64)
    # 所有竞争模型共享同一个局部平面背景。坐标按窗口尺度归一化，
    # 让正规方程的条件数不受 4096 像素绝对坐标影响。
    background_terms = (
        (local_x - float(center_x)) / max(radius, 1.0),
        (local_y - float(center_y)) / max(radius, 1.0),
    )

    point_template = np.exp(
        -0.5
        * (
            (local_x - float(center_x)) ** 2
            + (local_y - float(center_y)) ** 2
        )
        / (sigma * sigma)
    )
    point_peak = float(np.max(point_template))
    if point_peak <= 0.0 or not np.isfinite(point_peak):
        return {
            "line_template_snr": None,
            "line_vs_point_delta_bic": None,
            "line_vs_double_delta_bic": None,
            "line_vs_background_delta_bic": None,
            "single_frame_model": "not_fit",
            "single_frame_model_margin_bic": None,
        }
    point_template /= point_peak

    angle = np.deg2rad(float(angle_deg))
    ux, uy = float(np.cos(angle)), float(np.sin(angle))
    nx, ny = -uy, ux
    half_length = 0.5 * float(length_px)
    # PCA 给出的长度是线形提案，不一定等于双星的真实间距。若只在
    # ``±length/2`` 这一组端点上比较，两个相邻点源会被错误地迫使去
    # 适配线模板。因此双星竞争模型也在同一局部尺度内扫描间距；这
    # 个几何搜索参数在 BIC 中计为一个自由度，避免“多试几组位置”
    # 免费增加双星模型的优势。
    double_separation_candidates = tuple(
        sorted(
            {
                round(max(0.5, float(length_px) * factor), 4)
                for factor in (0.5, 0.75, 1.0, 1.25)
            }
        )
    )

    def double_templates_for_geometry(
        separation_px: float,
        along_offset_px: float = 0.0,
        across_offset_px: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a two-PSF negative control around the PCA geometry.

        PCA centres are weighted by brightness, while a double-star model is
        naturally centred on the midpoint of the two sources.  Keeping the
        midpoint fixed at the weighted centroid therefore makes an obvious
        unequal-brightness pair look artificially more like a finite trail.
        A small, explicitly penalised centre search lets the double model
        follow that midpoint without turning the comparison into an unlimited
        free-form fit.
        """

        half_separation = 0.5 * float(separation_px)
        midpoint_x = float(center_x) + float(along_offset_px) * ux + float(across_offset_px) * nx
        midpoint_y = float(center_y) + float(along_offset_px) * uy + float(across_offset_px) * ny
        endpoint_x = midpoint_x + np.asarray((-half_separation, half_separation)) * ux
        endpoint_y = midpoint_y + np.asarray((-half_separation, half_separation)) * uy
        templates: list[np.ndarray] = []
        for endpoint_x_value, endpoint_y_value in zip(endpoint_x, endpoint_y):
            template = np.exp(
                -0.5
                * (
                    (local_x - float(endpoint_x_value)) ** 2
                    + (local_y - float(endpoint_y_value)) ** 2
                )
                / (sigma * sigma)
            )
            peak = float(np.max(template))
            templates.append(template / peak if peak > 0.0 else template)
        return templates[0], templates[1]

    line_length_candidates = tuple(
        sorted(
            {
                round(max(0.5, float(length_px) * factor), 4)
                for factor in (0.75, 1.0, 1.25)
            }
        )
    )
    line_fit: tuple[float, float, float, float] | None = None
    for candidate_length in line_length_candidates:
        line_template = _finite_line_template(
            local_x.reshape(-1, 1),
            local_y.reshape(-1, 1),
            center_x=center_x,
            center_y=center_y,
            length_px=candidate_length,
            angle_deg=angle_deg,
            psf_sigma=sigma,
        ).reshape(-1)
        rss, snr, _baseline = _fit_single_frame_template(
            local_values,
            (line_template,),
            noise=noise,
            background_terms=background_terms,
        )
        if line_fit is None or rss < line_fit[0]:
            line_fit = (rss, snr, candidate_length, _baseline)
    if line_fit is None:
        return {
            "line_template_snr": None,
            "line_vs_point_delta_bic": None,
            "line_vs_double_delta_bic": None,
            "line_vs_background_delta_bic": None,
            "single_frame_model": "not_fit",
            "single_frame_model_margin_bic": None,
        }

    # 圆形 point 模型过于理想时，失焦、像差或固定姿态抖动会被 line
    # 模型错误吸收。增加一个方向固定、长短轴可在局部网格中变化的静态
    # 椭圆 PSF 负例；它只解释静态拉长星像，不赋予任何时间运动含义。
    ellipse_major_base = max(sigma, float(length_px) / 4.0)
    ellipse_minor_base = max(0.45, min(ellipse_major_base, float(width_px) / 4.0))
    ellipse_major_candidates = tuple(
        sorted({round(max(0.45, ellipse_major_base * factor), 4) for factor in (0.75, 1.0, 1.25)})
    )
    ellipse_minor_candidates = tuple(
        sorted({round(max(0.45, ellipse_minor_base * factor), 4) for factor in (0.75, 1.0, 1.25)})
    )
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    along = (local_x - float(center_x)) * cos_angle + (local_y - float(center_y)) * sin_angle
    across = -(local_x - float(center_x)) * sin_angle + (local_y - float(center_y)) * cos_angle
    elliptical_fit: tuple[float, float, float, float] | None = None
    for major_sigma in ellipse_major_candidates:
        for minor_sigma in ellipse_minor_candidates:
            elliptical_template = np.exp(
                -0.5
                * (
                    (along / float(major_sigma)) ** 2
                    + (across / float(minor_sigma)) ** 2
                )
            )
            peak = float(np.max(elliptical_template))
            if peak <= 0.0 or not np.isfinite(peak):
                continue
            elliptical_template /= peak
            candidate_rss, candidate_snr, _candidate_baseline = _fit_single_frame_template(
                local_values,
                (elliptical_template,),
                noise=noise,
                background_terms=background_terms,
            )
            if elliptical_fit is None or candidate_rss < elliptical_fit[0]:
                elliptical_fit = (candidate_rss, candidate_snr, major_sigma, minor_sigma)

    background_rss, _background_snr, _background_baseline = _fit_single_frame_template(
        local_values,
        (),
        noise=noise,
        background_terms=background_terms,
    )
    point_rss, _point_snr, _point_baseline = _fit_single_frame_template(
        local_values,
        (point_template,),
        noise=noise,
        background_terms=background_terms,
    )
    # 只在短/中等候选上搜索一个很小的中点偏移。这个搜索是为了覆盖
    # “亮星把 PCA 质心拉向自己”的真实双星几何，不是为了给模型无限
    # 自由度；因此沿线/垂线偏移和间距都在固定小网格内，并在 BIC 中
    # 各承担一个搜索自由度。对超长线段不做双星竞争，那里使用整段
    # profile 分支或跨帧证据更合适。
    double_center_offsets = (
        (-2.0, -1.0, 0.0, 1.0, 2.0),
        (-1.0, 0.0, 1.0),
    ) if float(length_px) <= 48.0 else ((0.0,), (0.0,))
    double_fit: tuple[float, float, float, float] | None = None
    for separation_px in double_separation_candidates:
        for along_offset_px in double_center_offsets[0]:
            for across_offset_px in double_center_offsets[1]:
                double_templates = double_templates_for_geometry(
                    separation_px,
                    along_offset_px,
                    across_offset_px,
                )
                candidate_rss, _double_snr, _double_baseline = _fit_single_frame_template(
                    local_values,
                    double_templates,
                    noise=noise,
                    background_terms=background_terms,
                )
                if double_fit is None or candidate_rss < double_fit[0]:
                    double_fit = (
                        candidate_rss,
                        separation_px,
                        along_offset_px,
                        across_offset_px,
                    )
    double_rss = double_fit[0] if double_fit is not None else float("inf")
    elliptical_rss = elliptical_fit[0] if elliptical_fit is not None else float("inf")
    line_rss, line_snr, _line_length, _line_baseline = line_fit
    sample_count = int(local_values.size)
    bics = {
        # 三个背景参数：常数项和两个局部梯度项。
        "background": _single_frame_model_bic(background_rss, sample_count, 3),
        "point": _single_frame_model_bic(point_rss, sample_count, 4),
        # 三个背景参数 + 两个振幅 + 扫描间距 + 中点沿线/垂线偏移。
        # 这三个几何搜索自由度必须承担复杂度惩罚，避免双星模型只因
        # 多试位置就无条件胜出。
        "double": _single_frame_model_bic(double_rss, sample_count, 8),
        # 三个背景参数 + 线模板振幅 + 扫描线长。线长虽然只在局部
        # 三点网格中选择，也必须承担一个搜索自由度的复杂度惩罚。
        "line": _single_frame_model_bic(line_rss, sample_count, 5),
        # 三个背景参数 + 椭圆 PSF 振幅 + 长短轴两个局部搜索参数；方向
        # 和中心沿用当前候选几何，避免把静态负例模型做成无限自由拟合。
        "elliptical_point": _single_frame_model_bic(elliptical_rss, sample_count, 6),
    }
    ranked = sorted(bics.items(), key=lambda item: item[1])
    best_model, best_bic = ranked[0]
    next_bic = ranked[1][1] if len(ranked) > 1 else float("inf")
    return {
        "line_template_snr": float(line_snr),
        "line_vs_point_delta_bic": float(bics["point"] - bics["line"]),
        "line_vs_double_delta_bic": float(bics["double"] - bics["line"]),
        "line_vs_background_delta_bic": float(bics["background"] - bics["line"]),
        "line_vs_elliptical_delta_bic": (
            float(bics["elliptical_point"] - bics["line"])
            if np.isfinite(bics["elliptical_point"])
            else None
        ),
        "single_frame_model": str(best_model),
        "single_frame_model_margin_bic": float(next_bic - best_bic),
    }


def _single_frame_long_line_profile_evidence(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    width_px: float,
    angle_deg: float,
    psf_fwhm: float,
    noise: float,
) -> dict[str, float | str | None]:
    """评估超出局部竞争窗口的整段线形候选。

    ``_single_frame_line_model_evidence`` 为了让大量短候选保持足够快，
    把局部 cutout 限制在 128 px 内。对 F01 那类数百像素的结构，继续
    返回 ``not_fit`` 会把“模型没覆盖”误解成“没有目标”。这里沿候选
    主轴建立一个完整的有限曝光剖面，只比较“局部背景”和“PSF 展宽的
    有限线段”两个模型。它仍然只读取当前残差图，不引入任何帧间信息。

    这个分支故意不伪造点源/双星/椭圆模型的 ΔBIC：那些竞争模型在数百
    像素的整段窗口内不再是同尺度的合理负例。因此结果最多升级为
    ``long_line_profile``，并保留“伪影/跨帧待复核”语义。
    """

    values = np.asarray(residual)
    validity = np.asarray(valid, dtype=bool)
    empty_result = {
        "line_template_snr": None,
        "line_vs_point_delta_bic": None,
        "line_vs_double_delta_bic": None,
        "line_vs_background_delta_bic": None,
        "line_vs_elliptical_delta_bic": None,
        "single_frame_model": "not_fit",
        "single_frame_model_margin_bic": None,
    }
    if (
        values.ndim != 2
        or validity.shape != values.shape
        or not np.isfinite(length_px)
        or length_px <= SINGLE_FRAME_MODEL_MAX_LENGTH_PX
        or psf_fwhm <= 0.0
        or noise <= 0.0
        or not np.isfinite(center_x)
        or not np.isfinite(center_y)
    ):
        return empty_result

    psf_sigma = max(float(psf_fwhm) / 2.354820045, 0.45)
    # 用观测到的横向宽度给 PSF 一个有限的宽化余量，但限制最大值，
    # 防止异常 PCA 宽度把整幅图都当成线模型窗口。
    transverse_sigma = max(psf_sigma, min(8.0, max(0.45, float(width_px) / 4.0)))
    half_length = 0.5 * float(length_px)
    angle = np.deg2rad(float(angle_deg))
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    along_margin = max(4.0, 4.0 * psf_sigma)
    across_radius = max(6.0, 4.0 * transverse_sigma + 2.0)
    x_radius = half_length * abs(cos_angle) + across_radius * abs(sin_angle) + along_margin
    y_radius = half_length * abs(sin_angle) + across_radius * abs(cos_angle) + along_margin
    x0 = max(0, int(np.floor(float(center_x) - x_radius)))
    x1 = min(values.shape[1], int(np.ceil(float(center_x) + x_radius + 1.0)))
    y0 = max(0, int(np.floor(float(center_y) - y_radius)))
    y1 = min(values.shape[0], int(np.ceil(float(center_y) + y_radius + 1.0)))
    if x1 - x0 < 15 or y1 - y0 < 15:
        return empty_result

    patch_values = np.asarray(values[y0:y1, x0:x1], dtype=np.float64)
    patch_valid = validity[y0:y1, x0:x1] & np.isfinite(patch_values)
    if int(np.count_nonzero(patch_valid)) < 100:
        return empty_result
    yy, xx = np.mgrid[y0:y1, x0:x1]
    local_values = patch_values[patch_valid]
    local_x = np.asarray(xx[patch_valid], dtype=np.float64)
    local_y = np.asarray(yy[patch_valid], dtype=np.float64)
    background_terms = (
        (local_x - float(center_x)) / max(x_radius, 1.0),
        (local_y - float(center_y)) / max(y_radius, 1.0),
    )
    along = (local_x - float(center_x)) * cos_angle + (local_y - float(center_y)) * sin_angle
    across = -(local_x - float(center_x)) * sin_angle + (local_y - float(center_y)) * cos_angle
    edge_scale = np.sqrt(2.0) * psf_sigma
    finite_segment = 0.5 * (
        erf((along + half_length) / edge_scale)
        - erf((along - half_length) / edge_scale)
    )
    transverse_profile = np.exp(-0.5 * (across / transverse_sigma) ** 2)
    line_template = finite_segment * transverse_profile
    template_peak = float(np.max(line_template))
    if template_peak <= 0.0 or not np.isfinite(template_peak):
        return empty_result
    line_template /= template_peak

    line_rss, line_snr, _line_baseline = _fit_single_frame_template(
        local_values,
        (line_template,),
        noise=noise,
        background_terms=background_terms,
    )
    background_rss, _background_snr, _background_baseline = _fit_single_frame_template(
        local_values,
        (),
        noise=noise,
        background_terms=background_terms,
    )
    sample_count = int(local_values.size)
    line_bic = _single_frame_model_bic(line_rss, sample_count, 5)
    background_bic = _single_frame_model_bic(background_rss, sample_count, 3)
    delta_bic = float(background_bic - line_bic)
    if not np.isfinite(line_snr) or not np.isfinite(delta_bic):
        return empty_result
    supported = (
        float(line_snr) >= SINGLE_FRAME_LONG_PROFILE_MIN_SNR
        and delta_bic >= SINGLE_FRAME_LONG_PROFILE_MIN_DELTA_BIC
    )
    return {
        "line_template_snr": float(line_snr),
        "line_vs_point_delta_bic": None,
        "line_vs_double_delta_bic": None,
        "line_vs_background_delta_bic": delta_bic,
        "line_vs_elliptical_delta_bic": None,
        "single_frame_model": "long_line_profile" if supported else "background",
        "single_frame_model_margin_bic": delta_bic if supported else None,
    }


def _apply_single_frame_model_evidence(
    point: MotionFeaturePoint,
    *,
    residual: np.ndarray,
    valid: np.ndarray,
    psf_fwhm: float,
    noise: float,
) -> MotionFeaturePoint:
    """Attach a model result obtained from the *raw current-frame* pixels.

    A temporal-difference image is useful for finding changes, but it is a
    poor negative control for a close pair: a small registration residual can
    make two compact sources look like a short line.  Before a temporal line
    proposal is associated across frames, re-score the same geometry on the
    current raw residual and record the competing-model winner.  This keeps
    the sequence detector from treating ``point/double/ellipse`` structures
    as line evidence while preserving the original temporal residual metrics.
    """

    if point.length_px > SINGLE_FRAME_MODEL_MAX_LENGTH_PX:
        evidence = _single_frame_long_line_profile_evidence(
            residual,
            valid,
            center_x=point.x,
            center_y=point.y,
            length_px=point.length_px,
            width_px=point.width_px,
            angle_deg=point.angle_deg,
            psf_fwhm=psf_fwhm,
            noise=noise,
        )
    else:
        evidence = _single_frame_line_model_evidence(
            residual,
            valid,
            center_x=point.x,
            center_y=point.y,
            length_px=point.length_px,
            width_px=point.width_px,
            angle_deg=point.angle_deg,
            psf_fwhm=psf_fwhm,
            noise=noise,
        )
    return replace(
        point,
        line_template_snr=evidence.get("line_template_snr"),
        line_vs_point_delta_bic=evidence.get("line_vs_point_delta_bic"),
        line_vs_double_delta_bic=evidence.get("line_vs_double_delta_bic"),
        line_vs_background_delta_bic=evidence.get("line_vs_background_delta_bic"),
        line_vs_elliptical_delta_bic=evidence.get("line_vs_elliptical_delta_bic"),
        single_frame_model=evidence.get("single_frame_model"),
        single_frame_model_margin_bic=evidence.get("single_frame_model_margin_bic"),
    )


def _single_frame_model_is_supported(point: MotionFeaturePoint) -> bool:
    """Return whether the local finite-line template supports a strong line."""

    if point.single_frame_model == "long_line_profile":
        return (
            point.line_template_snr is not None
            and float(point.line_template_snr) >= SINGLE_FRAME_LONG_PROFILE_MIN_SNR
            and point.line_vs_background_delta_bic is not None
            and float(point.line_vs_background_delta_bic) >= SINGLE_FRAME_LONG_PROFILE_MIN_DELTA_BIC
        )
    return (
        point.single_frame_model == "line"
        and point.line_template_snr is not None
        and float(point.line_template_snr) >= SINGLE_FRAME_MODEL_MIN_SNR
        and point.line_vs_point_delta_bic is not None
        and float(point.line_vs_point_delta_bic) >= SINGLE_FRAME_MODEL_MIN_DELTA_BIC
        and point.line_vs_double_delta_bic is not None
        and float(point.line_vs_double_delta_bic) >= SINGLE_FRAME_MODEL_MIN_DELTA_BIC
        and point.line_vs_elliptical_delta_bic is not None
        and float(point.line_vs_elliptical_delta_bic) >= SINGLE_FRAME_MODEL_MIN_DELTA_BIC
    )


def _single_frame_model_is_review_candidate(point: MotionFeaturePoint) -> bool:
    """Return whether a line model won locally but still needs review.

    The strong display gate below asks for a substantial BIC margin against
    every compact-source control.  That is appropriate for a solid line, but
    it must not erase a useful high-recall proposal from the review layer.  A
    review candidate is therefore allowed when the *line* model is the local
    winner, its template SNR is usable, and every recorded BIC comparison is
    at least the weak-evidence boundary.  A near tie below that boundary is
    kept only in the audit table: drawing it as ``EXP TRAIL?`` makes a compact
    bright star look like a streak.  It is deliberately not a fallback for
    missing evidence or for a point/double/elliptical winner; the GUI draws
    this result dashed and never treats it as a confirmed moving object.
    """

    if point.single_frame_model != "line":
        return False
    evidence = (
        point.line_template_snr,
        point.line_vs_point_delta_bic,
        point.line_vs_double_delta_bic,
        point.line_vs_background_delta_bic,
        point.line_vs_elliptical_delta_bic,
        point.single_frame_model_margin_bic,
    )
    if any(value is None or not np.isfinite(float(value)) for value in evidence):
        return False
    return (
        float(point.line_template_snr) >= SINGLE_FRAME_MODEL_MIN_SNR
        and float(point.line_vs_point_delta_bic) >= SINGLE_FRAME_REVIEW_MIN_DELTA_BIC
        and float(point.line_vs_double_delta_bic) >= SINGLE_FRAME_REVIEW_MIN_DELTA_BIC
        and float(point.line_vs_background_delta_bic) >= SINGLE_FRAME_REVIEW_MIN_DELTA_BIC
        and float(point.line_vs_elliptical_delta_bic) >= SINGLE_FRAME_REVIEW_MIN_DELTA_BIC
        and float(point.single_frame_model_margin_bic) >= SINGLE_FRAME_REVIEW_MIN_DELTA_BIC
    )


def _single_frame_sequence_line_model_is_supported(
    point: MotionFeaturePoint,
    *,
    psf_fwhm: float,
) -> bool:
    """Return whether a sequence line has a raw-frame line-like control.

    The normal gate requires the finite-line template to beat every compact
    negative model.  A detector can also produce a genuinely one-pixel-wide
    synthetic/quantised streak whose raw winner is an elongated ellipse only
    because the image contains no PSF-width cross-section.  Keep that narrow
    case available to the *sequence* route when continuity, fill, SNR and the
    double-star comparison are all strong.  It still needs cross-frame motion
    evidence before it can become ``moving``; a compact double/point winner
    never receives this exception.
    """

    if _single_frame_model_is_supported(point):
        return True
    if point.single_frame_model != "elliptical_point":
        return False
    if (
        point.line_template_snr is None
        or point.line_vs_point_delta_bic is None
        or point.line_vs_double_delta_bic is None
        or point.line_vs_background_delta_bic is None
        or point.continuity_ratio is None
        or point.profile_fill_ratio is None
    ):
        return False
    thin_width_limit = max(0.75, 0.5 * float(psf_fwhm))
    return (
        float(point.width_px) <= thin_width_limit
        and float(point.length_px) >= max(8.0, 3.0 * float(psf_fwhm))
        and float(point.line_template_snr) >= SINGLE_FRAME_MODEL_MIN_SNR
        and float(point.line_vs_point_delta_bic) >= SINGLE_FRAME_MODEL_MIN_DELTA_BIC
        and float(point.line_vs_double_delta_bic) >= SINGLE_FRAME_MODEL_MIN_DELTA_BIC
        and float(point.line_vs_background_delta_bic) >= SINGLE_FRAME_MODEL_MIN_DELTA_BIC
        and float(point.continuity_ratio) >= 0.80
        and float(point.profile_fill_ratio) >= 0.80
    )


def _single_frame_sequence_line_model_is_review_supported(
    point: MotionFeaturePoint,
    *,
    psf_fwhm: float,
) -> bool:
    """Return whether a raw line *review* point may enter sequence linking.

    This is intentionally weaker than ``_single_frame_sequence_line_model_is_supported``.
    A point is admitted only when the finite-line model is still the local
    winner (or a very thin line is represented by the quantised elliptical
    exception), the line has usable SNR/background evidence, and its raw
    profile is not completely disconnected.  Compact ``point``/``double``
    winners never pass.  The function is a proposal gate, not a moving gate:
    cross-frame motion and track-level shape checks remain mandatory.
    """

    if psf_fwhm <= 0.0:
        raise ValueError("psf_fwhm must be positive")
    if _single_frame_sequence_line_model_is_supported(point, psf_fwhm=psf_fwhm):
        return True
    if point.single_frame_model not in {"line", "long_line_profile", "elliptical_point"}:
        return False
    required = (
        point.line_template_snr,
        point.line_vs_background_delta_bic,
        point.continuity_ratio,
        point.profile_fill_ratio,
    )
    if any(value is None or not np.isfinite(float(value)) for value in required):
        return False
    length_floor = max(
        SEQUENCE_REVIEW_MIN_LENGTH_PX,
        2.5 * float(psf_fwhm),
    )
    if float(point.length_px) < length_floor:
        return False
    if (
        float(point.line_template_snr) < SEQUENCE_REVIEW_MIN_MODEL_SNR
        or float(point.line_vs_background_delta_bic) < SINGLE_FRAME_REVIEW_MIN_DELTA_BIC
        or float(point.continuity_ratio) < SEQUENCE_REVIEW_MIN_SHAPE_RATIO
        or float(point.profile_fill_ratio) < SEQUENCE_REVIEW_MIN_SHAPE_RATIO
    ):
        return False
    if point.single_frame_model in {"line", "elliptical_point"}:
        compact_deltas = (
            point.line_vs_point_delta_bic,
            point.line_vs_double_delta_bic,
            point.line_vs_elliptical_delta_bic,
        )
        if any(value is None or not np.isfinite(float(value)) for value in compact_deltas):
            return False
        if any(float(value) < SINGLE_FRAME_REVIEW_MIN_DELTA_BIC for value in compact_deltas):
            return False
    if point.single_frame_model == "elliptical_point":
        # Only retain the narrow, quantised-streak exception.  A normal
        # elongated static PSF must stay a compact-source negative control.
        if float(point.width_px) > max(0.75, 0.5 * float(psf_fwhm)):
            return False
    return True


def _motion_feature_track_model_is_supported(
    points: Sequence[MotionFeaturePoint],
    *,
    psf_fwhm: float,
    min_pass_fraction: float = 0.5,
) -> bool:
    """Require raw line-model review evidence on a majority of a track.

    A review point can seed linking, but a track made only from geometric
    fragments must not be promoted solely because its centroids happen to fit
    a line.  This aggregate gate is deliberately less strict than the
    per-frame strong-model gate so one degraded exposure does not erase an
    otherwise coherent target.
    """

    if len(points) < 3 or psf_fwhm <= 0.0:
        return False
    if not 0.0 < float(min_pass_fraction) <= 1.0:
        raise ValueError("min_pass_fraction must be in (0, 1]")
    passed = sum(
        _single_frame_sequence_line_model_is_review_supported(point, psf_fwhm=psf_fwhm)
        for point in points
    )
    return passed >= max(2, int(np.ceil(len(points) * float(min_pass_fraction))))


def _line_artifact_mask_from_coordinates(
    coordinates: np.ndarray | Sequence[tuple[int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Expand sparse ``LINE_ARTIFACT`` coordinates into a local boolean mask."""

    mask = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    raw = np.asarray(coordinates)
    if raw.size == 0:
        return mask
    if raw.ndim != 2 or raw.shape[1] != 2:
        return mask
    raw = np.asarray(raw, dtype=np.int64)
    inside = (
        (raw[:, 0] >= 0)
        & (raw[:, 0] < mask.shape[1])
        & (raw[:, 1] >= 0)
        & (raw[:, 1] < mask.shape[0])
    )
    if np.any(inside):
        mask[raw[inside, 1], raw[inside, 0]] = True
    return mask


def _line_continuity_ratio(
    residual: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    width_px: float,
    angle_deg: float,
    threshold_adu: float,
) -> float | None:
    """兼容旧调用，返回候选主轴上的绝对连续支撑度。"""

    continuity_ratio, _profile_fill_ratio = _line_profile_metrics(
        residual,
        center_x=center_x,
        center_y=center_y,
        length_px=length_px,
        width_px=width_px,
        angle_deg=angle_deg,
        threshold_adu=threshold_adu,
    )
    return continuity_ratio


def _motion_feature_shape_is_supported(
    point: MotionFeaturePoint,
    *,
    min_profile_fill_ratio: float = MOTION_FEATURE_MIN_PROFILE_FILL_RATIO,
    min_continuity_ratio: float = MOTION_FEATURE_MIN_CONTINUITY_RATIO,
) -> bool:
    """判断线状候选是否有足够的连续亮度形态证据。

    ``None`` 代表旧缓存或外部手工对象没有剖面证据；为避免旧结果被
    静默升级成“强线”，这种情况下返回 False。长线也必须通过剖面门，
    因为一条超长的断裂/边缘伪迹不应只靠 PCA 长度获得通过。
    """

    if point.profile_fill_ratio is None or point.continuity_ratio is None:
        return False
    return (
        float(point.profile_fill_ratio) >= float(min_profile_fill_ratio)
        and float(point.continuity_ratio) >= float(min_continuity_ratio)
    )


def _single_frame_trail_shape_is_supported(
    point: MotionFeaturePoint,
    *,
    min_length_px: float = SINGLE_FRAME_SHORT_TRAIL_MIN_LENGTH,
    min_profile_fill_ratio: float = SINGLE_FRAME_SHORT_TRAIL_MIN_PROFILE_FILL_RATIO,
    min_continuity_ratio: float = SINGLE_FRAME_SHORT_TRAIL_MIN_CONTINUITY_RATIO,
) -> bool:
    """判断单帧默认叠加是否应显示这条曝光内线形。

    单帧没有时间维度，不能把线形直接称为 moving；但把最小长度固定为
    16 px 又会漏掉当前 1.5 s 数据里肉眼可见的短拖影。因此单帧采用两段
    口径：较长候选沿用通用形态门，8--15 px 的短候选必须同时通过更严格
    的连续度和中段填充门。这样降低的是“漏检门槛”，不是把相邻双星的
    低填充连线直接升级为结果。
    """

    return _single_frame_trail_shape_reason(
        point,
        min_length_px=min_length_px,
        min_profile_fill_ratio=min_profile_fill_ratio,
        min_continuity_ratio=min_continuity_ratio,
    ) is None


def _sequence_line_shape_is_supported(point: MotionFeaturePoint) -> bool:
    """Return whether a line observation can seed sequence association.

    The single-frame display deliberately downgrades a line that overlaps the
    detector's generic ``LINE_ARTIFACT`` mask.  That is the right display
    warning, but it is too early to delete a moving object: the mask is made
    from the current image and can cover a real bright trail itself.  The
    sequence proposal layer therefore permits a *model-supported* multiscale
    line through this admission gate, while retaining the overlap fraction for
    the final audit.  A cross-frame track still has to pass displacement,
    temporal fit and multi-frame shape checks before it can be ``moving``.

    Points without local line-model evidence continue to use the stricter
    single-frame gate, so this exception cannot turn old raw artifact pixels
    into motion evidence by itself.
    """

    if _single_frame_trail_shape_is_supported(point):
        return True
    if point.single_frame_model not in {"line", "long_line_profile"}:
        return False
    if not _single_frame_model_is_supported(point):
        return False
    if float(point.length_px) < SINGLE_FRAME_SHORT_TRAIL_MIN_LENGTH:
        return False
    return _motion_feature_shape_is_supported(point)


def _single_frame_trail_shape_reason(
    point: MotionFeaturePoint,
    *,
    min_length_px: float = SINGLE_FRAME_SHORT_TRAIL_MIN_LENGTH,
    min_profile_fill_ratio: float = SINGLE_FRAME_SHORT_TRAIL_MIN_PROFILE_FILL_RATIO,
    min_continuity_ratio: float = SINGLE_FRAME_SHORT_TRAIL_MIN_CONTINUITY_RATIO,
) -> str | None:
    """返回单帧线状候选未通过显示形态门的第一个原因。

    线状候选的发现和强证据显示是两层不同的任务。这个函数让 GUI 可以
    把“发现了但证据较弱”的结构保留为复核线，而不是把它静默变成“没有
    目标”；返回 ``None`` 表示通过当前单帧强证据门。
    """

    if min_length_px <= 0:
        raise ValueError("single-frame short-trail length gate must be positive")
    length_px = float(point.length_px)
    if length_px < float(min_length_px):
        return f"长度不足（{length_px:.1f} < {float(min_length_px):.1f}px）"
    if point.continuity_ratio is None:
        return "缺少连续支撑证据"
    if point.profile_fill_ratio is None:
        return "缺少中段填充证据"
    if (
        point.line_artifact_overlap_fraction is not None
        and float(point.line_artifact_overlap_fraction) >= SINGLE_FRAME_ARTIFACT_REVIEW_OVERLAP_FRACTION
    ):
        return (
            "线状伪影候选重叠过高，保留待复核"
            f"（{float(point.line_artifact_overlap_fraction):.0%}）"
        )
    if point.single_frame_model == "not_fit":
        return "超出局部有限线段模型范围，保留几何候选"
    if _single_frame_model_is_supported(point):
        # 局部有限线段模板是第二层证据，不是对像素形态的豁免。它只能
        # 救回几何门因阈值断裂的短拖影；如果中段/连续像素本身太稀疏，
        # 仍保留为待复核，避免把噪声中的几个亮点直接画成实线。
        rescue_continuity = float(SINGLE_FRAME_MODEL_RESCUE_MIN_CONTINUITY_RATIO)
        rescue_fill = float(SINGLE_FRAME_MODEL_RESCUE_MIN_PROFILE_FILL_RATIO)
        if length_px >= 16.0:
            rescue_continuity = max(rescue_continuity, MOTION_FEATURE_MIN_CONTINUITY_RATIO)
            rescue_fill = max(rescue_fill, MOTION_FEATURE_MIN_PROFILE_FILL_RATIO)
        if (
            float(point.continuity_ratio) >= rescue_continuity
            and float(point.profile_fill_ratio) >= rescue_fill
        ):
            return None
        return (
            "有限线段模型较优，但像素连续支撑不足"
            f"（连续 {float(point.continuity_ratio):.0%} / 填充 {float(point.profile_fill_ratio):.0%}）"
        )
    if point.single_frame_model == "elliptical_point":
        return "局部模型更像静态椭圆星像"
    if point.single_frame_model == "point":
        return "局部模型更像单星"
    if point.single_frame_model == "double":
        return "局部模型更像双星"
    if point.single_frame_model == "background":
        return "局部模型更像背景起伏"
    if point.single_frame_model == "long_line_profile":
        return "超长线段剖面证据不足，保留待复核"
    if length_px >= 16.0:
        required_continuity = MOTION_FEATURE_MIN_CONTINUITY_RATIO
        required_fill = MOTION_FEATURE_MIN_PROFILE_FILL_RATIO
    else:
        required_continuity = float(min_continuity_ratio)
        required_fill = float(min_profile_fill_ratio)
    if float(point.continuity_ratio) < required_continuity:
        return f"连续支撑不足（{float(point.continuity_ratio):.0%} < {required_continuity:.0%}）"
    if float(point.profile_fill_ratio) < required_fill:
        return f"中段填充不足（{float(point.profile_fill_ratio):.0%} < {required_fill:.0%}）"
    return None


def _motion_feature_track_shape_is_supported(
    points: Sequence[MotionFeaturePoint],
    *,
    min_profile_fill_ratio: float = MOTION_FEATURE_MIN_PROFILE_FILL_RATIO,
    min_continuity_ratio: float = MOTION_FEATURE_MIN_CONTINUITY_RATIO,
    max_angle_deviation_deg: float = MOTION_FEATURE_MAX_ANGLE_DEVIATION_DEG,
    max_length_deviation_fraction: float = MOTION_FEATURE_MAX_LENGTH_DEVIATION_FRACTION,
) -> bool:
    """判断一条跨帧线是否有足够的逐帧连续形态证据。

    线状目标可能在个别帧被局部背景、饱和或帧边界截断，因此不要求
    每一帧都完美；但至少半数观测点（且不少于两个）必须同时通过连续
    支撑和剖面填充门，所有可用帧的中位填充度也必须达到门槛。把这条
    规则集中在一个函数里，避免“轨迹分类”和“界面绘制”出现不同口径。
    """

    if len(points) < 3:
        return False
    supported_points = []
    for point in points:
        point_min_profile_fill = float(min_profile_fill_ratio)
        point_min_continuity = float(min_continuity_ratio)
        if float(point.length_px) < MOTION_FEATURE_SHORT_TRAIL_LENGTH:
            # 短线更容易由两个相邻 PSF 拼成；它必须在当前帧有更完整的
            # 中段和连续支撑，并且还要在多帧上保持一致，才能进入 moving
            # 的形态证据集合。
            point_min_profile_fill = max(
                point_min_profile_fill,
                MOTION_FEATURE_SHORT_TRAIL_MIN_PROFILE_FILL_RATIO,
            )
            point_min_continuity = max(
                point_min_continuity,
                MOTION_FEATURE_SHORT_TRAIL_MIN_CONTINUITY_RATIO,
            )
        if _motion_feature_shape_is_supported(
            point,
            min_profile_fill_ratio=point_min_profile_fill,
            min_continuity_ratio=point_min_continuity,
        ):
            supported_points.append(point)
    required_count = max(2, int(np.ceil(len(points) * 0.5)))
    if len(supported_points) < required_count:
        return False

    profile_values = [
        float(point.profile_fill_ratio)
        for point in supported_points
        if point.profile_fill_ratio is not None
    ]
    if not profile_values or float(np.median(profile_values)) < float(min_profile_fill_ratio):
        return False

    # 线方向是无向量，使用 2*theta 后再做圆均值，避免 179° 和 1° 被
    # 错误地认为相差 178°。只用通过形态门的帧，降低边界截断帧的影响。
    angles = np.asarray(
        [float(point.angle_deg) for point in supported_points if np.isfinite(point.angle_deg)],
        dtype=np.float64,
    )
    if len(angles) >= 2:
        doubled = np.deg2rad(2.0 * angles)
        mean_doubled = np.arctan2(
            float(np.mean(np.sin(doubled))),
            float(np.mean(np.cos(doubled))),
        )
        angle_delta = 0.5 * np.arctan2(
            np.sin(doubled - mean_doubled),
            np.cos(doubled - mean_doubled),
        )
        max_deviation = float(np.rad2deg(np.max(np.abs(angle_delta))))
        if max_deviation > float(max_angle_deviation_deg):
            return False

    # 长度一致性要看整条轨迹，而不是只看形态通过的子集；否则两帧短线
    # 被形态门淘汰后，剩余长线恰好可以自洽，反而会绕过长度跳变检查。
    lengths = np.asarray(
        [float(point.length_px) for point in points if np.isfinite(point.length_px)],
        dtype=np.float64,
    )
    if len(lengths) >= 2:
        median_length = float(np.median(lengths))
        if median_length <= 0.0:
            return False
        relative_deviation = np.abs(lengths - median_length) / median_length
        if int(np.count_nonzero(relative_deviation <= float(max_length_deviation_fraction))) < required_count:
            return False
    return True


def _motion_feature_motion_axis_is_supported(
    points: Sequence[MotionFeaturePoint],
    *,
    max_deviation_deg: float = MOTION_FEATURE_MAX_MOTION_AXIS_DEVIATION_DEG,
    min_pass_fraction: float = MOTION_FEATURE_MIN_MOTION_AXIS_PASS_FRACTION,
) -> bool:
    """判断线段轴是否与轨迹实际帧间位移方向相容。

    ``angle_deg`` 表示无方向线轴，不能直接与有正负号的速度角相减，
    所以用 ``_angle_difference_mod_180`` 比较。这里是最终确认层的多数
    门，不是候选发现门：少数帧的质心抖动、端点阈值变化和边缘截断不应
    让整条轨迹消失；但若大多数过渡都近似垂直，继续命名为 moving 会
    把“线状形状”和“运动证据”混为一谈。
    """

    if len(points) < 3:
        return False
    if not np.isfinite(float(max_deviation_deg)) or float(max_deviation_deg) <= 0.0:
        raise ValueError("max_deviation_deg must be positive")
    if not 0.0 < float(min_pass_fraction) <= 1.0:
        raise ValueError("min_pass_fraction must be in (0, 1]")
    ordered = tuple(sorted(points, key=lambda point: int(point.frame_index)))
    pass_count = 0
    transition_count = 0
    minimum_step = 1.0
    for previous, current in zip(ordered, ordered[1:]):
        delta_x = float(current.aligned_x) - float(previous.aligned_x)
        delta_y = float(current.aligned_y) - float(previous.aligned_y)
        step = float(np.hypot(delta_x, delta_y))
        if not np.isfinite(step) or step < minimum_step:
            continue
        transition_count += 1
        if not np.isfinite(float(current.angle_deg)):
            continue
        motion_angle = float(np.degrees(np.arctan2(delta_y, delta_x)))
        if (
            _angle_difference_mod_180(float(current.angle_deg), motion_angle)
            <= float(max_deviation_deg)
        ):
            pass_count += 1
    if transition_count == 0:
        return False
    required = max(
        2,
        int(np.ceil(float(transition_count) * float(min_pass_fraction))),
    )
    return pass_count >= required


def _motion_feature_artifact_is_supported_as_moving(
    points: Sequence[MotionFeaturePoint],
    *,
    overlap_fraction: float = MOTION_FEATURE_ARTIFACT_OVERLAP_FRACTION,
    min_clear_fraction: float = MOTION_FEATURE_MIN_ARTIFACT_CLEAR_FRACTION,
) -> bool:
    """Reject a moving label when most known observations are fixed artifacts.

    ``line_artifact_overlap_fraction is None`` means that the artifact audit
    was unavailable, not that the point is clean. Unknown observations are
    therefore excluded from the denominator; only an explicit high-overlap
    majority can veto the stronger ``moving`` label. The point and track are
    still returned as a candidate for review.
    """

    if not points:
        return False
    if not np.isfinite(float(overlap_fraction)) or not 0.0 <= float(overlap_fraction) <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")
    if not 0.0 < float(min_clear_fraction) <= 1.0:
        raise ValueError("min_clear_fraction must be in (0, 1]")
    known = [
        float(point.line_artifact_overlap_fraction)
        for point in points
        if point.line_artifact_overlap_fraction is not None
        and np.isfinite(float(point.line_artifact_overlap_fraction))
    ]
    if not known:
        return True
    clear_fraction = float(np.mean(np.asarray(known) < float(overlap_fraction)))
    return clear_fraction >= float(min_clear_fraction)


def _motion_feature_artifact_overlap_is_compatible_with_motion(
    points: Sequence[MotionFeaturePoint],
    *,
    displacement_px: float,
    fit_rms_px: float | None,
    max_fit_rms_px: float,
    presence_supported: bool,
    min_displacement_px: float,
    shape_supported: bool,
    motion_axis_supported: bool,
    temporal_continuity_supported: bool,
    exposure_supported: bool,
    model_supported: bool = True,
) -> bool:
    """判断高伪影重叠是否可能只是“真实亮线被通用直线掩膜复用”。

    ``_linear_artifact_mask`` 是候选发现层的通用直线提示，并不掌握这条
    线是否来自固定仪器。真实曝光拖影自身也会完整落入该掩膜，因此“重叠
    100%”不能单独作为跨帧目标的否决条件。这里给出的覆盖条件刻意很窄：
    轨迹必须有足够帧数、明显位移、低拟合残差、逐帧线形、运动轴一致、
    连续观测和曝光内速度一致。缺任一项都继续保留伪影否决。

    这不是把伪影当成目标，而是把“通用直线检测器命中了同一条真实拖影”
    与“固定伪影被误关联成轨迹”分开。返回值只用于最终 moving 门；候选
    层和审计层仍保留原始重叠比例。
    """

    if _motion_feature_artifact_is_supported_as_moving(points):
        return True
    if not presence_supported:
        return False
    if not np.isfinite(float(displacement_px)) or float(displacement_px) < float(min_displacement_px):
        return False
    if fit_rms_px is None or not np.isfinite(float(fit_rms_px)):
        return False
    if not np.isfinite(float(max_fit_rms_px)) or float(fit_rms_px) > float(max_fit_rms_px):
        return False
    return bool(
        shape_supported
        and model_supported
        and motion_axis_supported
        and temporal_continuity_supported
        and exposure_supported
    )


def _motion_feature_edge_truncation_is_supported(
    points: Sequence[MotionFeaturePoint],
    frame_shapes: Sequence[tuple[int, int]],
    *,
    edge_margin_px: float = MOTION_FEATURE_EDGE_TRUNCATION_MARGIN_PX,
) -> bool:
    """判断线状轨迹是否是从画面首/末边界进出的连续观测。

    这不是“少一帧就算通过”的通用后门。只有当观测帧本身连续、整段
    贴在序列开头或结尾，并且“进入/离开观测块”的端点包围盒距离图像
    边界不超过少量像素时，才把它解释为边界截断。注意：F09--F15 这种
    从画面底边进入、随后一直观测到序列末尾的轨迹，贴边的是 F09，
    不是已经移动到画面内部的 F15。这样它才可以接受一帧信息缺失；
    中间有缺口、只出现中间几帧或只是普通短线的结果仍然不能使用这个
    例外。

    ``frame_shapes`` 使用原图的 ``(height, width)``，而 ``bbox`` 是原图
    坐标。返回值只是证据标签，不能单独升级候选，调用方还要继续检查
    位移、拟合、形态和曝光速度一致性。
    """

    if len(points) < 3 or len(frame_shapes) < 3:
        return False
    try:
        indices = sorted(int(point.frame_index) for point in points)
        margin = float(edge_margin_px)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(margin) or margin < 0.0:
        return False
    if len(set(indices)) != len(indices):
        return False
    if any(index < 0 or index >= len(frame_shapes) for index in indices):
        return False
    if indices[-1] - indices[0] + 1 != len(indices):
        return False
    touches_sequence_boundary = indices[0] == 0 or indices[-1] == len(frame_shapes) - 1
    if not touches_sequence_boundary:
        return False

    points_by_index = {
        int(point.frame_index): point
        for point in points
    }
    # A suffix run enters the field at its first observed frame and then
    # reaches the end of the sequence.  A prefix run is the mirror image:
    # it starts at frame 0 and leaves the field at its last observed frame.
    # Checking only frame 0 / frame N-1 was wrong for F09--F15: F09 touches
    # the detector edge, while F15 is already well inside the image.
    edge_endpoint_indices: set[int] = set()
    if indices[-1] == len(frame_shapes) - 1:
        edge_endpoint_indices.add(indices[0])
    if indices[0] == 0:
        edge_endpoint_indices.add(indices[-1])
    for index in edge_endpoint_indices:
        point = points_by_index[index]
        try:
            height, width = (int(value) for value in frame_shapes[index])
            x_min, y_min, x_max, y_max = (float(value) for value in point.bbox)
        except (TypeError, ValueError):
            continue
        if height <= 0 or width <= 0:
            continue
        edge_distance = min(
            x_min,
            y_min,
            float(width - 1) - x_max,
            float(height - 1) - y_max,
        )
        if np.isfinite(edge_distance) and edge_distance <= margin:
            return True
    return False


@dataclass(frozen=True, slots=True)
class MotionFeatureTrack:
    """线状运动证据。

    ``classification == "candidate"``、只有一个观测点且
    ``evidence_level == "single_frame_trail"`` 时，表示“单帧曝光线形
    候选”：线形已经提示曝光期间可能发生了位移，但还没有跨帧证据把它
    提升为 ``moving``。15 帧关联产生的单点候选不属于这个类别。
    """

    track_id: int
    classification: str
    points: tuple[MotionFeaturePoint, ...]
    displacement_px: float
    speed_px_per_frame: float
    fit_rms_px: float | None
    # ``single_frame_trail`` 只表示当前曝光内的线形证据，不含帧间量；
    # ``center_motion`` 使用线段质心的常速度拟合；``collinear_trail``
    # 使用跨帧线段中心的空间共线拟合。两者的 RMS 不是同一个模型，
    # 显式保存证据类型，避免 UI/报告把几何残差误读成速度拟合误差。
    evidence_level: str = "center_motion"
    geometry_rms_px: float | None = None
    # ``exposure_velocity_ratio`` 是经 PSF 二阶矩修正的线内速度与
    # 跨帧质心速度之比；``exposure_velocity_log_mad`` 描述各相邻帧比值
    # 的稳健离散程度；``exposure_consistency_fraction`` 是落在宽松物理
    # 比值门内的相邻转移比例。三者都只在头部提供明确曝光和时间时有值。
    exposure_velocity_ratio: float | None = None
    exposure_velocity_log_mad: float | None = None
    exposure_consistency_fraction: float | None = None
    # Stable diagnostic codes for candidate tracks.  These explain why a
    # geometry candidate was not promoted to ``moving``; they do not change
    # the classification and are empty for confirmed moving tracks.
    rejection_reasons: tuple[str, ...] = ()

    @property
    def presence(self) -> int:
        return len(self.points)

    @property
    def observed_frame_indices(self) -> tuple[int, ...]:
        """Return the distinct observed frame indices in chronological order."""

        return tuple(sorted({int(point.frame_index) for point in self.points}))

    @property
    def observed_frame_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return contiguous observed-frame runs without bridging missing frames."""

        indices = self.observed_frame_indices
        if not indices:
            return ()
        ranges: list[tuple[int, int]] = []
        start = previous = indices[0]
        for index in indices[1:]:
            if index != previous + 1:
                ranges.append((start, previous))
                start = index
            previous = index
        ranges.append((start, previous))
        return tuple(ranges)

    @property
    def is_single_frame_motion_candidate(self) -> bool:
        """Whether this track came from the independent single-frame route.

        A one-point track produced by the 15-frame association stage is still
        a sequence candidate, not a single-exposure candidate. Presence alone
        therefore cannot determine this label; the evidence type is part of
        the contract.
        """

        return (
            self.classification == "candidate"
            and self.presence == 1
            and self.evidence_level == "single_frame_trail"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "classification": self.classification,
            "presence": self.presence,
            "observed_frame_indices": list(self.observed_frame_indices),
            "observed_frame_ranges": [list(frame_range) for frame_range in self.observed_frame_ranges],
            "displacement_px": self.displacement_px,
            "speed_px_per_frame": self.speed_px_per_frame,
            "fit_rms_px": self.fit_rms_px,
            "evidence_level": self.evidence_level,
            "geometry_rms_px": self.geometry_rms_px,
            "exposure_velocity_ratio": self.exposure_velocity_ratio,
            "exposure_velocity_log_mad": self.exposure_velocity_log_mad,
            "exposure_consistency_fraction": self.exposure_consistency_fraction,
            "rejection_reasons": list(self.rejection_reasons),
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class MotionEvidenceLink:
    """点状轨迹与线状轨迹对同一运动源的跨通道关联。"""

    point_track_id: int
    line_track_id: int
    overlap_frame_count: int
    overlap_fraction: float
    median_perpendicular_distance_px: float
    direction_difference_deg: float

    def as_dict(self) -> dict[str, object]:
        return {
            "point_track_id": self.point_track_id,
            "line_track_id": self.line_track_id,
            "overlap_frame_count": self.overlap_frame_count,
            "overlap_fraction": self.overlap_fraction,
            "median_perpendicular_distance_px": self.median_perpendicular_distance_px,
            "direction_difference_deg": self.direction_difference_deg,
        }


@dataclass(frozen=True, slots=True)
class MotionFrameAudit:
    """逐帧线状筛选的中间计数，便于解释候选为何被保留或剔除。"""

    frame_index: int
    threshold_adu: float
    residual_noise_adu: float
    valid_pixel_count: int
    support_pixel_count: int
    component_count: int
    area_pass_count: int
    geometry_pass_count: int
    edge_rejected_count: int
    feature_count: int
    max_feature_residual_snr: float | None
    # 首个几何淘汰原因。它们只解释“连通域没有进入线形候选”的阶段，
    # 不把后续原图模型、跨帧关联和 moving 分类混在这里。
    short_area_rejected_count: int = 0
    length_rejected_count: int = 0
    axis_ratio_rejected_count: int = 0
    large_area_rejected_count: int = 0
    short_min_pixels: int = MOTION_FEATURE_SHORT_TRAIL_MIN_AREA

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "threshold_adu": self.threshold_adu,
            "residual_noise_adu": self.residual_noise_adu,
            "valid_pixel_count": self.valid_pixel_count,
            "support_pixel_count": self.support_pixel_count,
            "component_count": self.component_count,
            "area_pass_count": self.area_pass_count,
            "geometry_pass_count": self.geometry_pass_count,
            "edge_rejected_count": self.edge_rejected_count,
            "feature_count": self.feature_count,
            "max_feature_residual_snr": self.max_feature_residual_snr,
            "short_area_rejected_count": self.short_area_rejected_count,
            "length_rejected_count": self.length_rejected_count,
            "axis_ratio_rejected_count": self.axis_ratio_rejected_count,
            "large_area_rejected_count": self.large_area_rejected_count,
            "short_min_pixels": self.short_min_pixels,
        }


@dataclass(frozen=True, slots=True)
class FrameSequenceSummary:
    frame_index: int
    path: str
    candidate_count: int
    returned_count: int
    quality_count: int
    timestamp: str | None = None
    exposure_ms: float | None = None
    auxiliary: tuple[tuple[str, float], ...] = ()
    width_px: int | None = None
    height_px: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "quality_count": self.quality_count,
            "timestamp": self.timestamp,
            "exposure_ms": self.exposure_ms,
            "auxiliary": dict(self.auxiliary),
            "width_px": self.width_px,
            "height_px": self.height_px,
        }

    @property
    def auxiliary_dict(self) -> Mapping[str, float]:
        return dict(self.auxiliary)


@dataclass(frozen=True, slots=True)
class FixedSentinelAudit:
    """跨帧固定像素值审计；只提供诊断证据，不修改检测输入。"""

    sentinel_value: int
    frame_count: int
    image_shape: tuple[int, int] | None
    same_shape: bool
    total_occurrences: int
    per_frame_occurrences: tuple[int, ...]
    fixed_coordinate_count: int
    fixed_coordinates: tuple[tuple[int, int], ...]
    coordinates_truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "sentinel_value": self.sentinel_value,
            "frame_count": self.frame_count,
            "image_shape": list(self.image_shape) if self.image_shape is not None else None,
            "same_shape": self.same_shape,
            "total_occurrences": self.total_occurrences,
            "per_frame_occurrences": list(self.per_frame_occurrences),
            "fixed_coordinate_count": self.fixed_coordinate_count,
            "fixed_coordinates": [list(coordinate) for coordinate in self.fixed_coordinates],
            "coordinates_truncated": self.coordinates_truncated,
        }


def audit_fixed_sentinel(
    images: Sequence[np.ndarray],
    *,
    sentinel_value: int = -1,
    max_coordinates: int = 10_000,
) -> FixedSentinelAudit:
    """统计序列中恒定出现的精确像素值。

    比较前会排除比赛格式第一行前 104 个辅助字段位置。该函数刻意不把
    固定值直接加入质量掩膜：固定值可能是坏像素、填充值或编码约定，
    只有结合 FITS 头、原始范围和局部 PSF 证据后才能决定其物理含义。
    ``fixed_coordinates`` 使用 ``(x, y)`` 坐标，最多返回
    ``max_coordinates`` 个；``fixed_coordinate_count`` 始终是完整计数。
    """

    if max_coordinates < 0:
        raise ValueError("max_coordinates must be non-negative")
    frame_count = len(images)
    if frame_count == 0:
        return FixedSentinelAudit(
            sentinel_value=int(sentinel_value),
            frame_count=0,
            image_shape=None,
            same_shape=True,
            total_occurrences=0,
            per_frame_occurrences=(),
            fixed_coordinate_count=0,
            fixed_coordinates=(),
        )

    first_shape: tuple[int, int] | None = None
    same_shape = True
    fixed_mask: np.ndarray | None = None
    per_frame_occurrences: list[int] = []
    total_occurrences = 0
    for image in images:
        values = np.asarray(image)
        if values.ndim != 2:
            raise ValueError(f"expected a 2-D image, got shape {values.shape}")
        shape = (int(values.shape[0]), int(values.shape[1]))
        if first_shape is None:
            first_shape = shape
        elif shape != first_shape:
            same_shape = False

        mask = np.equal(values, sentinel_value)
        mask &= ~auxiliary_mask(shape)
        count = int(np.count_nonzero(mask))
        per_frame_occurrences.append(count)
        total_occurrences += count

        if same_shape:
            if fixed_mask is None:
                fixed_mask = mask.copy()
            else:
                fixed_mask &= mask
        else:
            # 不对不同尺寸的帧猜测坐标对应关系；仍保留逐帧/总量统计。
            fixed_mask = None

    if same_shape and fixed_mask is not None:
        coordinates_yx = np.argwhere(fixed_mask)
        fixed_coordinate_count = int(coordinates_yx.shape[0])
        kept_coordinates = coordinates_yx[:max_coordinates]
        fixed_coordinates = tuple(
            (int(x), int(y)) for y, x in kept_coordinates
        )
        coordinates_truncated = fixed_coordinate_count > len(fixed_coordinates)
    else:
        fixed_coordinate_count = 0
        fixed_coordinates = ()
        coordinates_truncated = False

    return FixedSentinelAudit(
        sentinel_value=int(sentinel_value),
        frame_count=frame_count,
        image_shape=first_shape,
        same_shape=same_shape,
        total_occurrences=total_occurrences,
        per_frame_occurrences=tuple(per_frame_occurrences),
        fixed_coordinate_count=fixed_coordinate_count,
        fixed_coordinates=fixed_coordinates,
        coordinates_truncated=coordinates_truncated,
    )


@dataclass(frozen=True, slots=True)
class FixedSentinelSourceImpact:
    """一个已测量源与固定异常坐标的孔径重叠证据。"""

    frame_index: int
    detection_id: int
    x: float
    y: float
    peak_x: float
    peak_y: float
    nearest_fixed_distance_px: float
    fixed_coordinate_count: int
    peak: float
    flux_snr: float | None
    quality_passed: bool
    flags: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "peak_x": self.peak_x,
            "peak_y": self.peak_y,
            "nearest_fixed_distance_px": self.nearest_fixed_distance_px,
            "fixed_coordinate_count": self.fixed_coordinate_count,
            "peak": self.peak,
            "flux_snr": self.flux_snr,
            "quality_passed": self.quality_passed,
            "flags": list(self.flags),
        }


@dataclass(frozen=True, slots=True)
class FixedSentinelImpactAudit:
    """固定异常坐标对候选峰和已测量源的影响汇总。"""

    frame_count: int
    affected_frame_count: int
    candidate_peak_data_available: bool
    candidate_peak_frame_count: int
    candidate_peak_count: int
    candidate_peak_affected_count: int
    affected_returned_source_count: int
    affected_quality_source_count: int
    records: tuple[FixedSentinelSourceImpact, ...]
    record_count: int
    records_truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "affected_frame_count": self.affected_frame_count,
            "candidate_peak_data_available": self.candidate_peak_data_available,
            "candidate_peak_frame_count": self.candidate_peak_frame_count,
            "candidate_peak_count": self.candidate_peak_count,
            "candidate_peak_affected_count": self.candidate_peak_affected_count,
            "affected_returned_source_count": self.affected_returned_source_count,
            "affected_quality_source_count": self.affected_quality_source_count,
            "record_count": self.record_count,
            "records_truncated": self.records_truncated,
            "records": [record.as_dict() for record in self.records],
        }


def audit_fixed_sentinel_impacts(
    frame_analyses: Sequence[FrameAnalysis],
    fixed_sentinel_audit: FixedSentinelAudit,
    *,
    max_records: int = 5_000,
) -> FixedSentinelImpactAudit:
    """把固定异常坐标关联到候选峰和已测量源。

    “受影响”定义为：固定坐标落入该帧检测参数中的圆孔径半径内。
    候选峰数组通常包含完整宽筛峰，而 ``Detection`` 记录只包含已经
    完成源级测量的返回候选；两种计数分开保存。这个函数不改变任何
    检测掩膜、像素或质量结果。
    """

    if max_records < 0:
        raise ValueError("max_records must be non-negative")
    frame_count = len(frame_analyses)
    fixed_coordinates = fixed_sentinel_audit.fixed_coordinates
    fixed_xy = (
        np.asarray(fixed_coordinates, dtype=np.float64)
        if fixed_coordinates
        else np.empty((0, 2), dtype=np.float64)
    )
    fixed_tree = cKDTree(fixed_xy) if fixed_xy.size else None
    candidate_peak_data_available = False
    candidate_peak_frame_count = 0
    candidate_peak_count = 0
    candidate_peak_affected_count = 0
    affected_frame_count = 0
    affected_returned_source_count = 0
    affected_quality_source_count = 0
    records: list[FixedSentinelSourceImpact] = []
    record_count = 0

    for frame_index, analysis in enumerate(frame_analyses):
        parameters = analysis.detection.parameters
        try:
            aperture_radius = float(parameters.get("aperture_radius", 4.0))
        except (TypeError, ValueError):
            aperture_radius = 4.0
        if not np.isfinite(aperture_radius) or aperture_radius <= 0:
            aperture_radius = 4.0

        frame_affected = False
        raw_candidates = np.asarray(analysis.detection.candidate_peaks)
        if raw_candidates.ndim == 2 and raw_candidates.shape[1] >= 2:
            candidate_peak_data_available = True
            candidate_peak_frame_count += 1
            candidate_peak_count += int(raw_candidates.shape[0])
            candidate_xy = np.asarray(raw_candidates[:, :2], dtype=np.float64)
            finite_candidates = np.isfinite(candidate_xy).all(axis=1)
            if fixed_tree is not None and finite_candidates.any():
                distances, indices = fixed_tree.query(
                    candidate_xy[finite_candidates],
                    distance_upper_bound=aperture_radius,
                )
                affected = np.isfinite(distances) & (indices < len(fixed_coordinates))
                candidate_peak_affected_count += int(np.count_nonzero(affected))
                frame_affected |= bool(np.any(affected))

        if fixed_tree is not None:
            for source in analysis.detection.sources:
                peak_x = (
                    float(source.peak_x)
                    if source.peak_x is not None and np.isfinite(source.peak_x)
                    else float(source.x)
                )
                peak_y = (
                    float(source.peak_y)
                    if source.peak_y is not None and np.isfinite(source.peak_y)
                    else float(source.y)
                )
                nearby = fixed_tree.query_ball_point((peak_x, peak_y), r=aperture_radius)
                if not nearby:
                    continue
                nearest_distance = float(fixed_tree.query((peak_x, peak_y))[0])
                record_count += 1
                affected_returned_source_count += 1
                affected_quality_source_count += int(bool(source.quality_passed))
                frame_affected = True
                if len(records) < max_records:
                    records.append(
                        FixedSentinelSourceImpact(
                            frame_index=frame_index,
                            detection_id=int(source.detection_id),
                            x=float(source.x),
                            y=float(source.y),
                            peak_x=peak_x,
                            peak_y=peak_y,
                            nearest_fixed_distance_px=nearest_distance,
                            fixed_coordinate_count=len(nearby),
                            peak=float(source.peak),
                            flux_snr=(
                                float(source.flux_snr)
                                if source.flux_snr is not None
                                else None
                            ),
                            quality_passed=bool(source.quality_passed),
                            flags=tuple(source.flags),
                        )
                    )
        affected_frame_count += int(frame_affected)

    return FixedSentinelImpactAudit(
        frame_count=frame_count,
        affected_frame_count=affected_frame_count,
        candidate_peak_data_available=candidate_peak_data_available,
        candidate_peak_frame_count=candidate_peak_frame_count,
        candidate_peak_count=candidate_peak_count,
        candidate_peak_affected_count=candidate_peak_affected_count,
        affected_returned_source_count=affected_returned_source_count,
        affected_quality_source_count=affected_quality_source_count,
        records=tuple(records),
        record_count=record_count,
        records_truncated=record_count > len(records),
    )


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """15 帧检测和关联的汇总结果。

    ``fixed_sentinel_audit`` 与 ``fixed_sentinel_impact_audit`` 是数据质量
    诊断字段，不会改变候选、质量源或轨迹的物理分类。
    """

    frames: tuple[FrameSequenceSummary, ...]
    cumulative_shifts: tuple[tuple[float, float], ...]
    tracks: tuple[SourceTrack, ...]
    link_radius_px: float
    min_presence: int
    motion_min_displacement_px: float
    max_motion_fit_rms_px: float
    motion_features: tuple[MotionFeatureTrack, ...] = ()
    motion_evidence_links: tuple[MotionEvidenceLink, ...] = ()
    motion_residual_threshold_adu: float = 100.0
    motion_reference_mode: str = "per_frame_background"
    motion_frame_audits: tuple[MotionFrameAudit, ...] = ()
    source_working_limit: int | None = None
    calculation_dtype: str = "float64"
    background_sample_limit: int | None = None
    fast_sequence: bool = False
    background_model_mode: str = "per_frame_local"
    persistent_min_presence: int = 0
    temporal_reference_candidate_count: int = 0
    temporal_coadd_candidate_count: int = 0
    temporal_proposal_mode: str = "median"
    temporal_candidate_min_snr: float = 7.5
    temporal_reference_min_snr: float = 15.0
    temporal_multiscale: bool = False
    temporal_min_psf_correlation: float = 0.8
    candidate_consensus_audit: tuple[tuple[str, int], ...] = ()
    stack_faint_candidate_count: int = 0
    stack_reference_mode: str = "median"
    stack_threshold_sigma: float = 4.0
    stack_min_flux_snr: float = 5.0
    stack_frame_min_flux_snr: float = 3.0
    stack_min_presence: int = 0
    fixed_sentinel_audit: FixedSentinelAudit | None = None
    fixed_sentinel_impact_audit: FixedSentinelImpactAudit | None = None
    relative_photometry: RelativePhotometryResult | None = None
    track_association_method: str = (
        "local_component_hungarian+exposure_velocity_gate+line_geometry_soft_cost+motion_axis_gate+contiguous_run_gate"
    )
    detector_parameters: tuple[tuple[str, object], ...] = ()
    registration_model: str = "translation"
    registration_affine_frame_count: int = 0
    registration_residual_median_px: float | None = None
    effective_motion_min_displacement_px: float | None = None
    # 线状曝光候选的关联半径与点源轨迹不同。点源通常只需几像素的
    # 最近邻门，而 1.5 s 曝光拖影的中心会随帧间位移变化；把两者混成
    # 一个字段会让 GUI 显示值和实际执行值不一致。放在新增字段末尾，
    # 保持既有位置参数构造方式兼容。
    motion_line_link_radius_px: float = 96.0

    @property
    def stable_source_count(self) -> int:
        return sum(track.classification == "static" for track in self.tracks)

    @property
    def moving_track_count(self) -> int:
        return sum(track.classification == "moving" for track in self.tracks)

    @property
    def motion_evidence_link_count(self) -> int:
        """点状与线状 moving 证据被判定为同一源的关联数。"""

        return len(self.motion_evidence_links)

    @property
    def motion_evidence_group_count(self) -> int:
        """去掉跨通道重复后的运动证据组数；不是物理目标真值。"""

        moving_lines = sum(track.classification == "moving" for track in self.motion_features)
        return max(0, self.moving_track_count + moving_lines - self.motion_evidence_link_count)

    @property
    def fast_point_motion_count(self) -> int:
        """由高速点源补充关联确认的 moving 轨迹数。"""

        return sum(
            track.classification == "moving"
            and track.evidence_level == FAST_POINT_MOTION_EVIDENCE
            for track in self.tracks
        )

    @property
    def transient_track_count(self) -> int:
        return sum(track.classification == "transient" for track in self.tracks)

    @property
    def persistent_source_count(self) -> int:
        """达到较低持续性门槛、但未达到严格静态门槛的点源轨迹数。

        叠加参考图恢复的 ``stack_faint`` 轨迹单独计入
        :meth:`stack_faint_count`，不混入本计数。
        """

        return sum(
            track.classification == "persistent" and track.evidence_level != "stack_faint"
            for track in self.tracks
        )

    @property
    def candidate_consensus_count(self) -> int:
        """由补充候选通道跨帧共识得到的低置信稳定候选总数。"""

        return sum(
            track.evidence_level in {"candidate_consensus", "temporal_reference"}
            for track in self.tracks
        )

    @property
    def temporal_reference_count(self) -> int:
        """由注册时间中值补提案形成的低置信候选数。"""

        return sum(track.evidence_level == "temporal_reference" for track in self.tracks)

    @property
    def stack_faint_count(self) -> int:
        """由叠加参考图恢复、经逐帧强制测光确认的低置信暗星轨迹数。

        这是单帧质量门下的低置信补充层，不是官方逐星真值。
        """

        return sum(track.evidence_level == "stack_faint" for track in self.tracks)

    @property
    def stable_field_candidate_count(self) -> int:
        """严格静态源与持续源候选的合计；不是物理恒星真值。"""

        return self.stable_source_count + self.persistent_source_count

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": len(self.frames),
            "stable_source_count": self.stable_source_count,
            "persistent_source_count": self.persistent_source_count,
            "stable_field_candidate_count": self.stable_field_candidate_count,
            "candidate_consensus_count": self.candidate_consensus_count,
            "temporal_reference_count": self.temporal_reference_count,
            "stack_faint_count": self.stack_faint_count,
            "moving_track_count": self.moving_track_count,
            "motion_evidence_link_count": self.motion_evidence_link_count,
            "motion_evidence_group_count": self.motion_evidence_group_count,
            "fast_point_motion_count": self.fast_point_motion_count,
            "transient_track_count": self.transient_track_count,
            "link_radius_px": self.link_radius_px,
            "min_presence": self.min_presence,
            "motion_min_displacement_px": self.motion_min_displacement_px,
            "max_motion_fit_rms_px": self.max_motion_fit_rms_px,
            "motion_line_link_radius_px": self.motion_line_link_radius_px,
            "track_association_method": self.track_association_method,
            "registration_model": self.registration_model,
            "registration_affine_frame_count": self.registration_affine_frame_count,
            "registration_residual_median_px": self.registration_residual_median_px,
            "effective_motion_min_displacement_px": self.effective_motion_min_displacement_px,
            "detector_parameters": dict(self.detector_parameters),
            "motion_residual_threshold_adu": self.motion_residual_threshold_adu,
            "motion_reference_mode": self.motion_reference_mode,
            "motion_frame_audits": [audit.as_dict() for audit in self.motion_frame_audits],
            "source_working_limit": self.source_working_limit,
            "calculation_dtype": self.calculation_dtype,
            "background_sample_limit": self.background_sample_limit,
            "fast_sequence": self.fast_sequence,
            "background_model_mode": self.background_model_mode,
            "persistent_min_presence": self.persistent_min_presence,
            "temporal_reference_candidate_count": self.temporal_reference_candidate_count,
            "temporal_coadd_candidate_count": self.temporal_coadd_candidate_count,
            "temporal_proposal_mode": self.temporal_proposal_mode,
            "temporal_candidate_min_snr": self.temporal_candidate_min_snr,
            "temporal_reference_min_snr": self.temporal_reference_min_snr,
            "temporal_multiscale": self.temporal_multiscale,
            "temporal_min_psf_correlation": self.temporal_min_psf_correlation,
            "candidate_consensus_audit": dict(self.candidate_consensus_audit),
            "stack_faint_candidate_count": self.stack_faint_candidate_count,
            "stack_reference_mode": self.stack_reference_mode,
            "stack_threshold_sigma": self.stack_threshold_sigma,
            "stack_min_flux_snr": self.stack_min_flux_snr,
            "stack_frame_min_flux_snr": self.stack_frame_min_flux_snr,
            "stack_min_presence": self.stack_min_presence,
            "fixed_sentinel_audit": (
                self.fixed_sentinel_audit.as_dict()
                if self.fixed_sentinel_audit is not None
                else None
            ),
            "fixed_sentinel_impact_audit": (
                self.fixed_sentinel_impact_audit.as_dict()
                if self.fixed_sentinel_impact_audit is not None
                else None
            ),
            "relative_photometry": (
                self.relative_photometry.as_dict()
                if self.relative_photometry is not None
                else None
            ),
            "frames": [frame.as_dict() for frame in self.frames],
            "cumulative_shifts": [list(shift) for shift in self.cumulative_shifts],
            "tracks": [track.as_dict() for track in self.tracks],
            "motion_features": [track.as_dict() for track in self.motion_features],
            "motion_evidence_links": [link.as_dict() for link in self.motion_evidence_links],
        }


def _signal_snr(source: Detection) -> float:
    value = source.flux_snr if source.flux_snr is not None else source.snr
    return float(value) if np.isfinite(value) else 0.0


def _estimate_translation(
    previous: Sequence[Detection],
    current: Sequence[Detection],
    *,
    radius_px: float,
    reference_limit: int = 1500,
) -> tuple[float, float]:
    """用高 SNR 源的最大一致位移簇估计 ``current - previous`` 平移。

    ``radius_px`` 是旧接口中的局部搜索半径，但它不应再限制注册器的
    视场漂移搜索范围：轨迹关联仍可保持很小的半径，注册阶段则先在一
    个有限的宽搜索窗内找互为最近邻的源对，再在二维位移空间中找最大
    一致簇。这样可以区分“真实的 18 px 漂移”和密集星场里数量较少的
    偶然近邻，而不是把错误近邻的中位数写进累计配准量。

    该方法仍是平移模型；若一致簇本身很弱，保守回退到原有的局部
    中位数逻辑，避免在低源数/测试数据上凭空制造大位移。
    """

    if not previous or not current:
        return 0.0, 0.0
    previous_top = sorted(previous, key=_signal_snr, reverse=True)[:reference_limit]
    current_top = sorted(current, key=_signal_snr, reverse=True)[:reference_limit]
    previous_points = np.array([(source.x, source.y) for source in previous_top], dtype=np.float64)
    current_points = np.array([(source.x, source.y) for source in current_top], dtype=np.float64)
    # 注册搜索与点轨迹关联解耦。上限仍然有限，防止在完全不相关的两
    # 帧之间把全幅随机近邻当成配准证据；默认 64 px 足以覆盖本次真实
    # 数据测到的 37--38 px 相邻帧位移。
    search_radius = max(
        float(radius_px),
        min(
            DEFAULT_REGISTRATION_SEARCH_RADIUS_PX,
            max(32.0, 8.0 * float(radius_px)),
        ),
    )
    current_tree = cKDTree(current_points)
    previous_tree = cKDTree(previous_points)
    previous_distances, previous_indices = current_tree.query(
        previous_points,
        distance_upper_bound=search_radius,
    )
    _current_distances, current_indices = previous_tree.query(
        current_points,
        distance_upper_bound=search_radius,
    )
    valid = np.isfinite(previous_distances) & (previous_indices < len(current_top))
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return 0.0, 0.0
    reciprocal = current_indices[previous_indices[valid_indices]] == valid_indices
    valid_indices = valid_indices[reciprocal]
    if valid_indices.size == 0:
        return 0.0, 0.0
    deltas = current_points[previous_indices[valid_indices]] - previous_points[valid_indices]
    # 只保留互相最近的一对一匹配，避免一个当前源被多个前帧源重复使用。
    # 真配对会在二维位移空间形成紧密簇；对簇做 KD-tree 邻域计数，比
    # 直接对混合了随机近邻的全部 delta 求中位数更不容易被密集星场
    # 拖偏。``return_length`` 只返回整数计数，不创建可观的邻接列表。
    if deltas.shape[0] >= 3:
        delta_tree = cKDTree(deltas)
        consensus_counts = delta_tree.query_ball_point(
            deltas,
            r=REGISTRATION_CONSENSUS_RADIUS_PX,
            return_length=True,
        )
        seed_index = int(np.argmax(consensus_counts))
        consensus_indices = np.asarray(
            delta_tree.query_ball_point(
                deltas[seed_index],
                r=REGISTRATION_CONSENSUS_RADIUS_PX,
            ),
            dtype=np.intp,
        )
        if consensus_indices.size >= 3:
            center = np.median(deltas[consensus_indices], axis=0)
            refined = np.flatnonzero(
                np.linalg.norm(deltas - center, axis=1)
                <= REGISTRATION_CONSENSUS_RADIUS_PX
            )
            if refined.size >= 3:
                return tuple(float(value) for value in np.median(deltas[refined], axis=0))
            return tuple(float(value) for value in center)
    # 源数太少时保留旧的稳健回退，兼容短序列和最小单元测试。
    center = np.median(deltas, axis=0)
    residual = np.linalg.norm(deltas - center, axis=1)
    mad = np.median(np.abs(residual - np.median(residual)))
    cutoff = max(1.0, 3.0 * 1.4826 * mad)
    inliers = residual <= cutoff
    if np.count_nonzero(inliers) < 3:
        return float(center[0]), float(center[1])
    return tuple(float(value) for value in np.median(deltas[inliers], axis=0))


def _fit_affine_registration_model(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, float, int] | None:
    """从跨帧源对中稳健拟合 ``observed = A @ reference + b``。

    当前序列的主要误差不是任意的局部形变，而是视场平移之外还带有
    很小的旋转/尺度变化。这里使用固定随机种子的 RANSAC 找空间上
    一致的三点仿射模型，再用 robust trimming 重拟合。移动目标、错误
    近邻和偶发质心跳动只能作为离群点参与，不能单独决定全局模型。

    返回模型、最终内点残差中位数和内点数；证据不足或模型明显失真时
    返回 ``None``，调用方会保守地退回纯平移配准。
    """

    reference = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if (
        reference.ndim != 2
        or observed.ndim != 2
        or reference.shape != observed.shape
        or reference.shape[1] != 2
        or reference.shape[0] < REGISTRATION_AFFINE_MIN_PAIRS
    ):
        return None
    finite = np.isfinite(reference).all(axis=1) & np.isfinite(observed).all(axis=1)
    reference = reference[finite]
    observed = observed[finite]
    if reference.shape[0] < REGISTRATION_AFFINE_MIN_PAIRS:
        return None

    design = np.column_stack((reference, np.ones(reference.shape[0], dtype=np.float64)))

    def valid_model(coefficients: np.ndarray) -> bool:
        if coefficients.shape != (3, 2) or not np.isfinite(coefficients).all():
            return False
        matrix = coefficients[:2, :].T
        determinant = float(np.linalg.det(matrix))
        if not np.isfinite(determinant) or determinant <= 0.0:
            return False
        deviation = np.abs(matrix - np.eye(2, dtype=np.float64))
        # 平移已在上游消除；如果这里出现大尺度/大剪切，通常意味着
        # 当前帧没有真正的对应源，宁可放弃仿射修正也不制造假稳定。
        return bool(
            np.max(np.diag(deviation)) <= REGISTRATION_AFFINE_MAX_SCALE_DEVIATION
            and abs(float(matrix[0, 1])) <= REGISTRATION_AFFINE_MAX_SKEW
            and abs(float(matrix[1, 0])) <= REGISTRATION_AFFINE_MAX_SKEW
        )

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        return np.linalg.norm(design @ coefficients - observed, axis=1)

    rng = np.random.default_rng(int(seed))
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    point_count = int(reference.shape[0])
    for _ in range(REGISTRATION_AFFINE_RANSAC_TRIALS):
        sample_indices = rng.choice(point_count, size=3, replace=False)
        sample_design = design[sample_indices]
        if np.linalg.matrix_rank(sample_design) < 3:
            continue
        coefficients = np.linalg.lstsq(
            sample_design,
            observed[sample_indices],
            rcond=None,
        )[0]
        if not valid_model(coefficients):
            continue
        residual = residuals(coefficients)
        mask = residual <= REGISTRATION_AFFINE_INLIER_RADIUS_PX
        inlier_count = int(np.count_nonzero(mask))
        if inlier_count < REGISTRATION_AFFINE_MIN_PAIRS:
            continue
        median_residual = float(np.median(residual[mask]))
        score = (inlier_count, -median_residual)
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        return None
    inliers = best_mask
    coefficients = np.linalg.lstsq(design[inliers], observed[inliers], rcond=None)[0]
    for _ in range(6):
        if not valid_model(coefficients):
            return None
        residual = residuals(coefficients)
        median_residual = float(np.median(residual[inliers])) if np.any(inliers) else np.inf
        mad = float(np.median(np.abs(residual[inliers] - median_residual))) if np.any(inliers) else np.inf
        cutoff = max(
            REGISTRATION_AFFINE_INLIER_RADIUS_PX,
            min(3.0, median_residual + 3.0 * 1.4826 * mad),
        )
        updated = residual <= cutoff
        if np.count_nonzero(updated) < REGISTRATION_AFFINE_MIN_PAIRS:
            break
        if np.array_equal(updated, inliers):
            break
        inliers = updated
        coefficients = np.linalg.lstsq(design[inliers], observed[inliers], rcond=None)[0]

    if not valid_model(coefficients) or np.count_nonzero(inliers) < REGISTRATION_AFFINE_MIN_PAIRS:
        return None
    inlier_fraction = float(np.count_nonzero(inliers)) / float(point_count)
    if inlier_fraction < 0.5:
        return None
    residual = residuals(coefficients)
    return (
        coefficients.T.astype(np.float64, copy=False),
        float(np.median(residual[inliers])),
        int(np.count_nonzero(inliers)),
    )


def _estimate_sequence_affine_models(
    tracks: Sequence[Sequence[TrackPoint]],
    frame_count: int,
) -> tuple[tuple[np.ndarray, ...], int, float | None]:
    """用首帧稳定源为每一帧估计平移后的全局仿射残差。"""

    identity = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64)
    models: list[np.ndarray] = [identity.copy()]
    frame_residuals: list[float] = []
    fitted_frame_count = 0
    for frame_index in range(1, int(frame_count)):
        reference_rows: list[tuple[float, float]] = []
        observed_rows: list[tuple[float, float]] = []
        for history in tracks:
            reference_point = next(
                (point for point in history if point.frame_index == 0),
                None,
            )
            observed_point = next(
                (point for point in history if point.frame_index == frame_index),
                None,
            )
            if reference_point is None or observed_point is None:
                continue
            reference_rows.append((reference_point.aligned_x, reference_point.aligned_y))
            observed_rows.append((observed_point.aligned_x, observed_point.aligned_y))
        if len(reference_rows) < REGISTRATION_AFFINE_MIN_PAIRS:
            models.append(identity.copy())
            continue
        fit = _fit_affine_registration_model(
            np.asarray(reference_rows, dtype=np.float64),
            np.asarray(observed_rows, dtype=np.float64),
            seed=20260913 + frame_index,
        )
        if fit is None:
            models.append(identity.copy())
            continue
        model, residual_median, _inlier_count = fit
        models.append(model)
        fitted_frame_count += 1
        frame_residuals.append(float(residual_median))
    return (
        tuple(models),
        fitted_frame_count,
        float(np.median(frame_residuals)) if frame_residuals else None,
    )


def _inverse_affine_coordinates(
    coordinates: np.ndarray,
    model: np.ndarray,
) -> np.ndarray:
    """把平移配准坐标变换到全局仿射校正后的参考坐标。"""

    matrix = np.asarray(model[:, :2], dtype=np.float64)
    offset = np.asarray(model[:, 2], dtype=np.float64)
    try:
        return np.linalg.solve(matrix, (np.asarray(coordinates, dtype=np.float64) - offset).T).T
    except np.linalg.LinAlgError:
        # 模型已经过合法性检查；这个分支仅防止外部构造的缓存/测试
        # 数据传入奇异矩阵时让整组序列崩溃。
        return np.asarray(coordinates, dtype=np.float64)


def _fit_track_coordinates(
    points: Sequence[TrackPoint],
    coordinates: np.ndarray,
    *,
    frame_times: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float, float | None]:
    if len(points) < 2:
        return 0.0, 0.0, None
    time = _track_point_motion_times(points, frame_times)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (len(points), 2):
        raise ValueError("track coordinates must have shape (len(points), 2)")
    x_slope, x_intercept = np.polyfit(time, coordinates[:, 0], 1)
    y_slope, y_intercept = np.polyfit(time, coordinates[:, 1], 1)
    fitted = np.column_stack(
        (
            x_slope * time + x_intercept,
            y_slope * time + y_intercept,
        )
    )
    rms = float(np.sqrt(np.mean(np.sum((coordinates - fitted) ** 2, axis=1))))
    duration = float(time[-1] - time[0])
    displacement = float(np.hypot(x_slope * duration, y_slope * duration))
    speed = float(np.hypot(x_slope, y_slope))
    if frame_times is not None:
        # Keep the serialized/public field name ``speed_px_per_frame`` while
        # fitting against seconds.  Convert px/s back to px per median
        # observation interval so old UI/cache consumers keep their units.
        try:
            raw_times = np.asarray(frame_times, dtype=np.float64)
            frame_indices = np.asarray(
                [int(point.frame_index) for point in points],
                dtype=np.intp,
            )
            if raw_times.ndim == 1 and raw_times.shape == (len(points),):
                selected_times = raw_times
            elif raw_times.ndim == 1 and frame_indices.size and int(frame_indices.max()) < raw_times.size:
                selected_times = raw_times[frame_indices]
            else:
                selected_times = np.empty(0, dtype=np.float64)
            if selected_times.size >= 2 and np.all(np.isfinite(selected_times)) and np.all(np.diff(selected_times) > 0.0):
                speed *= float(np.median(np.diff(selected_times)))
        except (TypeError, ValueError, IndexError):
            pass
    return displacement, speed, rms


def _track_point_motion_times(
    points: Sequence[TrackPoint],
    frame_times: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    """Resolve one increasing time value per point-track observation.

    ``TrackPoint.frame_index`` is still the association key, but it is not a
    physical time.  Real FITS sequences in this project carry ``DATE-OBS``
    with roughly 1.5 s gaps, and the gaps are not exactly equal.  Accept both
    a full sequence time vector and a point-local vector so the private fit
    helper remains convenient for tests and downstream callers.
    """

    fallback = np.asarray(
        [float(point.frame_index) for point in points],
        dtype=np.float64,
    )
    if not points or frame_times is None:
        return fallback
    try:
        values = np.asarray(frame_times, dtype=np.float64)
    except (TypeError, ValueError):
        return fallback
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        return fallback
    frame_indices = np.asarray(
        [int(point.frame_index) for point in points],
        dtype=np.intp,
    )
    if values.shape == (len(points),):
        selected = values
    elif frame_indices.size and int(frame_indices.max()) < values.size:
        selected = values[frame_indices]
    else:
        return fallback
    if not np.all(np.isfinite(selected)) or np.any(np.diff(selected) <= 0.0):
        return fallback
    return selected - selected[0]


def _derive_motion_displacement_floor(
    metrics: Sequence[tuple[int, float, float, float | None]],
    *,
    required_presence: int,
    user_floor: float,
    max_motion_fit_rms_px: float,
    affine_frame_count: int,
) -> float:
    """估计配准残差可伪造的最大低速位移，并返回分类下限。"""

    if affine_frame_count <= 0:
        return float(user_floor)
    displacements = np.asarray(
        [
            displacement
            for presence, displacement, _speed, fit_rms in metrics
            if presence >= int(required_presence)
            and fit_rms is not None
            and np.isfinite(float(fit_rms))
            and float(fit_rms) <= float(max_motion_fit_rms_px)
            and np.isfinite(float(displacement))
        ],
        dtype=np.float64,
    )
    if displacements.size < REGISTRATION_MOTION_FLOOR_MIN_TRACKS:
        return float(user_floor)
    median = float(np.median(displacements))
    mad = float(np.median(np.abs(displacements - median)))
    if not np.isfinite(median) or not np.isfinite(mad):
        return float(user_floor)
    robust_floor = median + REGISTRATION_MOTION_FLOOR_MAD_FACTOR * 1.4826 * mad
    if not np.isfinite(robust_floor):
        return float(user_floor)
    return max(float(user_floor), float(robust_floor))


def _fit_track(
    points: Sequence[TrackPoint],
    *,
    frame_times: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float, float | None]:
    if len(points) < 2:
        return 0.0, 0.0, None
    return _fit_track_coordinates(
        points,
        np.asarray(
            [(point.aligned_x, point.aligned_y) for point in points],
            dtype=np.float64,
        ),
        frame_times=frame_times,
    )


def _global_frame_assignments(
    track_positions: Sequence[tuple[int, float, float]],
    current_points: np.ndarray,
    *,
    link_radius_px: float,
    pair_allowed: Callable[[int, int, float], bool] | None = None,
    pair_cost: Callable[[int, int, float], float] | None = None,
) -> tuple[tuple[float, int, int], ...]:
    """在当前帧内对候选轨迹和源点做局部全局一对一关联。

    旧实现对每条轨迹只查询一个最近源，然后按距离贪心占用。这个策略
    在两个轨迹共享一个最近源时会先满足距离更小的轨迹，可能让另一条轨迹
    被拆断，即使存在一个总距离更小、且两条轨迹都能延续的组合。

    这里先用半径邻域建立二分图的连通分量，再在每个小分量内运行
    Hungarian assignment。分量化避免在整幅高密度星场上创建
    ``轨迹数 × 源数`` 的巨大矩阵；没有邻域边的轨迹/源不参与匹配。
    ``pair_allowed`` 可在半径搜索后再加一个候选对级别的科学门控；它
    从二分图中移除不允许的边。``pair_cost`` 则只增加候选对的软代价，
    用于把方向、端点等几何证据纳入 Hungarian 的选择，但不会把一个
    半径内的候选硬删掉。返回值仍保持旧调用方需要的
    ``(distance, track_index, source_index)`` 结构，并按确定性顺序排列。
    """

    if link_radius_px <= 0:
        raise ValueError("link_radius_px must be positive")
    if not track_positions or current_points.size == 0:
        return ()
    points = np.asarray(current_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("current_points must have shape (n, 2)")
    positions = np.asarray([(item[1], item[2]) for item in track_positions], dtype=np.float64)
    tree = cKDTree(points)
    neighbors_by_row: list[list[int]] = []
    for row_index, indices in enumerate(
        tree.query_ball_point(positions, float(link_radius_px))
    ):
        accepted: list[int] = []
        for index in indices:
            source_index = int(index)
            distance = float(np.hypot(*(positions[row_index] - points[source_index])))
            if pair_allowed is not None and not pair_allowed(
                int(track_positions[row_index][0]),
                source_index,
                distance,
            ):
                continue
            accepted.append(source_index)
        neighbors_by_row.append(sorted(accepted))
    source_to_rows: dict[int, list[int]] = {}
    for row_index, source_indices in enumerate(neighbors_by_row):
        for source_index in source_indices:
            source_to_rows.setdefault(source_index, []).append(row_index)
    if not source_to_rows:
        return ()

    visited_rows: set[int] = set()
    components: list[tuple[list[int], list[int]]] = []
    for start_row, start_neighbors in enumerate(neighbors_by_row):
        if not start_neighbors or start_row in visited_rows:
            continue
        component_rows: set[int] = set()
        component_sources: set[int] = set()
        pending_rows = [start_row]
        while pending_rows:
            row_index = pending_rows.pop()
            if row_index in component_rows:
                continue
            component_rows.add(row_index)
            visited_rows.add(row_index)
            for source_index in neighbors_by_row[row_index]:
                if source_index in component_sources:
                    continue
                component_sources.add(source_index)
                pending_rows.extend(
                    linked_row
                    for linked_row in source_to_rows[source_index]
                    if linked_row not in component_rows
                )
        components.append((sorted(component_rows), sorted(component_sources)))

    assignments: list[tuple[float, int, int]] = []
    invalid_cost = max(1.0, float(link_radius_px)) * 1_000_000.0
    for component_rows, component_sources in components:
        row_count = len(component_rows)
        source_count = len(component_sources)
        costs = np.full((row_count, source_count), invalid_cost, dtype=np.float64)
        source_columns = {source_index: column for column, source_index in enumerate(component_sources)}
        for row, global_row in enumerate(component_rows):
            for source_index in neighbors_by_row[global_row]:
                column = source_columns[source_index]
                distance = float(np.hypot(*(positions[global_row] - points[source_index])))
                # A tiny deterministic tie-break leaves the geometric cost
                # unchanged at display precision but avoids platform-dependent
                # choices for exactly symmetric points.
                extra_cost = 0.0
                if pair_cost is not None:
                    extra_cost = float(
                        pair_cost(
                            int(track_positions[global_row][0]),
                            source_index,
                            distance,
                        )
                    )
                    # 软评分失效时回退到距离，避免 NaN 污染 Hungarian
                    # 矩阵，也避免诊断函数把一个合法边变成负代价。
                    if not np.isfinite(extra_cost) or extra_cost < 0.0:
                        extra_cost = 0.0
                costs[row, column] = (
                    distance
                    + extra_cost
                    + 1e-9 * (row * source_count + column)
                )
        row_indices, source_columns_selected = linear_sum_assignment(costs)
        for row, column in zip(row_indices.tolist(), source_columns_selected.tolist()):
            global_row = component_rows[row]
            source_index = component_sources[column]
            actual_distance = float(
                np.hypot(*(positions[global_row] - points[source_index]))
            )
            # 半径是物理搜索门，必须检查原始距离而不是“距离 + 软
            # 代价”。否则合法候选可能因为形状惩罚被误认为超出半径。
            if actual_distance > float(link_radius_px) + 1e-7:
                continue
            assignments.append(
                (actual_distance, int(track_positions[global_row][0]), source_index)
            )
    assignments.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(assignments)


def track_fast_point_movers(
    frame_sources: Sequence[Sequence[Detection]],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    frame_times: Sequence[float] | np.ndarray | None = None,
    link_radius_px: float = 4.0,
    min_presence: int | None = None,
    motion_min_displacement_px: float = 2.0,
    min_flux_snr: float = DEFAULT_FAST_POINT_MIN_SNR,
    max_step_px: float = DEFAULT_FAST_POINT_MAX_STEP_PX,
    gate_px: float = DEFAULT_FAST_POINT_GATE_PX,
    max_fit_rms_px: float = DEFAULT_FAST_POINT_FIT_RMS_PX,
    existing_tracks: Sequence[SourceTrack] = (),
) -> tuple[SourceTrack, ...]:
    """补充关联超出普通最近邻半径的点状运动目标。

    普通 ``track_detections`` 的 4 px 最近邻关联是固定星场的保守基线：
    它能避免在高密度星场中误连，但一个每帧移动 8--10 px 的点目标会被
    拆成许多 ``transient``。本函数只处理这类“单帧质量已通过、但没有被
    普通关联串起来”的源，流程是：

    1. 在注册坐标中，以相邻帧源对的位移作为宽筛速度种子；
    2. 按常速度预测，在每帧 ``gate_px`` 邻域内选一个未占用的质量源；
    3. 要求至少达到序列持续性、最小速度、逐点残差和常速度拟合 RMS；
    4. 对已被普通多点轨迹占用的源不再复用，并合并同一条轨迹的不同种子。

    这不是把 ``link_radius_px`` 全局放大：静态星场仍按原有关联口径，
    高速点源则单独写入 ``evidence_level='fast_point_motion'``，便于 UI、
    缓存和答辩材料区分“严格质量点源”与“速度关联补充证据”。默认
    ``min_flux_snr=7``、``max_fit_rms_px=1.5`` 是补充层门槛，不降低单帧
    的 SNR/PSF/掩膜/线状伪迹规则。返回轨迹的点仍然全部来自
    ``quality_sources``，因此不会把候选噪点直接升级为运动目标。

    ``cumulative_shifts`` 使用与 :class:`TrackPoint` 相同的约定：原始坐标
    减去全局平移得到注册坐标。该函数不读取或写入 FITS/cache，方便单元
    测试，也避免在序列结果已经释放单帧对象后重复计算。
    """

    if not frame_sources:
        return ()
    frame_count = len(frame_sources)
    if frame_count < 2:
        # 点状运动至少需要一对跨帧观测；单帧只保留普通检测/长线候选。
        return ()
    if len(cumulative_shifts) != frame_count:
        raise ValueError("cumulative_shifts length must match frame_sources")
    resolved_frame_times = _normalise_motion_times(frame_times, frame_count)
    reference_interval_s = _reference_time_interval_seconds(resolved_frame_times)
    if link_radius_px <= 0 or motion_min_displacement_px <= 0:
        raise ValueError("link radius and minimum displacement must be positive")
    if min_flux_snr <= 0 or max_step_px <= 0 or gate_px <= 0 or max_fit_rms_px <= 0:
        raise ValueError("fast point motion thresholds must be positive")
    required_presence = (
        int(min_presence)
        if min_presence is not None
        else max(3, int(np.ceil(frame_count * 0.8)))
    )
    if required_presence < 2 or required_presence > frame_count:
        raise ValueError("min_presence must be within 2..frame count")
    if max_step_px <= link_radius_px:
        raise ValueError("max_step_px must exceed link_radius_px")

    # 只让“没有在普通轨迹中形成多帧证据”的质量点进入补充层。这样一
    # 颗固定星即使在相邻帧有一次误配，也不会同时出现在静态层和高速层。
    reserved: set[tuple[int, int]] = set()
    for track in existing_tracks:
        if track.presence < 2:
            continue
        reserved.update(
            (int(point.frame_index), int(point.detection_id))
            for point in track.points
        )

    # 先按补充层的最低 SNR 做廉价宽筛，并将坐标转换到注册坐标。质量源
    # 数量通常远小于全量候选，KD-tree 查询不会把数万噪点带入速度搜索。
    frame_records: list[list[tuple[Detection, float, float, float]]] = []
    trees: list[cKDTree | None] = []
    for frame_index, sources in enumerate(frame_sources):
        shift_x, shift_y = cumulative_shifts[frame_index]
        records: list[tuple[Detection, float, float, float]] = []
        for source in sources:
            if not bool(source.quality_passed):
                continue
            if (frame_index, int(source.detection_id)) in reserved:
                continue
            signal_snr = _signal_snr(source)
            if not np.isfinite(signal_snr) or signal_snr < float(min_flux_snr):
                continue
            records.append(
                (
                    source,
                    float(source.x) - float(shift_x),
                    float(source.y) - float(shift_y),
                    float(signal_snr),
                )
            )
        frame_records.append(records)
        if records:
            trees.append(cKDTree(np.asarray([(item[1], item[2]) for item in records], dtype=np.float64)))
        else:
            trees.append(None)

    if not any(frame_records):
        return ()

    def to_point(frame_index: int, record: tuple[Detection, float, float, float]) -> TrackPoint:
        source, aligned_x, aligned_y, signal_snr = record
        return TrackPoint(
            frame_index=frame_index,
            detection_id=int(source.detection_id),
            x=float(source.x),
            y=float(source.y),
            aligned_x=aligned_x,
            aligned_y=aligned_y,
            flux_snr=signal_snr,
            quality_passed=bool(source.quality_passed),
        )

    def fit_prediction(
        observations: Sequence[tuple[int, tuple[Detection, float, float, float]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(
            [resolved_frame_times[item[0]] for item in observations],
            dtype=np.float64,
        )
        coordinates = np.asarray([(item[1][1], item[1][2]) for item in observations], dtype=np.float64)
        design = np.column_stack((np.ones(times.size, dtype=np.float64), times))
        x_coefficients = np.linalg.lstsq(design, coordinates[:, 0], rcond=None)[0]
        y_coefficients = np.linalg.lstsq(design, coordinates[:, 1], rcond=None)[0]
        return x_coefficients, y_coefficients

    def choose_nearby(
        frame_index: int,
        predicted: tuple[float, float],
        used: set[tuple[int, int]],
    ) -> tuple[Detection, float, float, float] | None:
        tree = trees[frame_index]
        if tree is None:
            return None
        candidate_indices = tree.query_ball_point(predicted, float(gate_px))
        choices: list[tuple[float, float, int]] = []
        for candidate_index in candidate_indices:
            record = frame_records[frame_index][int(candidate_index)]
            source = record[0]
            key = (frame_index, int(source.detection_id))
            if key in used:
                continue
            distance = float(np.hypot(record[1] - predicted[0], record[2] - predicted[1]))
            # 预测距离优先；同距离时优先 SNR 更高的质量源，避免密集
            # 区域的弱噪点抢占真实点目标的位置。
            choices.append((distance, -record[3], int(candidate_index)))
        if not choices:
            return None
        choices.sort()
        return frame_records[frame_index][choices[0][2]]

    def follow_seed(
        start_frame: int,
        first: tuple[Detection, float, float, float],
        second: tuple[Detection, float, float, float],
    ) -> tuple[tuple[int, tuple[Detection, float, float, float]], ...]:
        observations: dict[int, tuple[Detection, float, float, float]] = {
            start_frame: first,
            start_frame + 1: second,
        }
        used = {
            (start_frame, int(first[0].detection_id)),
            (start_frame + 1, int(second[0].detection_id)),
        }

        # 允许分散的一帧缺测，但不允许连续缺测。required_presence 仍是
        # 主要持续性门槛；这个额外限制能显著降低随机峰“跳跃拼线”的概率。
        missed = 0
        for frame_index in range(start_frame + 2, frame_count):
            x_coefficients, y_coefficients = fit_prediction(tuple(sorted(observations.items())))
            target_time = float(resolved_frame_times[frame_index])
            predicted = (
                float(x_coefficients[0] + x_coefficients[1] * target_time),
                float(y_coefficients[0] + y_coefficients[1] * target_time),
            )
            selected = choose_nearby(frame_index, predicted, used)
            if selected is None:
                missed += 1
                if missed > 1:
                    break
                continue
            missed = 0
            observations[frame_index] = selected
            used.add((frame_index, int(selected[0].detection_id)))

        missed = 0
        for frame_index in range(start_frame - 1, -1, -1):
            x_coefficients, y_coefficients = fit_prediction(tuple(sorted(observations.items())))
            target_time = float(resolved_frame_times[frame_index])
            predicted = (
                float(x_coefficients[0] + x_coefficients[1] * target_time),
                float(y_coefficients[0] + y_coefficients[1] * target_time),
            )
            selected = choose_nearby(frame_index, predicted, used)
            if selected is None:
                missed += 1
                if missed > 1:
                    break
                continue
            missed = 0
            observations[frame_index] = selected
            used.add((frame_index, int(selected[0].detection_id)))
        return tuple(sorted(observations.items()))

    min_step_px = max(
        float(link_radius_px) * 1.01,
        float(motion_min_displacement_px) / max(1, frame_count - 1),
    )
    # ``max_step_px`` and the legacy speed field are expressed per observation
    # interval.  Convert them to px/s only to scale gates for slightly uneven
    # DATE-OBS gaps; with uniform frames this is exactly the old threshold.
    min_speed_px_s = min_step_px / reference_interval_s
    max_speed_px_s = float(max_step_px) / reference_interval_s
    raw_candidates: list[tuple[tuple[int, float, float, float], tuple[TrackPoint, ...]]] = []
    for start_frame in range(frame_count - 1):
        tree = trees[start_frame + 1]
        if tree is None:
            continue
        elapsed_s = float(
            resolved_frame_times[start_frame + 1]
            - resolved_frame_times[start_frame]
        )
        if elapsed_s <= 0.0:
            continue
        max_step_for_pair = max_speed_px_s * elapsed_s
        for first in frame_records[start_frame]:
            for second_index in tree.query_ball_point((first[1], first[2]), max_step_for_pair):
                second = frame_records[start_frame + 1][int(second_index)]
                step = float(np.hypot(second[1] - first[1], second[2] - first[2]))
                if step < min_speed_px_s * elapsed_s or step > max_step_for_pair:
                    continue
                observations = follow_seed(start_frame, first, second)
                if len(observations) < required_presence:
                    continue
                points = tuple(to_point(frame_index, record) for frame_index, record in observations)
                displacement, speed, fit_rms = _fit_track(
                    points,
                    frame_times=resolved_frame_times,
                )
                if fit_rms is None or fit_rms > float(max_fit_rms_px):
                    continue
                if speed < min_step_px or speed > float(max_step_px):
                    continue
                if displacement < float(motion_min_displacement_px):
                    continue
                x_coefficients, y_coefficients = fit_prediction(observations)
                residuals = np.asarray(
                    [
                        np.hypot(
                            point.aligned_x
                            - (
                                x_coefficients[0]
                                + x_coefficients[1] * resolved_frame_times[point.frame_index]
                            ),
                            point.aligned_y
                            - (
                                y_coefficients[0]
                                + y_coefficients[1] * resolved_frame_times[point.frame_index]
                            ),
                        )
                        for point in points
                    ],
                    dtype=np.float64,
                )
                if residuals.size == 0 or float(np.max(residuals)) > float(gate_px):
                    continue
                frame_indices = [item[0] for item in observations]
                if any((right - left) > 2 for left, right in zip(frame_indices, frame_indices[1:])):
                    continue
                median_snr = float(np.median([point.flux_snr for point in points if point.flux_snr is not None]))
                raw_candidates.append(
                    (
                        (len(points), -float(fit_rms), median_snr, -float(speed)),
                        points,
                    )
                )

    if not raw_candidates:
        return ()

    # 同一条目标会由每一对相邻帧重复播种。先按完整性、拟合残差和
    # SNR 排序，再按观测键去重；两个真实目标即便靠近，也不会因为空间
    # 距离小而合并，只要它们使用的是不同逐帧检测源即可。
    raw_candidates.sort(key=lambda item: item[0], reverse=True)
    accepted: list[SourceTrack] = []
    accepted_keys: list[set[tuple[int, int]]] = []
    for _score, points in raw_candidates:
        keys = {(point.frame_index, point.detection_id) for point in points}
        if any(
            len(keys & previous_keys) >= max(2, int(np.ceil(0.5 * min(len(keys), len(previous_keys)))))
            for previous_keys in accepted_keys
        ):
            continue
        displacement, speed, fit_rms = _fit_track(
            points,
            frame_times=resolved_frame_times,
        )
        accepted.append(
            SourceTrack(
                track_id=-1,
                classification="moving",
                points=points,
                displacement_px=float(displacement),
                speed_px_per_frame=float(speed),
                fit_rms_px=float(fit_rms) if fit_rms is not None else None,
                evidence_level=FAST_POINT_MOTION_EVIDENCE,
            )
        )
        accepted_keys.append(keys)
    return tuple(accepted)


def _associate_fast_candidate_records(
    frame_records: Sequence[Sequence[_FastCandidateRecord]],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    frame_times: Sequence[float] | np.ndarray | None = None,
    link_radius_px: float,
    min_presence: int,
    motion_min_displacement_px: float,
    max_step_px: float,
    gate_px: float,
    max_fit_rms_px: float,
) -> tuple[tuple[tuple[int, _FastCandidateRecord], ...], ...]:
    """在候选峰坐标上寻找常速度点源轨迹。

    这是宽筛候选的纯坐标关联层，不读取图像，也不把匹配滤波分数当成
    最终 flux SNR。为了不让每帧数万候选产生大量无意义的后续拟合，先
    要求三个连续候选点满足同一速度方向，再向前/向后跟踪；最终仍由
    ``_track_fast_candidate_movers`` 回到原始 ADU 做孔径、PSF 和值域复核。
    """

    frame_count = len(frame_records)
    if frame_count < 3 or len(cumulative_shifts) != frame_count:
        return ()
    resolved_frame_times = _normalise_motion_times(frame_times, frame_count)
    reference_interval_s = _reference_time_interval_seconds(resolved_frame_times)
    if min_presence < 3 or min_presence > frame_count:
        raise ValueError("min_presence must be within 3..frame count")
    if link_radius_px <= 0 or max_step_px <= link_radius_px or gate_px <= 0 or max_fit_rms_px <= 0:
        raise ValueError("invalid fast candidate association thresholds")
    required_presence = int(min_presence)
    min_step_px = max(
        float(link_radius_px) * 1.01,
        float(motion_min_displacement_px) / max(1, frame_count - 1),
    )
    min_speed_px_s = min_step_px / reference_interval_s
    max_speed_px_s = float(max_step_px) / reference_interval_s

    trees: list[cKDTree | None] = []
    for records in frame_records:
        if records:
            trees.append(
                cKDTree(
                    np.asarray(
                        [(record.aligned_x, record.aligned_y) for record in records],
                        dtype=np.float64,
                    )
                )
            )
        else:
            trees.append(None)
    if not any(frame_records):
        return ()

    def fit_prediction(
        observations: Sequence[tuple[int, _FastCandidateRecord]],
    ) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(
            [resolved_frame_times[item[0]] for item in observations],
            dtype=np.float64,
        )
        coordinates = np.asarray(
            [(item[1].aligned_x, item[1].aligned_y) for item in observations],
            dtype=np.float64,
        )
        design = np.column_stack((np.ones(times.size, dtype=np.float64), times))
        x_coefficients = np.linalg.lstsq(design, coordinates[:, 0], rcond=None)[0]
        y_coefficients = np.linalg.lstsq(design, coordinates[:, 1], rcond=None)[0]
        return x_coefficients, y_coefficients

    def choose_nearby(
        frame_index: int,
        predicted: tuple[float, float],
        used: set[tuple[int, int]],
    ) -> _FastCandidateRecord | None:
        tree = trees[frame_index]
        if tree is None:
            return None
        candidate_indices = tree.query_ball_point(predicted, float(gate_px))
        choices: list[tuple[float, float, int]] = []
        for candidate_index in candidate_indices:
            record = frame_records[frame_index][int(candidate_index)]
            if (frame_index, record.row_index) in used:
                continue
            distance = float(
                np.hypot(
                    record.aligned_x - predicted[0],
                    record.aligned_y - predicted[1],
                )
            )
            choices.append((distance, -record.filter_snr, int(candidate_index)))
        if not choices:
            return None
        choices.sort()
        return frame_records[frame_index][choices[0][2]]

    def follow_seed(
        seed: Sequence[tuple[int, _FastCandidateRecord]],
    ) -> tuple[tuple[int, _FastCandidateRecord], ...]:
        observations = {int(frame_index): record for frame_index, record in seed}
        used = {(int(frame_index), record.row_index) for frame_index, record in seed}
        first_frame = min(observations)
        last_frame = max(observations)

        missed = 0
        for frame_index in range(last_frame + 1, frame_count):
            x_coefficients, y_coefficients = fit_prediction(tuple(sorted(observations.items())))
            target_time = float(resolved_frame_times[frame_index])
            predicted = (
                float(x_coefficients[0] + x_coefficients[1] * target_time),
                float(y_coefficients[0] + y_coefficients[1] * target_time),
            )
            selected = choose_nearby(frame_index, predicted, used)
            if selected is None:
                missed += 1
                if missed > 1:
                    break
                continue
            missed = 0
            observations[frame_index] = selected
            used.add((frame_index, selected.row_index))

        missed = 0
        for frame_index in range(first_frame - 1, -1, -1):
            x_coefficients, y_coefficients = fit_prediction(tuple(sorted(observations.items())))
            target_time = float(resolved_frame_times[frame_index])
            predicted = (
                float(x_coefficients[0] + x_coefficients[1] * target_time),
                float(y_coefficients[0] + y_coefficients[1] * target_time),
            )
            selected = choose_nearby(frame_index, predicted, used)
            if selected is None:
                missed += 1
                if missed > 1:
                    break
                continue
            missed = 0
            observations[frame_index] = selected
            used.add((frame_index, selected.row_index))
        return tuple(sorted(observations.items()))

    def track_metrics(
        observations: Sequence[tuple[int, _FastCandidateRecord]],
    ) -> tuple[float, float, float, float]:
        times = np.asarray(
            [resolved_frame_times[item[0]] for item in observations],
            dtype=np.float64,
        )
        coordinates = np.asarray(
            [(item[1].aligned_x, item[1].aligned_y) for item in observations],
            dtype=np.float64,
        )
        design = np.column_stack((np.ones(times.size, dtype=np.float64), times))
        x_coefficients = np.linalg.lstsq(design, coordinates[:, 0], rcond=None)[0]
        y_coefficients = np.linalg.lstsq(design, coordinates[:, 1], rcond=None)[0]
        fitted = np.column_stack(
            (
                x_coefficients[0] + x_coefficients[1] * times,
                y_coefficients[0] + y_coefficients[1] * times,
            )
        )
        residuals = np.linalg.norm(coordinates - fitted, axis=1)
        duration = float(times[-1] - times[0])
        speed_per_second = float(np.hypot(x_coefficients[1], y_coefficients[1]))
        speed = speed_per_second * reference_interval_s
        displacement = float(np.hypot(x_coefficients[1] * duration, y_coefficients[1] * duration))
        fit_rms = float(np.sqrt(np.mean(residuals**2)))
        max_residual = float(np.max(residuals)) if residuals.size else np.inf
        return displacement, speed, fit_rms, max_residual

    # 三点种子比逐个二点种子少一个数量级的后续预测调用。候选峰本身
    # 是整像素坐标，第三点允许在 gate 内有少量局部测量抖动。
    raw_candidates: list[tuple[tuple[int, float, float, float], tuple[tuple[int, _FastCandidateRecord], ...]]] = []
    for start_frame in range(frame_count - 2):
        first_records = frame_records[start_frame]
        second_tree = trees[start_frame + 1]
        third_tree = trees[start_frame + 2]
        if not first_records or second_tree is None or third_tree is None:
            continue
        first_points = np.asarray(
            [(record.aligned_x, record.aligned_y) for record in first_records],
            dtype=np.float64,
        )
        first_elapsed_s = float(
            resolved_frame_times[start_frame + 1]
            - resolved_frame_times[start_frame]
        )
        second_elapsed_s = float(
            resolved_frame_times[start_frame + 2]
            - resolved_frame_times[start_frame + 1]
        )
        if first_elapsed_s <= 0.0 or second_elapsed_s <= 0.0:
            continue
        max_first_step = max_speed_px_s * first_elapsed_s
        min_first_step = min_speed_px_s * first_elapsed_s
        max_second_step = max_speed_px_s * second_elapsed_s
        min_second_step = min_speed_px_s * second_elapsed_s
        second_neighbors = second_tree.query_ball_point(first_points, max_first_step)
        for first_index, second_indices in enumerate(second_neighbors):
            first = first_records[first_index]
            for second_index in second_indices:
                second = frame_records[start_frame + 1][int(second_index)]
                step_x = second.aligned_x - first.aligned_x
                step_y = second.aligned_y - first.aligned_y
                step = float(np.hypot(step_x, step_y))
                if step < min_first_step or step > max_first_step:
                    continue
                velocity_x = step_x / first_elapsed_s
                velocity_y = step_y / first_elapsed_s
                predicted_third = (
                    second.aligned_x + velocity_x * second_elapsed_s,
                    second.aligned_y + velocity_y * second_elapsed_s,
                )
                third_indices = third_tree.query_ball_point(predicted_third, float(gate_px))
                third_choices: list[tuple[float, float, int]] = []
                for third_index in third_indices:
                    third = frame_records[start_frame + 2][int(third_index)]
                    step_two = float(
                        np.hypot(
                            third.aligned_x - second.aligned_x,
                            third.aligned_y - second.aligned_y,
                        )
                    )
                    if step_two < 0.55 * min_second_step or step_two > max_second_step:
                        continue
                    distance = float(
                        np.hypot(
                            third.aligned_x - predicted_third[0],
                            third.aligned_y - predicted_third[1],
                        )
                    )
                    third_choices.append((distance, -third.filter_snr, int(third_index)))
                if not third_choices:
                    continue
                third_choices.sort()
                third = frame_records[start_frame + 2][third_choices[0][2]]
                observations = follow_seed(
                    (
                        (start_frame, first),
                        (start_frame + 1, second),
                        (start_frame + 2, third),
                    )
                )
                if len(observations) < required_presence:
                    continue
                frame_indices = [item[0] for item in observations]
                if any((right - left) > 2 for left, right in zip(frame_indices, frame_indices[1:])):
                    continue
                displacement, speed, fit_rms, max_residual = track_metrics(observations)
                if speed < min_step_px or speed > float(max_step_px):
                    continue
                if displacement < float(motion_min_displacement_px):
                    continue
                if fit_rms > float(max_fit_rms_px) or max_residual > float(gate_px):
                    continue
                median_snr = float(np.median([item[1].filter_snr for item in observations]))
                raw_candidates.append(
                    (
                        (len(observations), -fit_rms, median_snr, -speed),
                        observations,
                    )
                )

    if not raw_candidates:
        return ()
    raw_candidates.sort(key=lambda item: item[0], reverse=True)
    accepted: list[tuple[tuple[int, _FastCandidateRecord], ...]] = []
    accepted_keys: list[set[tuple[int, int]]] = []
    for _score, observations in raw_candidates:
        keys = {(frame_index, record.row_index) for frame_index, record in observations}
        if any(
            len(keys & previous_keys) >= max(2, int(np.ceil(0.5 * min(len(keys), len(previous_keys)))))
            for previous_keys in accepted_keys
        ):
            continue
        accepted.append(observations)
        accepted_keys.append(keys)
    return tuple(accepted)


def _track_fast_candidate_movers(
    candidate_peak_frames: Sequence[object],
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    frame_times: Sequence[float] | np.ndarray | None = None,
    link_radius_px: float,
    min_presence: int,
    motion_min_displacement_px: float,
    min_filter_snr: float,
    max_step_px: float,
    gate_px: float,
    max_fit_rms_px: float,
    candidate_limit: int = DEFAULT_FAST_POINT_CANDIDATE_WORKING_LIMIT,
    existing_tracks: Sequence[SourceTrack] = (),
    aperture_radius: int = 4,
    min_psf_support_pixels: int = 3,
    min_fwhm: float = 0.8,
    max_fwhm: float = 12.0,
    max_ellipticity: float = 0.65,
    line_artifact_frames: Sequence[object] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[SourceTrack, ...]:
    """从完整匹配滤波候选中恢复被源级工作集截断的高速点源。

    ``DetectionResult.candidate_peaks`` 保留了 ``max_sources`` 截断前的
    峰坐标。本函数只把其中响应较强的一层作为高速轨迹宽筛，不对这层
    的每个峰做昂贵测光；只有满足三点常速度、至少约 80% 帧出现、拟合
    残差和线状掩膜规则的轨迹，才批量回到每帧原始 FITS 做强制孔径测光。
    最终输出的 ``flux_snr`` 来自原图，``candidate_snr`` 才是匹配滤波
    宽筛分数，因此不会把候选峰分数误报成星点 SNR。
    """

    frame_count = len(candidate_peak_frames)
    if frame_count < 3 or len(frame_analyses) != frame_count or len(cumulative_shifts) != frame_count:
        return ()
    if min_filter_snr <= 0 or candidate_limit < 1:
        raise ValueError("candidate working limit and filter SNR must be positive")
    resolved_frame_times = _normalise_motion_times(frame_times, frame_count)

    # 已由普通质量轨迹解释的位置不参加高速候选关联；否则固定星的
    # 强匹配滤波峰会成为数量巨大的零速度/伪速度种子。
    explained_trees: list[cKDTree | None] = []
    for frame_index in range(frame_count):
        points = [
            (point.aligned_x, point.aligned_y)
            for track in existing_tracks
            if track.presence >= 2
            for point in track.points
            if point.frame_index == frame_index
        ]
        explained_trees.append(cKDTree(np.asarray(points, dtype=np.float64)) if points else None)

    line_trees: list[cKDTree | None] = []
    for frame_index in range(frame_count):
        if line_artifact_frames is None or frame_index >= len(line_artifact_frames):
            line_trees.append(None)
            continue
        values = np.asarray(line_artifact_frames[frame_index])
        if values.ndim != 2 or values.shape[1] < 2 or values.size == 0:
            line_trees.append(None)
            continue
        line_trees.append(cKDTree(np.asarray(values[:, :2], dtype=np.float64)))

    frame_records: list[list[_FastCandidateRecord]] = []
    for frame_index, raw_value in enumerate(candidate_peak_frames):
        raw = np.asarray(raw_value, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 3 or raw.size == 0:
            frame_records.append([])
            if progress is not None:
                progress(frame_index + 1, frame_count, "候选峰筛选")
            continue
        finite = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1]) & np.isfinite(raw[:, 2])
        keep = finite & (raw[:, 2] >= float(min_filter_snr))
        row_indices = np.flatnonzero(keep)
        if row_indices.size == 0:
            frame_records.append([])
            if progress is not None:
                progress(frame_index + 1, frame_count, "候选峰筛选")
            continue
        if row_indices.size > int(candidate_limit):
            scores = raw[row_indices, 2]
            chosen = np.argpartition(scores, -int(candidate_limit))[-int(candidate_limit):]
            row_indices = row_indices[chosen]
        # 对每个帧只保留前一层响应候选；排序让相同距离时的优先级
        # 稳定，也便于审计“候选工作集”到底覆盖到哪一个响应分数。
        row_indices = row_indices[np.argsort(raw[row_indices, 2])[::-1]]
        shift_x, shift_y = cumulative_shifts[frame_index]
        image_height, image_width = frame_analyses[frame_index].frame.data.shape
        safe_margin = int(aperture_radius) + 4
        records: list[_FastCandidateRecord] = []
        explained_tree = explained_trees[frame_index]
        line_tree = line_trees[frame_index]
        # 这里原来逐候选调用两次 cKDTree.query。每帧最多 20,000 个峰、
        # 15 帧时会产生几十万次 Python/Scipy 往返，运动宽筛因此成为主要
        # 等待来源。查询本身是逐点独立的，改成批量最近邻只改变执行方式，
        # 保留原来的边界、已解释源和线状伪影门槛；布尔索引保持响应排序。
        candidate_rows = row_indices.astype(np.intp, copy=False)
        raw_coordinates = raw[candidate_rows, :2]
        inside = (
            (raw_coordinates[:, 0] >= float(safe_margin))
            & (raw_coordinates[:, 0] < float(image_width - safe_margin))
            & (raw_coordinates[:, 1] >= float(safe_margin))
            & (raw_coordinates[:, 1] < float(image_height - safe_margin))
        )
        candidate_rows = candidate_rows[inside]
        raw_coordinates = raw_coordinates[inside]
        if candidate_rows.size:
            aligned_coordinates = raw_coordinates - np.asarray(
                (float(shift_x), float(shift_y)),
                dtype=np.float64,
            )
            keep = np.ones(candidate_rows.size, dtype=bool)
            if explained_tree is not None:
                explained_distances = explained_tree.query(aligned_coordinates, k=1)[0]
                keep &= explained_distances > float(link_radius_px)
            if line_tree is not None:
                artifact_distances = line_tree.query(raw_coordinates, k=1)[0]
                keep &= artifact_distances > float(aperture_radius) + 0.5
            candidate_rows = candidate_rows[keep]
            raw_coordinates = raw_coordinates[keep]
        for row_index, (raw_x, raw_y) in zip(candidate_rows, raw_coordinates, strict=True):
            raw_x = float(raw_x)
            raw_y = float(raw_y)
            filter_snr = float(raw[int(row_index), 2])
            aligned_x = raw_x - float(shift_x)
            aligned_y = raw_y - float(shift_y)
            records.append(
                _FastCandidateRecord(
                    row_index=int(row_index),
                    raw_x=raw_x,
                    raw_y=raw_y,
                    aligned_x=aligned_x,
                    aligned_y=aligned_y,
                    filter_snr=filter_snr,
                )
            )
        frame_records.append(records)
        if progress is not None:
            progress(frame_index + 1, frame_count, "候选峰筛选")

    associated = _associate_fast_candidate_records(
        frame_records,
        cumulative_shifts,
        frame_times=resolved_frame_times,
        link_radius_px=link_radius_px,
        min_presence=min_presence,
        motion_min_displacement_px=motion_min_displacement_px,
        max_step_px=max_step_px,
        gate_px=gate_px,
        max_fit_rms_px=max_fit_rms_px,
    )
    if not associated:
        return ()

    # 轨迹候选的数量通常只有个位数/几十条；每个帧只对这些坐标做一次
    # 向量化强制测光，并按帧释放有效掩膜，避免复制 15 张 4096² bool 图。
    unique_by_frame: list[dict[int, _FastCandidateRecord]] = [dict() for _ in range(frame_count)]
    for observations in associated:
        for frame_index, record in observations:
            unique_by_frame[frame_index][record.row_index] = record
    measured: list[dict[int, tuple[float, int, bool, float, float]]] = [dict() for _ in range(frame_count)]
    for frame_index, records_by_id in enumerate(unique_by_frame):
        if not records_by_id:
            if progress is not None:
                progress(frame_index + 1, frame_count, "原图强制测光")
            continue
        analysis = frame_analyses[frame_index]
        image = np.asarray(analysis.frame.data)
        parameters = analysis.detection.parameters
        saturation_value = parameters.get("saturation_level")
        saturation_level = None
        if saturation_value is not None:
            try:
                candidate_saturation = float(saturation_value)
            except (TypeError, ValueError):
                candidate_saturation = -1.0
            if candidate_saturation > 0:
                saturation_level = candidate_saturation
        effective_mask, _inferred_saturation, _mask_zero_pixels = _working_mask(
            image,
            auxiliary_mask(image.shape),
            background=float(analysis.detection.background),
            mask_zero_pixels=(
                bool(parameters.get("mask_zero_pixels"))
                if parameters.get("mask_zero_pixels") is not None
                else None
            ),
            saturation_level=saturation_level,
            numeric=np.asarray(image, dtype=np.float32),
        )
        records = tuple(records_by_id.values())
        xs = np.asarray([record.raw_x for record in records], dtype=np.float64)
        ys = np.asarray([record.raw_y for record in records], dtype=np.float64)
        flux_snr, support, center_valid, fwhm, ellipticity = _stack_forced_frame_measure_batch(
            image,
            effective_mask,
            xs,
            ys,
            aperture_radius=int(aperture_radius),
            min_psf_support_pixels=int(min_psf_support_pixels),
            global_background=float(analysis.detection.background),
            global_noise=float(analysis.detection.noise),
        )
        for index, record in enumerate(records):
            measured[frame_index][record.row_index] = (
                float(flux_snr[index]),
                int(support[index]),
                bool(center_valid[index]),
                float(fwhm[index]),
                float(ellipticity[index]),
            )
        if progress is not None:
            progress(frame_index + 1, frame_count, "原图强制测光")

    rendered: list[SourceTrack] = []
    rendered_keys: list[set[tuple[int, int]]] = []
    for observations in associated:
        points: list[TrackPoint] = []
        for frame_index, record in observations:
            value = measured[frame_index].get(record.row_index)
            if value is None:
                continue
            flux_snr, support, center_valid, fwhm, ellipticity = value
            if not center_valid or support < int(min_psf_support_pixels):
                continue
            if not np.isfinite(flux_snr) or flux_snr < float(min_filter_snr):
                continue
            if not np.isfinite(fwhm) or not float(min_fwhm) <= fwhm <= float(max_fwhm):
                continue
            if not np.isfinite(ellipticity) or ellipticity > float(max_ellipticity):
                continue
            shift_x, shift_y = cumulative_shifts[frame_index]
            points.append(
                TrackPoint(
                    frame_index=int(frame_index),
                    detection_id=-1 - int(record.row_index),
                    x=float(record.raw_x),
                    y=float(record.raw_y),
                    aligned_x=float(record.aligned_x),
                    aligned_y=float(record.aligned_y),
                    flux_snr=float(flux_snr),
                    quality_passed=True,
                    candidate_snr=float(record.filter_snr),
                )
            )
        if len(points) < int(min_presence):
            continue
        frame_indices = [point.frame_index for point in points]
        if any((right - left) > 2 for left, right in zip(frame_indices, frame_indices[1:])):
            continue
        displacement, speed, fit_rms = _fit_track(
            points,
            frame_times=resolved_frame_times,
        )
        if fit_rms is None or fit_rms > float(max_fit_rms_px):
            continue
        if speed < max(float(link_radius_px) * 1.01, float(motion_min_displacement_px) / max(1, frame_count - 1)):
            continue
        if speed > float(max_step_px) or displacement < float(motion_min_displacement_px):
            continue
        keys = {(point.frame_index, point.detection_id) for point in points}
        if any(
            len(keys & previous_keys) >= max(2, int(np.ceil(0.5 * min(len(keys), len(previous_keys)))))
            for previous_keys in rendered_keys
        ):
            continue
        rendered.append(
            SourceTrack(
                track_id=-1,
                classification="moving",
                points=tuple(points),
                displacement_px=float(displacement),
                speed_px_per_frame=float(speed),
                fit_rms_px=float(fit_rms),
                evidence_level=FAST_POINT_MOTION_EVIDENCE,
            )
        )
        rendered_keys.append(keys)
    return tuple(rendered)


def _parse_observation_timestamp(value: object) -> datetime | None:
    """Parse a FITS ``DATE-OBS`` value without making local-time assumptions."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        # FITS DATE-OBS is conventionally UTC in this data set.  Attaching UTC
        # also lets a mixed header set fail safely instead of using machine
        # local time during a scientific fit.
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _observation_times_seconds(
    frame_analyses: Sequence[FrameAnalysis],
) -> np.ndarray:
    """Return relative observation times, or frame numbers when headers fail.

    A sequence may contain old synthetic fixtures without ``DATE-OBS``.  The
    fallback is deliberately explicit and deterministic; real FITS with a
    complete, strictly increasing timestamp sequence use elapsed seconds.
    """

    fallback = np.arange(len(frame_analyses), dtype=np.float64)
    timestamps = [
        _parse_observation_timestamp(analysis.frame.header.get("DATE-OBS"))
        for analysis in frame_analyses
    ]
    if any(timestamp is None for timestamp in timestamps):
        return fallback
    first = timestamps[0]
    assert first is not None
    values = np.asarray(
        [(timestamp - first).total_seconds() for timestamp in timestamps if timestamp is not None],
        dtype=np.float64,
    )
    if values.shape != fallback.shape or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        return fallback
    return values


def _normalise_motion_times(
    frame_times: Sequence[float] | np.ndarray | None,
    frame_count: int,
) -> np.ndarray:
    """Validate caller-provided relative times and fall back to frame index."""

    fallback = np.arange(int(frame_count), dtype=np.float64)
    if frame_times is None:
        return fallback
    try:
        values = np.asarray(frame_times, dtype=np.float64)
    except (TypeError, ValueError):
        return fallback
    if (
        values.shape != fallback.shape
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        return fallback
    return values - values[0]


def _sequence_motion_candidate_min_length(
    *,
    psf_fwhm: float,
    min_length_px: float | None,
    exposure_s: Sequence[float | None],
    metadata_supported: bool,
) -> float:
    """Choose the geometry-only length floor for sequence line proposals.

    A 1.5 s exposure can produce a real trail whose thresholded support is
    only a few PSF widths long.  The old fixed floor (``max(8, 3*FWHM)``)
    could discard that proposal before the raw-frame line-vs-double model
    ever saw it.  This helper lowers *only the proposal floor* when every
    frame provides usable time/exposure metadata and the exposure is the
    known RST19 1.5 s regime.  Explicit caller thresholds remain authoritative;
    all later shape, model, temporal, velocity and artifact gates are
    unchanged.
    """

    if min_length_px is not None:
        return float(min_length_px)
    base = max(DEFAULT_SEQUENCE_TRAIL_MIN_LENGTH, 3.0 * float(psf_fwhm))
    if not metadata_supported or psf_fwhm <= 0.0:
        return base
    declared = tuple(exposure_s)
    values = np.asarray(
        [float(value) for value in declared if value is not None],
        dtype=np.float64,
    )
    if len(values) != len(declared) or values.size == 0:
        return base
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or float(np.ptp(values)) > max(0.05, 0.10 * float(np.median(values)))
    ):
        return base
    median_exposure = float(np.median(values))
    if not 1.35 <= median_exposure <= 1.65:
        return base
    # Keep a floor of 6 px so isolated PSF-sized fluctuations do not enter
    # the proposal pool.  This is deliberately above a single-pixel/noise
    # structure and is not a moving-confirmation threshold.
    return min(base, max(6.0, 2.5 * float(psf_fwhm)))


def _exposure_aware_short_trail_min_area(
    *,
    psf_fwhm: float,
    exposure_s: float | None,
) -> int:
    """Return the short-line proposal area floor for a known exposure.

    The historical 20-pixel floor is a useful noise guard for an unknown
    exposure, but it can discard a real 1.5 s trail before the line-vs-point
    model sees it: in the supplied data several such trails retain only
    roughly 15--17 pixels above the residual threshold.  This is deliberately
    a *proposal* floor, not a moving-object acceptance threshold.  The lower
    floor is enabled only for the known RST19 1.5 s regime; all raw-pixel
    shape, point/double negative-control, temporal and artifact gates remain
    unchanged downstream.
    """

    try:
        fwhm = float(psf_fwhm)
        exposure = float(exposure_s) if exposure_s is not None else math.nan
    except (TypeError, ValueError):
        return int(MOTION_FEATURE_SHORT_TRAIL_MIN_AREA)
    if (
        not np.isfinite(fwhm)
        or fwhm <= 0.0
        or not np.isfinite(exposure)
        or not 1.35 <= exposure <= 1.65
    ):
        return int(MOTION_FEATURE_SHORT_TRAIL_MIN_AREA)

    # A short finite trail still needs several PSF footprints.  The 1.5
    # factor is a conservative connected-support estimate; the absolute
    # floor prevents a 1--2 pixel fluctuation from entering the proposal pool.
    exposure_floor = max(12, int(np.ceil(1.5 * fwhm * fwhm)))
    return int(min(MOTION_FEATURE_SHORT_TRAIL_MIN_AREA, exposure_floor))


def _reference_time_interval_seconds(frame_times: np.ndarray) -> float:
    """Return the median valid frame interval used by legacy per-frame fields."""

    differences = np.diff(np.asarray(frame_times, dtype=np.float64))
    valid = differences[np.isfinite(differences) & (differences > 0.0)]
    return float(np.median(valid)) if valid.size else 1.0


def _deconvolved_exposure_motion_length(length_px: float, psf_fwhm: float) -> float | None:
    """Estimate the motion-only part of a PCA line length.

    ``MotionFeaturePoint.length_px`` is defined as ``4*sqrt(var_major)``.
    For a uniform finite trail convolved with a circular Gaussian PSF,
    ``var_major ~= L_motion**2 / 12 + sigma_psf**2``.  The returned estimate
    is therefore only a scale correction, not a claim that thresholded pixels
    are an exact physical endpoint measurement.  Keeping this correction in a
    named helper makes the approximation explicit and testable.
    """

    try:
        measured = float(length_px)
        fwhm = float(psf_fwhm)
    except (TypeError, ValueError):
        return None
    if (
        not np.isfinite(measured)
        or not np.isfinite(fwhm)
        or measured <= 0.0
        or fwhm <= 0.0
    ):
        return None
    psf_sigma = fwhm / 2.3548200450309493
    motion_variance = 0.75 * (measured * measured - 16.0 * psf_sigma * psf_sigma)
    if not np.isfinite(motion_variance) or motion_variance <= 0.0:
        return None
    return float(np.sqrt(motion_variance))


def _motion_feature_exposure_metrics(
    points: Sequence[MotionFeaturePoint],
    frame_analyses: Sequence[FrameAnalysis],
    frame_times: Sequence[float] | np.ndarray | None,
    *,
    psf_fwhm: float,
) -> tuple[float | None, float | None, float | None]:
    """Measure consistency between exposure-line and inter-frame speeds.

    For a constant-velocity source, a line formed during one exposure and the
    displacement between two exposure centres should obey

    ``(L_motion / T_exp) / (D / dt) ~= 1``.

    The calculation is intentionally *diagnostic first*.  It only returns a
    value when every frame declares both ``DATE-OBS`` and an exposure value;
    missing metadata must not be silently replaced by a guessed one-second
    exposure.  Edge-truncated lines are omitted because their measured length
    is a lower bound.  The tuple is ``(median_ratio, log_MAD,
    in_range_fraction)``; ``None`` means that the dataset cannot support this
    physical check.
    """

    if len(points) < 2 or len(frame_analyses) < 2 or psf_fwhm <= 0.0:
        return None, None, None
    declared_exposures: list[float] = []
    for analysis in frame_analyses:
        if analysis.frame.header.get("DATE-OBS") is None:
            return None, None, None
        exposure_ms = exposure_milliseconds(analysis.frame.header)
        if exposure_ms is None or not np.isfinite(float(exposure_ms)) or float(exposure_ms) <= 0.0:
            return None, None, None
        declared_exposures.append(float(exposure_ms) / 1000.0)

    resolved_times = _normalise_motion_times(frame_times, len(frame_analyses))
    ordered = tuple(sorted(points, key=lambda point: int(point.frame_index)))
    ratios: list[float] = []
    minimum_step = max(2.0, float(psf_fwhm))
    for previous, current in zip(ordered, ordered[1:]):
        previous_index = int(previous.frame_index)
        current_index = int(current.frame_index)
        if (
            previous_index < 0
            or current_index <= previous_index
            or current_index >= len(frame_analyses)
            or previous_index >= len(frame_analyses)
        ):
            continue
        if current.touches_edge:
            continue
        elapsed = float(resolved_times[current_index] - resolved_times[previous_index])
        distance = float(
            np.hypot(
                float(current.aligned_x) - float(previous.aligned_x),
                float(current.aligned_y) - float(previous.aligned_y),
            )
        )
        if not np.isfinite(elapsed) or elapsed <= 0.0 or not np.isfinite(distance) or distance < minimum_step:
            continue
        motion_length = _deconvolved_exposure_motion_length(current.length_px, psf_fwhm)
        if motion_length is None:
            continue
        ratio = (
            motion_length
            * elapsed
            / max(distance * declared_exposures[current_index], np.finfo(np.float64).eps)
        )
        if np.isfinite(ratio) and ratio > 0.0:
            ratios.append(float(ratio))

    if len(ratios) < MOTION_FEATURE_EXPOSURE_MIN_TRANSITIONS:
        return None, None, None
    log_ratios = np.log(np.asarray(ratios, dtype=np.float64))
    median_log = float(np.median(log_ratios))
    log_mad = float(np.median(np.abs(log_ratios - median_log)))
    median_ratio = float(np.exp(median_log))
    in_range = (
        (np.asarray(ratios, dtype=np.float64) >= MOTION_FEATURE_EXPOSURE_RATIO_MIN)
        & (np.asarray(ratios, dtype=np.float64) <= MOTION_FEATURE_EXPOSURE_RATIO_MAX)
    )
    return median_ratio, log_mad, float(np.mean(in_range))


def _motion_feature_exposure_is_supported(
    ratio: float | None,
    log_mad: float | None,
    consistency_fraction: float | None,
) -> bool:
    """Return whether available exposure metadata supports a moving label."""

    # Unknown FITS metadata is an unknown, not a negative observation.  The
    # regular geometric/temporal gates remain responsible in that case.
    if ratio is None or log_mad is None or consistency_fraction is None:
        return True
    return bool(
        MOTION_FEATURE_EXPOSURE_RATIO_MIN <= float(ratio) <= MOTION_FEATURE_EXPOSURE_RATIO_MAX
        and float(log_mad) <= MOTION_FEATURE_EXPOSURE_MAX_LOG_MAD
        and float(consistency_fraction) >= MOTION_FEATURE_EXPOSURE_MIN_PASS_FRACTION
    )


def _motion_feature_confirmation_presence(frame_count: int) -> int:
    """Return the frame coverage needed to call a line track ``moving``.

    A 15-frame observation cannot treat a 3-frame straight coincidence as a
    confirmed moving object.  The underlying geometry and shape candidates
    remain available; this helper only separates short temporal evidence from
    the stronger ``moving`` label.  Small synthetic sequences keep the legacy
    three-point minimum so the detector remains useful outside the contest's
    15-frame batch.
    """

    count = int(frame_count)
    if count <= 0:
        raise ValueError("frame_count must be positive")
    if count < MOTION_FEATURE_LONG_SEQUENCE_MIN_FRAMES:
        return min(3, count)
    return max(
        3,
        int(np.ceil(float(count) * MOTION_FEATURE_CONFIRMATION_PRESENCE_FRACTION)),
    )


def _motion_feature_rejection_reasons(
    *,
    classification: str,
    evidence_level: str,
    points: Sequence[MotionFeaturePoint],
    presence_supported: bool | None = None,
    displacement_px: float = 0.0,
    min_displacement_px: float | None = None,
    fit_rms_px: float | None = None,
    max_fit_rms_px: float | None = None,
    shape_supported: bool | None = None,
    model_supported: bool | None = None,
    motion_axis_supported: bool | None = None,
    temporal_continuity_supported: bool | None = None,
    exposure_supported: bool | None = None,
    artifact_supported: bool | None = None,
    edge_truncated: bool = False,
) -> tuple[str, ...]:
    """Return stable reasons why a line track was not promoted to ``moving``.

    The detector deliberately keeps weak evidence as ``candidate``.  A single
    classification is not enough for the UI or an audit report to explain the
    result, so this helper records every failed final gate without changing any
    gate itself.  Codes are intentionally language-neutral and stable enough
    for JSON consumers; the GUI maps them to Chinese labels.
    """

    if classification != "candidate":
        return ()
    if evidence_level == "single_frame_trail":
        reasons = ["SINGLE_FRAME_ONLY", "NO_CROSS_FRAME_ASSOCIATION"]
        if not points:
            reasons.append("NO_GEOMETRIC_POINT")
        return tuple(reasons)

    reasons: list[str] = []
    if len(points) < 2:
        reasons.extend(("NO_CROSS_FRAME_ASSOCIATION", "INSUFFICIENT_FRAME_PRESENCE"))
    elif presence_supported is False:
        reasons.append(
            "EDGE_TRUNCATION_INSUFFICIENT_COVERAGE"
            if edge_truncated
            else "INSUFFICIENT_FRAME_PRESENCE"
        )
    if min_displacement_px is not None and (
        not np.isfinite(float(displacement_px))
        or float(displacement_px) < float(min_displacement_px)
    ):
        reasons.append("INSUFFICIENT_DISPLACEMENT")
    if fit_rms_px is None or not np.isfinite(float(fit_rms_px)):
        reasons.append("NO_MOTION_FIT")
    elif max_fit_rms_px is not None and float(fit_rms_px) > float(max_fit_rms_px):
        reasons.append("MOTION_FIT_RMS_TOO_HIGH")
    if shape_supported is False:
        reasons.append("FRAME_SHAPE_INCONSISTENT")
    if model_supported is False:
        models = {point.single_frame_model for point in points}
        if "double" in models:
            reasons.append("COMPACT_DOUBLE_MODEL_PRESENT")
        if "point" in models:
            reasons.append("COMPACT_POINT_MODEL_PRESENT")
        if "elliptical_point" in models:
            reasons.append("COMPACT_ELLIPTICAL_MODEL_PRESENT")
        reasons.append("RAW_LINE_MODEL_MAJORITY_NOT_MET")
    if motion_axis_supported is False:
        reasons.append("MOTION_AXIS_MISMATCH")
    if temporal_continuity_supported is False:
        reasons.append("FRAME_GAP_OR_NONCONTIGUOUS")
    if exposure_supported is False:
        reasons.append("EXPOSURE_SPEED_INCONSISTENT")
    if artifact_supported is False:
        reasons.append("LINE_ARTIFACT_OVERLAP")
    if not reasons:
        reasons.append("CONFIRMATION_GATES_NOT_SATISFIED")
    return tuple(dict.fromkeys(reasons))


def _motion_feature_temporal_continuity_is_supported(
    points: Sequence[MotionFeaturePoint],
    *,
    min_presence: int,
    edge_truncated: bool = False,
) -> bool:
    """Require the confirmed line track to contain one sufficiently long run.

    A total of eight detections can otherwise be assembled from two unrelated
    four-frame streaks separated by a gap.  The detections remain in the
    candidate/audit pool; this gate only prevents that discontinuous evidence
    from being labelled ``moving``.  An edge-entering/leaving track keeps the
    existing one-frame relaxation because the unobserved part is outside the
    frame, not a missing interior detection.
    """

    if not points:
        return False
    required = int(min_presence)
    if required <= 0:
        raise ValueError("min_presence must be positive")
    if edge_truncated:
        required = max(3, required - MOTION_FEATURE_EDGE_MIN_PRESENCE_RELAXATION)
    frame_indices = sorted({int(point.frame_index) for point in points})
    longest_run = 1
    current_run = 1
    for previous, current in zip(frame_indices, frame_indices[1:]):
        if current == previous + 1:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
    return longest_run >= required


def _motion_feature_internal_gap_bridge_is_supported(
    points: Sequence[MotionFeaturePoint],
    *,
    max_gap_frames: int = MOTION_FEATURE_INTERNAL_GAP_BRIDGE_MAX_FRAMES,
    min_run_length: int = MOTION_FEATURE_INTERNAL_GAP_BRIDGE_MIN_RUN,
) -> bool:
    """判断轨迹是否适合进入严格的内部缺帧桥接分支。

    这个函数只判断时间结构，不单独授予 ``moving``。调用方还必须检查
    原图线模型、线形、运动轴、常速度拟合、曝光内速度和伪影审计。把
    “F05--F08、F11--F14”这种短内部缺口与两个无关的 singleton/短线
    分开，避免为了召回率把所有不连续轨迹都合并。
    """

    if max_gap_frames <= 0 or min_run_length < 2 or not points:
        return False
    frame_indices = sorted({int(point.frame_index) for point in points})
    runs: list[list[int]] = [[frame_indices[0]]]
    for frame_index in frame_indices[1:]:
        if frame_index == runs[-1][-1] + 1:
            runs[-1].append(frame_index)
        else:
            runs.append([frame_index])
    strong_runs = [run for run in runs if len(run) >= int(min_run_length)]
    if len(strong_runs) != 2:
        return False
    left_run, right_run = strong_runs
    gap = right_run[0] - left_run[-1] - 1
    if not 0 < gap <= int(max_gap_frames):
        return False

    # Only a small number of isolated points may sit outside the two strong
    # runs.  They are allowed at the ends when they are close enough to a
    # strong run to have been reached by the bounded prediction stitch.  A
    # third run of length >= ``min_run_length`` remains a hard rejection.
    isolated_runs = [run for run in runs if len(run) < int(min_run_length)]
    isolated_count = sum(len(run) for run in isolated_runs)
    if isolated_count > MOTION_FEATURE_INTERNAL_GAP_BRIDGE_MAX_ISOLATED_POINTS:
        return False
    for run in isolated_runs:
        if run[-1] < left_run[0]:
            distance = left_run[0] - run[-1] - 1
        elif run[0] > right_run[-1]:
            distance = run[0] - right_run[-1] - 1
        else:
            # An isolated run inside the two strong runs would mean the run
            # decomposition itself is not the intended two-block pattern.
            return False
        if distance > int(max_gap_frames) + 1:
            return False
    return True


def _point_motion_times(
    points: Sequence[MotionFeaturePoint],
    frame_times: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    """Resolve one time value per line-feature point."""

    if not points:
        return np.empty(0, dtype=np.float64)
    max_frame = max(int(point.frame_index) for point in points)
    resolved = _normalise_motion_times(frame_times, max_frame + 1)
    selected = np.asarray([resolved[int(point.frame_index)] for point in points], dtype=np.float64)
    if not np.all(np.isfinite(selected)) or np.any(np.diff(selected) <= 0.0):
        return np.asarray([point.frame_index for point in points], dtype=np.float64)
    return selected - selected[0]


def _fit_motion_feature_track(
    points: Sequence[MotionFeaturePoint],
    *,
    frame_times: Sequence[float] | np.ndarray | None = None,
    reference_interval_s: float | None = None,
) -> tuple[float, float, float | None]:
    """在线状候选坐标中按实际观测时间拟合常速度轨迹。

    The public/serialized field remains ``speed_px_per_frame`` for backward
    compatibility.  When timestamps are available, the fitted px/s is
    converted back using the median frame interval for that legacy field;
    displacement and RMS use the real elapsed seconds.
    """

    if len(points) < 2:
        return 0.0, 0.0, None
    # 轨迹通常按帧顺序生成，但短线回接、几何补救或外部调用可能把
    # 较早帧追加到已有历史之后。拟合的时间轴必须和坐标使用同一顺序；
    # 否则 ``time[-1] - time[0]`` 会变成负值，位移虽然经过 hypot 后
    # 仍然是正数，速度/跨度却会悄悄使用错误的首尾帧。
    ordered = tuple(sorted(points, key=lambda point: int(point.frame_index)))
    time = _point_motion_times(ordered, frame_times)
    x = np.array([point.aligned_x for point in ordered], dtype=np.float64)
    y = np.array([point.aligned_y for point in ordered], dtype=np.float64)
    x_slope, x_intercept = np.polyfit(time, x, 1)
    y_slope, y_intercept = np.polyfit(time, y, 1)
    fitted_x = x_slope * time + x_intercept
    fitted_y = y_slope * time + y_intercept
    rms = float(np.sqrt(np.mean((x - fitted_x) ** 2 + (y - fitted_y) ** 2)))
    # 与点源轨迹保持同一语义：用拟合速度乘以观测跨度，避免首末帧
    # 的单次质心/连通域形状抖动改变运动判定。
    duration = float(time[-1] - time[0])
    displacement = float(np.hypot(x_slope * duration, y_slope * duration))
    speed_per_second = float(np.hypot(x_slope, y_slope))
    interval = (
        float(reference_interval_s)
        if reference_interval_s is not None and np.isfinite(float(reference_interval_s)) and float(reference_interval_s) > 0.0
        else 1.0
    )
    speed_per_frame = speed_per_second * interval
    return displacement, speed_per_frame, rms


def _angle_difference_mod_180(first: float, second: float) -> float:
    """返回两条无方向直线的最小夹角（degree）。"""

    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _motion_feature_pair_geometry_penalty(
    previous: MotionFeaturePoint,
    current: MotionFeaturePoint,
    *,
    psf_fwhm: float,
    expected_step_px: float | None = None,
) -> float:
    """计算一对线状候选的软几何代价。

    线段的方向应与曝光期间的运动方向大致一致；在曝光时间、姿态和
    配准坐标一致时，上一帧线段平移到当前质心后，两个端点也应大致重合。
    如果头部提供了曝光时长和时间间隔，还会把当前中心与“上一帧中心沿
    线段方向前进/后退一个预期步长”的两个位置作软比较。线段没有正反
    方向，所以这里必须保留两个分支，不能擅自给单帧线段指定运动方向。
    这里把这两类证据作为 *cost*，而不是 ``pair_allowed`` 的硬门：阈值化、
    饱和和边缘截断都会让端点抖动，硬删会误伤真实目标。边缘候选降低
    端点项的权重，因为看不见的那一端本来就没有可验证的端点。

    返回值的单位是像素量级的附加代价，供局部 Hungarian 分配器比较；
    它不改变对外返回的中心距离。
    """

    if psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive")
    delta = np.asarray(
        (
            float(current.aligned_x) - float(previous.aligned_x),
            float(current.aligned_y) - float(previous.aligned_y),
        ),
        dtype=np.float64,
    )
    movement = float(np.hypot(delta[0], delta[1]))
    if not np.isfinite(movement) or movement <= max(1.0, 0.75 * float(psf_fwhm)):
        # 没有可辨识的帧间方向时，不凭方向猜测；中心预测和曝光速度
        # 门仍由调用方负责。
        return 0.0
    movement_angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    edge_weight = 0.30 if (previous.touches_edge or current.touches_edge) else 1.0
    penalty = 0.0

    angles = (float(previous.angle_deg), float(current.angle_deg))
    if all(np.isfinite(angle) for angle in angles):
        angle_error = float(
            0.5
            * (
                _angle_difference_mod_180(angles[0], movement_angle)
                + _angle_difference_mod_180(angles[1], movement_angle)
            )
        )
        # 35 度是软尺度，不是通过/不通过的分界；即使完全不一致，
        # 代价也只会降低该配对优先级，不会绕过中心距离搜索门。
        penalty += edge_weight * min(5.0, 2.0 * (angle_error / 35.0) ** 2)

        if (
            expected_step_px is not None
            and np.isfinite(float(expected_step_px))
            and float(expected_step_px) > 0.0
        ):
            previous_unit = np.asarray(
                (np.cos(np.deg2rad(angles[0])), np.sin(np.deg2rad(angles[0]))),
                dtype=np.float64,
            )
            previous_center = np.asarray(
                (float(previous.aligned_x), float(previous.aligned_y)),
                dtype=np.float64,
            )
            current_center = np.asarray(
                (float(current.aligned_x), float(current.aligned_y)),
                dtype=np.float64,
            )
            expected_centers = np.asarray(
                (
                    previous_center + float(expected_step_px) * previous_unit,
                    previous_center - float(expected_step_px) * previous_unit,
                ),
                dtype=np.float64,
            )
            expected_center_error = float(
                np.min(np.hypot(*(expected_centers - current_center).T))
            )
            expected_step_scale = max(
                2.0 * float(psf_fwhm),
                0.20 * float(expected_step_px),
            )
            if np.isfinite(expected_center_error) and expected_step_scale > 0.0:
                penalty += edge_weight * min(
                    5.0,
                    1.25 * (expected_center_error / expected_step_scale) ** 2,
                )

    lengths = (float(previous.length_px), float(current.length_px))
    if all(np.isfinite(length) and length > 0.0 for length in lengths):
        previous_angle = np.deg2rad(float(previous.angle_deg))
        current_angle = np.deg2rad(float(current.angle_deg))
        if np.isfinite(previous_angle) and np.isfinite(current_angle):
            previous_half = 0.5 * lengths[0] * np.asarray(
                (np.cos(previous_angle), np.sin(previous_angle)),
                dtype=np.float64,
            )
            current_half = 0.5 * lengths[1] * np.asarray(
                (np.cos(current_angle), np.sin(current_angle)),
                dtype=np.float64,
            )
            previous_endpoints = np.asarray(
                (
                    (float(previous.aligned_x), float(previous.aligned_y))
                    - previous_half,
                    (float(previous.aligned_x), float(previous.aligned_y))
                    + previous_half,
                ),
                dtype=np.float64,
            )
            current_endpoints = np.asarray(
                (
                    (float(current.aligned_x), float(current.aligned_y))
                    - current_half,
                    (float(current.aligned_x), float(current.aligned_y))
                    + current_half,
                ),
                dtype=np.float64,
            )
            translated_previous = previous_endpoints + delta
            direct_error = float(
                np.mean(
                    np.hypot(
                        *(translated_previous - current_endpoints).T
                    )
                )
            )
            swapped_error = float(
                np.mean(
                    np.hypot(
                        *(translated_previous - current_endpoints[::-1]).T
                    )
                )
            )
            endpoint_error = min(direct_error, swapped_error)
            endpoint_scale = max(
                1.5 * float(psf_fwhm),
                0.12 * min(lengths),
            )
            if np.isfinite(endpoint_error) and endpoint_scale > 0.0:
                penalty += edge_weight * min(
                    3.0,
                    0.75 * (endpoint_error / endpoint_scale) ** 2,
                )
    return float(penalty) if np.isfinite(penalty) else 0.0


def _fit_collinear_motion_geometry(
    points: Sequence[MotionFeaturePoint],
    *,
    frame_times: Sequence[float] | np.ndarray | None = None,
    reference_interval_s: float | None = None,
) -> tuple[float, float, float | None, float | None]:
    """拟合线状候选中心的空间直线。

    返回 ``(projected_span, mean_speed, rms, angle)``。这里的 span 是
    配准坐标中沿运动路径的空间跨度，mean_speed 是跨度除以首末观测
    时间；返回值仍按中位帧间隔折算为兼容的 px/frame 字段。它不假定
    曝光中途的质心一定位于线段中点，因此适合处理
    目标高速穿过视场、线段被图像边界截断的情况。
    """

    if len(points) < 2:
        return 0.0, 0.0, None, None
    coordinates = np.asarray(
        [(point.aligned_x, point.aligned_y) for point in points],
        dtype=np.float64,
    )
    center = np.mean(coordinates, axis=0)
    centered = coordinates - center
    # 这里始终是二维坐标。直接求 2×2 协方差的主轴，和
    # ``np.linalg.eigh`` 的结果等价，但避免在长线补救的两两候选搜索中
    # 反复创建小矩阵、调用通用特征分解器。残差和投影跨度的定义保持不变。
    xx = float(np.mean(centered[:, 0] * centered[:, 0]))
    yy = float(np.mean(centered[:, 1] * centered[:, 1]))
    xy = float(np.mean(centered[:, 0] * centered[:, 1]))
    if not np.all(np.isfinite((xx, yy, xy))):
        return 0.0, 0.0, None, None
    anisotropy = float(np.hypot(xx - yy, 2.0 * xy))
    if not np.isfinite(anisotropy) or anisotropy <= np.finfo(np.float64).eps:
        return 0.0, 0.0, None, None
    principal_angle = 0.5 * float(np.arctan2(2.0 * xy, xx - yy))
    direction = np.asarray(
        (np.cos(principal_angle), np.sin(principal_angle)),
        dtype=np.float64,
    )
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    projections = centered @ direction
    residuals = np.abs(centered @ normal)
    span = float(np.max(projections) - np.min(projections))
    rms = float(np.sqrt(np.mean(residuals**2)))
    time = _point_motion_times(points, frame_times)
    time_span = float(np.max(time) - np.min(time))
    speed_per_second = span / time_span if time_span > 0 else 0.0
    interval = (
        float(reference_interval_s)
        if reference_interval_s is not None and np.isfinite(float(reference_interval_s)) and float(reference_interval_s) > 0.0
        else 1.0
    )
    mean_speed = speed_per_second * interval
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return span, float(mean_speed), rms, angle


def _link_motion_evidence_tracks(
    point_tracks: Sequence[SourceTrack],
    line_tracks: Sequence[MotionFeatureTrack],
    *,
    min_overlap_fraction: float = 0.5,
    max_perpendicular_distance_px: float = 10.0,
    max_direction_difference_deg: float = 25.0,
) -> tuple[MotionEvidenceLink, ...]:
    """关联点源和线状通道对同一运动源的重复证据。

    点源通道测到的是每帧最亮的质心/峰值，线状通道测到的是 1.5 s
    曝光留下的线段中心；两者沿运动方向可以有一个稳定的纵向偏移，
    因此不能用“中心必须重合”去匹配。这里比较共同帧中的垂直距离，
    再要求运动方向、位移尺度和时间覆盖一致。返回的 link 只做证据
    去重和审计，不删除任何 ``SourceTrack`` 或 ``MotionFeatureTrack``。
    """

    if not 0.0 < float(min_overlap_fraction) <= 1.0:
        raise ValueError("min_overlap_fraction must be in (0, 1]")
    if max_perpendicular_distance_px <= 0.0 or max_direction_difference_deg <= 0.0:
        raise ValueError("motion evidence link thresholds must be positive")

    moving_points = tuple(
        track
        for track in point_tracks
        if track.classification == "moving" and len(track.points) >= 2
    )
    moving_lines = tuple(
        track
        for track in line_tracks
        if track.classification == "moving" and len(track.points) >= 3
    )
    if not moving_points or not moving_lines:
        return ()

    pair_candidates: list[tuple[tuple[float, float, float, float], MotionEvidenceLink]] = []
    for point_track in moving_points:
        point_points = tuple(sorted(point_track.points, key=lambda point: point.frame_index))
        point_first = np.asarray(
            (point_points[0].aligned_x, point_points[0].aligned_y),
            dtype=np.float64,
        )
        point_last = np.asarray(
            (point_points[-1].aligned_x, point_points[-1].aligned_y),
            dtype=np.float64,
        )
        point_vector = point_last - point_first
        point_displacement = float(np.linalg.norm(point_vector))
        if not np.isfinite(point_displacement) or point_displacement <= 0.0:
            continue
        point_direction = point_vector / point_displacement
        point_normal = np.asarray((-point_direction[1], point_direction[0]), dtype=np.float64)
        point_by_frame = {int(point.frame_index): point for point in point_points}

        for line_track in moving_lines:
            line_points = tuple(sorted(line_track.points, key=lambda point: point.frame_index))
            line_first = np.asarray(
                (line_points[0].aligned_x, line_points[0].aligned_y),
                dtype=np.float64,
            )
            line_last = np.asarray(
                (line_points[-1].aligned_x, line_points[-1].aligned_y),
                dtype=np.float64,
            )
            line_vector = line_last - line_first
            line_displacement = float(np.linalg.norm(line_vector))
            if not np.isfinite(line_displacement) or line_displacement <= 0.0:
                continue
            direction_cosine = float(np.dot(point_direction, line_vector / line_displacement))
            direction_cosine = float(np.clip(direction_cosine, -1.0, 1.0))
            direction_difference = float(np.degrees(np.arccos(direction_cosine)))
            if direction_difference > float(max_direction_difference_deg):
                continue
            displacement_ratio = line_displacement / point_displacement
            if not 0.35 <= displacement_ratio <= 2.5:
                continue

            line_by_frame = {int(point.frame_index): point for point in line_points}
            common_frames = sorted(set(point_by_frame).intersection(line_by_frame))
            minimum_overlap = max(
                3,
                int(
                    np.ceil(
                        float(min_overlap_fraction)
                        * min(len(point_points), len(line_points))
                    )
                ),
            )
            if len(common_frames) < minimum_overlap:
                continue

            perpendicular_distances = []
            for frame_index in common_frames:
                line_point = line_by_frame[frame_index]
                line_position = np.asarray(
                    (line_point.aligned_x, line_point.aligned_y),
                    dtype=np.float64,
                )
                perpendicular_distances.append(
                    abs(float(np.dot(line_position - point_first, point_normal)))
                )
            median_perpendicular_distance = float(np.median(perpendicular_distances))
            if not np.isfinite(median_perpendicular_distance) or median_perpendicular_distance > float(max_perpendicular_distance_px):
                continue

            overlap_fraction = float(
                len(common_frames) / max(1, min(len(point_points), len(line_points)))
            )
            link = MotionEvidenceLink(
                point_track_id=int(point_track.track_id),
                line_track_id=int(line_track.track_id),
                overlap_frame_count=len(common_frames),
                overlap_fraction=overlap_fraction,
                median_perpendicular_distance_px=median_perpendicular_distance,
                direction_difference_deg=direction_difference,
            )
            pair_candidates.append(
                (
                    (
                        overlap_fraction,
                        -median_perpendicular_distance,
                        -direction_difference,
                        -abs(np.log(displacement_ratio)),
                    ),
                    link,
                )
            )

    if not pair_candidates:
        return ()
    pair_candidates.sort(key=lambda item: item[0], reverse=True)
    used_points: set[int] = set()
    used_lines: set[int] = set()
    links: list[MotionEvidenceLink] = []
    for _score, link in pair_candidates:
        if link.point_track_id in used_points or link.line_track_id in used_lines:
            continue
        used_points.add(link.point_track_id)
        used_lines.add(link.line_track_id)
        links.append(link)
    return tuple(sorted(links, key=lambda link: (link.point_track_id, link.line_track_id)))


def _best_collinear_motion_feature_group(
    points: Sequence[MotionFeaturePoint],
    candidate_indices: Sequence[int],
    *,
    min_length_px: float,
    angle_tolerance_deg: float,
    line_tolerance_px: float,
    min_span_px: float,
    frame_times: Sequence[float] | np.ndarray | None = None,
    reference_interval_s: float | None = None,
) -> tuple[int, ...] | None:
    """从未关联的长线候选中找一组跨帧共线、单调推进的线段。

    线段之间可能相隔数百像素，不能用相邻帧质心半径关联。这里以两条
    长线的中心连线作为暂定路径，再检查所有候选的方向、到该路径的
    垂距以及随帧号的投影是否单调。候选规模通常很小；仍使用数组化
    的成员筛选，避免在高阈值误检增多时出现三重 Python 循环。
    """

    if len(candidate_indices) < 3:
        return None
    valid_indices = [
        int(index)
        for index in candidate_indices
        if float(points[int(index)].length_px) >= float(min_length_px)
    ]
    if len(valid_indices) < 3:
        return None
    coordinates = np.asarray(
        [(points[index].aligned_x, points[index].aligned_y) for index in valid_indices],
        dtype=np.float64,
    )
    angles = np.asarray([points[index].angle_deg for index in valid_indices], dtype=np.float64)
    frame_numbers = np.asarray([points[index].frame_index for index in valid_indices], dtype=np.intp)
    best_score: tuple[int, float, float, float] | None = None
    best_group: tuple[int, ...] | None = None
    monotonic_tolerance = max(2.0, 0.25 * float(line_tolerance_px))

    for left in range(len(valid_indices) - 1):
        for right in range(left + 1, len(valid_indices)):
            if frame_numbers[left] == frame_numbers[right]:
                continue
            vector = coordinates[right] - coordinates[left]
            separation = float(np.linalg.norm(vector))
            if not np.isfinite(separation) or separation < float(min_span_px):
                continue
            direction = vector / separation
            line_angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
            angle_mask = np.asarray(
                [
                    _angle_difference_mod_180(angle, line_angle) <= float(angle_tolerance_deg)
                    for angle in angles
                ],
                dtype=bool,
            )
            if np.count_nonzero(angle_mask) < 3:
                continue
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
            perpendicular = np.abs((coordinates - coordinates[left]) @ normal)
            member_mask = angle_mask & (perpendicular <= float(line_tolerance_px))
            member_positions = np.flatnonzero(member_mask)
            if member_positions.size < 3:
                continue

            # 同一帧可能有多个近似平行的连通域，只取到暂定路径最近的
            # 一个；随后再用单调性和空间直线 RMS 复核。
            selected_positions: list[int] = []
            for frame_index in sorted(set(frame_numbers[member_positions].tolist())):
                same_frame = member_positions[frame_numbers[member_positions] == frame_index]
                selected_positions.append(
                    int(same_frame[np.argmin(perpendicular[same_frame])])
                )
            if len(selected_positions) < 3:
                continue
            selected_positions.sort(key=lambda position: int(frame_numbers[position]))
            selected_projection = (coordinates[selected_positions] - coordinates[left]) @ direction
            differences = np.diff(selected_projection)
            increasing = bool(np.all(differences >= -monotonic_tolerance))
            decreasing = bool(np.all(differences <= monotonic_tolerance))
            if not increasing and not decreasing:
                continue
            selected_span = float(np.max(selected_projection) - np.min(selected_projection))
            if selected_span < float(min_span_px):
                continue
            selected_points = tuple(points[valid_indices[position]] for position in selected_positions)
            fitted_span, _speed, fitted_rms, _fitted_angle = _fit_collinear_motion_geometry(
                selected_points,
                frame_times=frame_times,
                reference_interval_s=reference_interval_s,
            )
            if fitted_rms is None or fitted_span < float(min_span_px) or fitted_rms > float(line_tolerance_px):
                continue
            score = (
                len(selected_points),
                fitted_span,
                -float(fitted_rms),
                -float(np.sum(perpendicular[selected_positions])),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_group = tuple(valid_indices[position] for position in selected_positions)
    return best_group


def _stitch_unassociated_single_frame_points(
    tracks: Sequence[Sequence[MotionFeaturePoint]],
    points: Sequence[MotionFeaturePoint],
    *,
    psf_fwhm: float,
    frame_times: Sequence[float] | np.ndarray,
    max_frame_gap: int = 3,
    max_prediction_error_px: float | None = None,
) -> tuple[list[list[MotionFeaturePoint]], tuple[MotionFeaturePoint, ...]]:
    """Attach well-predicted single-frame line evidence to an existing track.

    The normal linker intentionally uses a local frame-to-frame radius.  That
    is safe for ordinary candidates, but it leaves a short line in F01/F03
    disconnected when the same object is only visible again from F05 onward.
    This second pass is deliberately narrow: an existing track needs at least
    three points, the candidate must be within a bounded temporal gap and a
    constant-velocity prediction, and its line model/axis/length must agree
    with the track.  It never labels a track ``moving``; the normal final gates
    run again after stitching.
    """

    if psf_fwhm <= 0.0 or max_frame_gap <= 0:
        raise ValueError("stitching parameters must be positive")
    time = _normalise_motion_times(frame_times, len(frame_times))
    tolerance = (
        max(8.0, 3.0 * float(psf_fwhm))
        if max_prediction_error_px is None
        else float(max_prediction_error_px)
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("max_prediction_error_px must be positive")

    mutable_tracks = [list(history) for history in tracks]
    remaining = list(points)
    # Process later frames first.  Once F03 is attached, F01 can be tested
    # against the extended history instead of being compared with a stale F05
    # endpoint.
    remaining.sort(key=lambda point: int(point.frame_index), reverse=True)

    def score_candidate(
        history: Sequence[MotionFeaturePoint],
        point: MotionFeaturePoint,
        track_index: int,
    ) -> tuple[float, float, float, int] | None:
        ordered = sorted(history, key=lambda item: int(item.frame_index))
        point_frame = int(point.frame_index)
        if len(ordered) < 3 or point_frame in {int(item.frame_index) for item in ordered}:
            return None
        if point_frame < 0 or point_frame >= len(time):
            return None
        nearest_gap = min(abs(point_frame - int(item.frame_index)) for item in ordered)
        if nearest_gap < 1 or nearest_gap > int(max_frame_gap):
            return None
        indices = np.asarray([int(item.frame_index) for item in ordered], dtype=np.intp)
        if np.any(indices < 0) or np.any(indices >= len(time)):
            return None
        fit_time = time[indices]
        if len(np.unique(fit_time)) < 2:
            return None
        x = np.asarray([float(item.aligned_x) for item in ordered], dtype=np.float64)
        y = np.asarray([float(item.aligned_y) for item in ordered], dtype=np.float64)
        if not np.all(np.isfinite((*x, *y))):
            return None
        x_slope, x_intercept = np.polyfit(fit_time, x, 1)
        y_slope, y_intercept = np.polyfit(fit_time, y, 1)
        velocity = np.asarray((float(x_slope), float(y_slope)), dtype=np.float64)
        velocity_norm = float(np.linalg.norm(velocity))
        if not np.isfinite(velocity_norm) or velocity_norm <= 0.0:
            return None
        predicted = np.asarray(
            (
                x_slope * time[point_frame] + x_intercept,
                y_slope * time[point_frame] + y_intercept,
            ),
            dtype=np.float64,
        )
        observed = np.asarray((float(point.aligned_x), float(point.aligned_y)), dtype=np.float64)
        error = float(np.linalg.norm(observed - predicted))
        if not np.isfinite(error) or error > tolerance:
            return None
        motion_angle = float(np.degrees(np.arctan2(velocity[1], velocity[0])))
        if _angle_difference_mod_180(float(point.angle_deg), motion_angle) > 35.0:
            return None
        median_length = float(np.median([float(item.length_px) for item in ordered]))
        if (
            not np.isfinite(median_length)
            or median_length <= 0.0
            or not 0.35 <= float(point.length_px) / median_length <= 2.5
        ):
            return None
        return (error, float(nearest_gap), -float(len(ordered)), int(track_index))

    for point in tuple(remaining):
        if not _single_frame_sequence_line_model_is_review_supported(
            point,
            psf_fwhm=float(psf_fwhm),
        ):
            continue
        if (
            point.line_artifact_overlap_fraction is not None
            and float(point.line_artifact_overlap_fraction)
            >= SINGLE_FRAME_ARTIFACT_REVIEW_OVERLAP_FRACTION
        ):
            continue
        scored = [
            score
            for track_index, history in enumerate(mutable_tracks)
            if (score := score_candidate(history, point, track_index)) is not None
        ]
        best = min(scored) if scored else None
        if best is None:
            continue
        track_index = int(best[3])
        mutable_tracks[track_index].append(point)
        remaining.remove(point)
    # 回接过程按“后面的 singleton 优先”进行，便于预测链逐步延长；
    # 对外返回前必须恢复时间顺序，保证拟合、导出和 UI 绘制都看到同一
    # 条从早到晚的轨迹。这里不改变点的内容，也不凭空补观测帧。
    for history in mutable_tracks:
        history.sort(key=lambda item: int(item.frame_index))
    return mutable_tracks, tuple(remaining)


def _recover_predicted_roi_motion_feature_points(
    tracks: list[list[MotionFeaturePoint]],
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    frame_times: Sequence[float] | np.ndarray,
    *,
    psf_fwhm: float,
) -> set[int]:
    """Recover short internal gaps with a bounded original-pixel recheck.

    This is intentionally a post-fit rescue, not a second whole-frame search.
    Only a short-line track with two strong observed runs and an internal gap of
    at most two frames is eligible.  The predicted centre limits the search to
    a small ROI; a recovered point must still win the raw line-vs-point/double/
    ellipse model and the normal sequence review gate.  The returned object ids
    let the report distinguish a local remeasurement from an ordinary detection.
    """

    if len(frame_analyses) < 2 or len(tracks) == 0 or psf_fwhm <= 0.0:
        return set()
    if len(frame_analyses) != len(cumulative_shifts):
        return set()
    times = _normalise_motion_times(frame_times, len(frame_analyses))
    recovered_ids: set[int] = set()
    claimed: set[tuple[int, int, int]] = set()
    short_length_limit = max(32.0, 8.0 * float(psf_fwhm))
    roi_radius = max(16, int(np.ceil(4.0 * float(psf_fwhm) + 12.0)))

    for history in tracks:
        ordered = sorted(history, key=lambda point: int(point.frame_index))
        if len(ordered) < 6:
            continue
        if float(np.median([point.length_px for point in ordered])) >= short_length_limit:
            continue
        observed_frames = {int(point.frame_index) for point in ordered}
        if min(observed_frames) < 0 or max(observed_frames) >= len(frame_analyses):
            continue

        # A single continuous run can simply be an incomplete observation;
        # require two strong runs so the local centre prediction is anchored on
        # both sides of the gap.
        runs: list[list[int]] = []
        for frame_index in sorted(observed_frames):
            if not runs or frame_index != runs[-1][-1] + 1:
                runs.append([frame_index])
            else:
                runs[-1].append(frame_index)
        strong_runs = [run for run in runs if len(run) >= 3]
        if len(strong_runs) < 2:
            continue
        rescue_frames = [
            (frame_index, False)
            for frame_index in range(min(observed_frames) + 1, max(observed_frames))
            if frame_index not in observed_frames
        ]
        # 末端没有“后一帧”做中心误差验证，因此只对至少四帧连续强观测
        # 的末端启用单侧补检，并在下面要求完整 strong line model；这比
        # 把所有末端外推位置都当成观测严格得多。
        last_run = strong_runs[-1]
        endpoint_frame = max(observed_frames) + 1
        if len(last_run) >= 4 and endpoint_frame < len(frame_analyses):
            rescue_frames.append((endpoint_frame, True))
        if not rescue_frames:
            continue
        gap_blocks: list[list[int]] = []
        for frame_index, is_endpoint in rescue_frames:
            if is_endpoint:
                continue
            if not gap_blocks or frame_index != gap_blocks[-1][-1] + 1:
                gap_blocks.append([frame_index])
            else:
                gap_blocks[-1].append(frame_index)
        if any(len(block) > 2 for block in gap_blocks):
            continue

        frame_indices = np.asarray([point.frame_index for point in ordered], dtype=np.intp)
        aligned_x = np.asarray([point.aligned_x for point in ordered], dtype=np.float64)
        aligned_y = np.asarray([point.aligned_y for point in ordered], dtype=np.float64)
        if len(np.unique(times[frame_indices])) < 2:
            continue
        x_slope, x_intercept = np.polyfit(times[frame_indices], aligned_x, 1)
        y_slope, y_intercept = np.polyfit(times[frame_indices], aligned_y, 1)
        velocity_angle = float(np.degrees(np.arctan2(y_slope, x_slope)))
        median_length = float(np.median([point.length_px for point in ordered]))

        for frame_index, is_endpoint in rescue_frames:
            predicted_aligned_x = float(x_slope * times[frame_index] + x_intercept)
            predicted_aligned_y = float(y_slope * times[frame_index] + y_intercept)
            shift_x, shift_y = cumulative_shifts[frame_index]
            predicted_x = predicted_aligned_x + float(shift_x)
            predicted_y = predicted_aligned_y + float(shift_y)
            analysis = frame_analyses[frame_index]
            values = np.asarray(analysis.frame.data, dtype=np.float32)
            height, width = values.shape
            x0 = max(0, int(np.floor(predicted_x)) - roi_radius)
            y0 = max(0, int(np.floor(predicted_y)) - roi_radius)
            x1 = min(width, x0 + 2 * roi_radius + 1)
            y1 = min(height, y0 + 2 * roi_radius + 1)
            crop = values[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            parameters = analysis.detection.parameters
            valid = np.isfinite(crop)
            if y0 == 0:
                auxiliary_width = max(0, min(crop.shape[1], 104 - x0))
                if auxiliary_width:
                    valid[0, :auxiliary_width] = False
            if int(parameters.get("mask_zero_pixels", 0)):
                valid &= crop != 0.0
            saturation_level = float(parameters.get("saturation_level", -1.0))
            if saturation_level > 0.0:
                valid &= crop < saturation_level
            background = float(analysis.detection.background)
            noise = max(float(analysis.detection.noise), np.finfo(np.float32).eps)
            residual = np.where(valid, crop.astype(np.float64) - background, 0.0)

            artifact_mask = np.zeros(crop.shape, dtype=bool)
            for artifact_x, artifact_y in np.asarray(
                analysis.detection.line_artifact_coordinates,
                dtype=np.int64,
            ).reshape(-1, 2):
                local_x = int(artifact_x) - x0
                local_y = int(artifact_y) - y0
                if 0 <= local_x < artifact_mask.shape[1] and 0 <= local_y < artifact_mask.shape[0]:
                    artifact_mask[local_y, local_x] = True
            try:
                proposals = detect_local_pixel_trail_proposals(
                    crop,
                    valid,
                    background,
                    noise,
                    float(psf_fwhm),
                    length_bank_px=(6.0, 8.0, 10.0, 12.0, 16.0, 20.0),
                    angle_step_deg=10.0,
                    max_proposals=12,
                    max_seeds=64,
                    seed_threshold=1.5,
                )
            except (TypeError, ValueError, FloatingPointError):
                proposals = ()
            if not proposals:
                continue

            candidates: list[tuple[float, MotionFeaturePoint]] = []
            for proposal in proposals:
                global_x = float(proposal.x) + x0
                global_y = float(proposal.y) + y0
                centre_error = float(np.hypot(global_x - predicted_x, global_y - predicted_y))
                if centre_error > 2.5:
                    continue
                if _angle_difference_mod_180(float(proposal.angle), velocity_angle) > 20.0:
                    continue
                if not 0.50 <= float(proposal.length) / median_length <= 1.80:
                    continue
                claim = (frame_index, int(round(global_x)), int(round(global_y)))
                if claim in claimed:
                    continue
                local_angle = float(proposal.angle) % 180.0
                local_width = max(float(psf_fwhm), float(proposal.width))
                half_width = max(0.5, 0.5 * local_width)
                radians = np.deg2rad(local_angle)
                x_extent = 0.5 * float(proposal.length) * abs(float(np.cos(radians))) + half_width * abs(float(np.sin(radians))) + 1.0
                y_extent = 0.5 * float(proposal.length) * abs(float(np.sin(radians))) + half_width * abs(float(np.cos(radians))) + 1.0
                bbox = (
                    max(0, int(np.floor(global_x - x_extent))),
                    max(0, int(np.floor(global_y - y_extent))),
                    min(width - 1, int(np.ceil(global_x + x_extent))),
                    min(height - 1, int(np.ceil(global_y + y_extent))),
                )
                continuity, fill = _line_profile_metrics(
                    residual,
                    center_x=float(proposal.x),
                    center_y=float(proposal.y),
                    length_px=float(proposal.length),
                    width_px=local_width,
                    angle_deg=local_angle,
                    threshold_adu=max(3.0 * noise, np.finfo(np.float64).eps),
                )
                peak = _single_frame_proposal_peak_residual(
                    residual,
                    center_x=float(proposal.x),
                    center_y=float(proposal.y),
                    length_px=float(proposal.length),
                    width_px=local_width,
                    angle_deg=local_angle,
                )
                point = MotionFeaturePoint(
                    frame_index=frame_index,
                    x=float(proposal.x),
                    y=float(proposal.y),
                    aligned_x=float(proposal.x) + x0 - float(shift_x),
                    aligned_y=float(proposal.y) + y0 - float(shift_y),
                    residual_snr=peak / noise,
                    area_pixels=max(1, int(round(float(proposal.length) * local_width))),
                    length_px=float(proposal.length),
                    width_px=local_width,
                    angle_deg=local_angle,
                    bbox=bbox,
                    touches_edge=bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width - 1 or bbox[3] == height - 1,
                    continuity_ratio=continuity,
                    profile_fill_ratio=fill,
                    line_artifact_overlap_fraction=_single_frame_proposal_artifact_overlap(
                        artifact_mask,
                        center_x=float(proposal.x),
                        center_y=float(proposal.y),
                        length_px=float(proposal.length),
                        width_px=local_width,
                        angle_deg=local_angle,
                    ),
                    observation_source="recovered_by_predicted_roi",
                )
                point = _apply_single_frame_model_evidence(
                    point,
                    residual=residual,
                    valid=valid,
                    psf_fwhm=float(psf_fwhm),
                    noise=noise,
                )
                if (
                    not _single_frame_model_is_supported(point)
                    if is_endpoint
                    else not _single_frame_sequence_line_model_is_review_supported(
                        point,
                        psf_fwhm=float(psf_fwhm),
                    )
                ):
                    continue
                if (
                    point.line_artifact_overlap_fraction is not None
                    and float(point.line_artifact_overlap_fraction)
                    >= SINGLE_FRAME_ARTIFACT_REVIEW_OVERLAP_FRACTION
                ):
                    continue
                point = replace(
                    point,
                    x=float(proposal.x) + x0,
                    y=float(proposal.y) + y0,
                    bbox=bbox,
                    observation_source="recovered_by_predicted_roi",
                )
                candidates.append((centre_error, point))
            if not candidates:
                continue
            _, best_point = min(candidates, key=lambda item: item[0])
            history.append(best_point)
            observed_frames.add(frame_index)
            claimed.add((frame_index, int(round(best_point.x)), int(round(best_point.y))))
            recovered_ids.add(id(best_point))
        history.sort(key=lambda point: int(point.frame_index))
    return recovered_ids


def _rescue_collinear_motion_feature_groups(
    tracks: Sequence[Sequence[MotionFeaturePoint]],
    *,
    psf_fwhm: float,
    min_presence: int = 3,
    angle_tolerance_deg: float = 12.0,
    line_tolerance_px: float = 12.0,
    min_span_px: float = 64.0,
    frame_times: Sequence[float] | np.ndarray | None = None,
    reference_interval_s: float | None = None,
) -> tuple[tuple[MotionFeaturePoint, ...], ...]:
    """在中心点关联失败后，补找跨大位移的长线几何轨迹。"""

    singleton_points = [
        history[0]
        for history in tracks
        if len(history) == 1 and float(history[0].length_px) >= max(32.0, 8.0 * float(psf_fwhm))
    ]
    if len(singleton_points) < int(min_presence):
        return ()
    available = set(range(len(singleton_points)))
    rescued: list[tuple[MotionFeaturePoint, ...]] = []
    minimum_length = max(32.0, 8.0 * float(psf_fwhm))
    while len(available) >= int(min_presence):
        group = _best_collinear_motion_feature_group(
            singleton_points,
            sorted(available),
            min_length_px=minimum_length,
            angle_tolerance_deg=angle_tolerance_deg,
            line_tolerance_px=line_tolerance_px,
            min_span_px=max(float(min_span_px), 4.0 * float(psf_fwhm)),
            frame_times=frame_times,
            reference_interval_s=reference_interval_s,
        )
        if group is None or len(group) < int(min_presence):
            break
        selected = tuple(singleton_points[index] for index in group)
        rescued.append(selected)
        available.difference_update(group)
    return tuple(rescued)


def _registered_median_reference(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    tile_rows: int = 256,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray | None:
    """以低内存方式生成注册后的时间中值参考图。

    每帧先按 ``aligned = raw - cumulative_shift`` 的约定重采样到第 1 帧
    坐标，再对时间轴取中值。使用分块而不是一次性堆叠完整的 15 帧，避免
    4096² 图像在 GUI 后台额外占用约 1 GB 浮点内存。
    """

    if len(frame_analyses) < 2:
        return None
    if len(frame_analyses) != len(cumulative_shifts):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    shape = tuple(int(value) for value in frame_analyses[0].frame.data.shape)
    if len(shape) != 2 or any(tuple(int(value) for value in analysis.frame.data.shape) != shape for analysis in frame_analyses):
        return None
    height, width = shape
    tile_height = max(16, int(tile_rows))
    tile_count = max(1, int(np.ceil(height / tile_height)))
    reference = np.full(shape, np.nan, dtype=np.float32)
    source_x = np.arange(width, dtype=np.float32)[None, :]
    # 当前 15 帧的累计配准量小于 1 px。此时直接按原坐标分块取中值，
    # 把亚像素差异交给后续高阈值/细长形态筛选，能显著降低 GUI 的峰值
    # 内存；只有较大平移时才使用全图重采样。
    direct_reference = max(float(np.max(np.abs(cumulative_shifts))), 0.0) <= 0.75
    for tile_index, y_start in enumerate(range(0, height, tile_height), start=1):
        y_stop = min(height, y_start + tile_height)
        output_y = np.arange(y_start, y_stop, dtype=np.float32)[:, None]
        stack = np.full((len(frame_analyses), y_stop - y_start, width), np.nan, dtype=np.float32)
        for frame_index, analysis in enumerate(frame_analyses):
            parameters = analysis.detection.parameters
            if direct_reference:
                # 只把当前行块转成 float32，避免把整帧再复制一份。
                working = np.asarray(analysis.frame.data[y_start:y_stop], dtype=np.float32).copy()
                if y_start == 0:
                    working[0, : min(working.shape[1], 104)] = np.nan
            else:
                # 较大平移需要在统一坐标中插值；复制只活在本次 frame 循环。
                working = np.asarray(analysis.frame.data, dtype=np.float32).copy()
                working[0, : min(working.shape[1], 104)] = np.nan
            if int(parameters.get("mask_zero_pixels", 0)):
                working[working == 0.0] = np.nan
            saturation_level = float(parameters.get("saturation_level", -1.0))
            if saturation_level > 0:
                working[working >= saturation_level] = np.nan
            shift_x, shift_y = cumulative_shifts[frame_index]
            if direct_reference:
                stack[frame_index] = working
            else:
                source_y = output_y + np.float32(shift_y)
                source_x_frame = source_x + np.float32(shift_x)
                coordinates = (
                    np.broadcast_to(source_y, (y_stop - y_start, width)),
                    np.broadcast_to(source_x_frame, (y_stop - y_start, width)),
                )
                stack[frame_index] = ndimage.map_coordinates(
                    working,
                    coordinates,
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                ).astype(np.float32, copy=False)
        tile_reference = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        valid_positions = np.any(np.isfinite(stack), axis=0)
        if np.any(valid_positions):
            with np.errstate(invalid="ignore"):
                tile_reference[valid_positions] = np.nanmedian(stack[:, valid_positions], axis=0).astype(np.float32, copy=False)
        reference[y_start:y_stop] = tile_reference
        if progress is not None:
            progress(
                5.0 + 15.0 * tile_index / tile_count,
                f"构建时间中值参考 {tile_index}/{tile_count}",
            )
    return reference


def _registered_coadd_reference(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    tile_rows: int = 256,
    clip_sigma: float = 3.0,
    progress: Callable[[float, str], None] | None = None,
) -> np.ndarray | None:
    """生成用于宽筛提案的注册稳健均值图。

    与时间中值参考不同，稳健均值对稳定弱源的信噪比更友好；对每个像素
    先围绕时间中位数做一次 MAD 裁剪，再求均值，尽量压制只出现在少数
    帧中的亮线、宇宙线或瞬态结构。这个结果只用于提出坐标，不能绕过
    ``_candidate_consensus_tracks`` 的逐帧原图支持、形状和持续性细筛。

    分块读取和累加保持与中值参考相同的内存上限。当前 15 帧累计平移小
    于 1 px 时直接在原坐标上累加；平移较大时才使用一次线性重采样。
    ``clip_sigma`` 只控制多帧提案的瞬态抑制，不改变单帧检测阈值。
    """

    if len(frame_analyses) < 2:
        return None
    if len(frame_analyses) != len(cumulative_shifts):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    if clip_sigma <= 0:
        raise ValueError("clip_sigma must be positive")
    shape = tuple(int(value) for value in frame_analyses[0].frame.data.shape)
    if len(shape) != 2 or any(tuple(int(value) for value in analysis.frame.data.shape) != shape for analysis in frame_analyses):
        return None
    height, width = shape
    tile_height = max(16, int(tile_rows))
    tile_count = max(1, int(np.ceil(height / tile_height)))
    coadd = np.full(shape, np.nan, dtype=np.float32)
    source_x = np.arange(width, dtype=np.float32)[None, :]
    direct_reference = max(float(np.max(np.abs(cumulative_shifts))), 0.0) <= 0.75
    for tile_index, y_start in enumerate(range(0, height, tile_height), start=1):
        y_stop = min(height, y_start + tile_height)
        output_y = np.arange(y_start, y_stop, dtype=np.float32)[:, None]
        stack = np.full(
            (len(frame_analyses), y_stop - y_start, width),
            np.nan,
            dtype=np.float32,
        )
        for frame_index, analysis in enumerate(frame_analyses):
            parameters = analysis.detection.parameters
            if direct_reference:
                working = np.asarray(
                    analysis.frame.data[y_start:y_stop],
                    dtype=np.float32,
                ).copy()
                if y_start == 0:
                    working[0, : min(working.shape[1], 104)] = np.nan
            else:
                working = np.asarray(analysis.frame.data, dtype=np.float32).copy()
                working[0, : min(working.shape[1], 104)] = np.nan
            if int(parameters.get("mask_zero_pixels", 0)):
                working[working == 0.0] = np.nan
            saturation_level = float(parameters.get("saturation_level", -1.0))
            if saturation_level > 0:
                working[working >= saturation_level] = np.nan
            shift_x, shift_y = cumulative_shifts[frame_index]
            if direct_reference:
                stack[frame_index] = working
            else:
                source_y = output_y + np.float32(shift_y)
                source_x_frame = source_x + np.float32(shift_x)
                coordinates = (
                    np.broadcast_to(source_y, (y_stop - y_start, width)),
                    np.broadcast_to(source_x_frame, (y_stop - y_start, width)),
                )
                stack[frame_index] = ndimage.map_coordinates(
                    working,
                    coordinates,
                    order=1,
                    mode="constant",
                    cval=np.nan,
                    prefilter=False,
                ).astype(np.float32, copy=False)

        valid_positions = np.any(np.isfinite(stack), axis=0)
        median = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        if np.any(valid_positions):
            with np.errstate(invalid="ignore"):
                median[valid_positions] = np.nanmedian(
                    stack[:, valid_positions],
                    axis=0,
                ).astype(np.float32, copy=False)
        deviation = np.abs(stack - median[None, :, :])
        mad = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        if np.any(valid_positions):
            with np.errstate(invalid="ignore"):
                mad[valid_positions] = np.nanmedian(
                    deviation[:, valid_positions],
                    axis=0,
                ).astype(np.float32, copy=False)
        scale = np.maximum(
            1.4826 * mad,
            np.finfo(np.float32).eps,
        )
        keep = np.isfinite(stack) & (
            np.abs(stack - median[None, :, :]) <= float(clip_sigma) * scale[None, :, :]
        )
        finite_count = np.count_nonzero(keep, axis=0)
        summed = np.sum(np.where(keep, stack, 0.0), axis=0, dtype=np.float32)
        tile_coadd = np.full((y_stop - y_start, width), np.nan, dtype=np.float32)
        valid_positions = finite_count > 0
        tile_coadd[valid_positions] = (
            summed[valid_positions] / finite_count[valid_positions]
        ).astype(np.float32, copy=False)
        coadd[y_start:y_stop] = tile_coadd
        if progress is not None:
            progress(
                5.0 + 15.0 * tile_index / tile_count,
                f"构建稳健时间叠加 {tile_index}/{tile_count}",
            )
    return coadd


def _temporal_reference_candidate_frames(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    reference: np.ndarray | None,
    *,
    psf_fwhm: float,
    min_candidate_snr: float,
    temporal_multiscale: bool = False,
) -> tuple[tuple[np.ndarray, ...], int]:
    """从注册时间中值中提出需要回到逐帧核验的补充候选。

    单帧 Gaussian 峰提案有一个结构性盲点：如果两个紧邻源的响应重叠，
    或某一帧的峰恰好被坏像素/线状结构污染，源可能从首帧候选集合中消失。
    15 帧注册后的时间中值可以作为“宽筛补充”，但不能直接作为星点真值：
    固定图样和静态伪影也会被中值保留。因此这里只生成临时的逐帧强制测量
    坐标，并要求后续候选共识再次通过逐帧 PSF 响应、3×3 支持、孔径有效性
    和位置拟合；结果仍进入低置信 ``candidate_consensus`` 层。

    返回值是每帧的 ``(x, y, frame_filter_snr, channel, reference_snr)`` 数组
    以及时间中值宽筛提案数。``channel=1`` 和最后一列的参考 SNR 让后续
    共识可以使用较低的逐帧强制响应门槛，同时仍要求时间中值种子自身达到
    正式候选的显著性；不能把这个较低门槛误用于普通单帧候选。
    ``temporal_multiscale=True`` 时，参考图和逐帧强制响应使用相对
    ``psf_fwhm`` 的窄/基准/宽三尺度组，取各尺度的最大标准化响应；这样
    可以做窄 PSF 漏检抽测，但也会增加纹理候选，默认关闭。数组不写入
    序列缓存，避免把一个依赖当前参考构造的中间层误当成正式源表。
    """

    frame_count = len(frame_analyses)
    empty = tuple(np.empty((0, 5), dtype=np.float32) for _ in range(frame_count))
    if reference is None or frame_count < 2 or len(cumulative_shifts) != frame_count:
        return empty, 0
    if psf_fwhm <= 0 or min_candidate_snr <= 0:
        raise ValueError("temporal reference candidate thresholds must be positive")

    values = np.asarray(reference, dtype=np.float32)
    if values.ndim != 2:
        return empty, 0
    valid = np.isfinite(values)
    if not np.any(valid):
        return empty, 0
    # 注册中值已经把无效孔径变成 NaN；第一行辅助区再显式排除一次，防止
    # 在参考帧平移较大时把辅助字节重采样成一个亮的局部峰。
    valid = valid.copy()
    valid[0, : min(values.shape[1], 104)] = False
    background, noise = sigma_clipped_stats(
        values,
        mask=~valid,
        sample_limit=1_000_000,
    )
    noise = max(float(noise), np.finfo(np.float32).eps)
    residual = np.where(valid, values - float(background), 0.0).astype(np.float32, copy=False)
    temporal_fwhms = (
        _temporal_psf_fwhm_bank(float(psf_fwhm))
        if temporal_multiscale
        else (float(psf_fwhm),)
    )
    temporal_sigmas = tuple(max(fwhm / 2.35482, 0.5) for fwhm in temporal_fwhms)
    response_snr_bank: list[np.ndarray] = []
    response_local_max_bank: list[np.ndarray] = []
    for sigma in temporal_sigmas:
        response = ndimage.gaussian_filter(residual, sigma=sigma, mode="nearest")
        response_values = response[valid]
        response_background, response_noise = sigma_clipped_stats(
            response_values,
            sample_limit=1_000_000,
        )
        response_noise = max(float(response_noise), np.finfo(np.float32).eps)
        response_snr = (response - float(response_background)) / response_noise
        response_snr_bank.append(response_snr.astype(np.float32, copy=False))
        peak_radius = max(1, int(np.ceil(2.0 * sigma)))
        response_local_max_bank.append(
            response_snr == ndimage.maximum_filter(
                response_snr,
                size=2 * peak_radius + 1,
                mode="nearest",
            )
        )

    # 时间中值只负责补提案，所以阈值可以低于逐帧候选门槛；真正进入
    # 共识的逐帧行仍按 min_candidate_snr 过滤。这里的 0.5 倍是宽筛层，
    # 不是把 15σ 质量口径偷偷改成 7.5σ。
    seed_threshold = max(5.0, 0.5 * float(min_candidate_snr))
    raw_local_max = values == ndimage.maximum_filter(values, size=3, mode="nearest")
    response_snr_map = np.maximum.reduce(response_snr_bank)
    response_peak_mask = np.zeros(values.shape, dtype=bool)
    for response_snr, response_local_max in zip(
        response_snr_bank,
        response_local_max_bank,
        strict=True,
    ):
        response_peak_mask |= response_local_max & (response_snr >= seed_threshold)
    candidate_mask = (
        valid
        & response_peak_mask
        & raw_local_max
        & ((values - float(background)) >= 2.0 * noise)
    )
    candidate_y, candidate_x = np.nonzero(candidate_mask)
    if candidate_x.size == 0:
        return empty, 0

    first_parameters = frame_analyses[0].detection.parameters
    required_support = max(3, int(first_parameters.get("min_psf_support_pixels", 3)))
    # 只在参考图上做一次宽松形状审计，主要挡住时间中值中的长条和极窄
    # 单像素尖峰；最终的单帧质量规则仍然不被这个宽筛替代。
    shape_radius = max(4, int(np.ceil(3.5 * max(temporal_sigmas))))
    reference_rows: list[tuple[float, float, float]] = []
    for x_raw, y_raw in zip(candidate_x, candidate_y, strict=True):
        x = int(x_raw)
        y = int(y_raw)
        if (
            x < shape_radius
            or y < shape_radius
            or x >= values.shape[1] - shape_radius
            or y >= values.shape[0] - shape_radius
        ):
            continue
        patch = values[
            y - shape_radius : y + shape_radius + 1,
            x - shape_radius : x + shape_radius + 1,
        ].astype(np.float64, copy=False) - float(background)
        peak_excess = max(float(values[y, x] - float(background)), 0.0)
        core_slice = patch[shape_radius - 1 : shape_radius + 2, shape_radius - 1 : shape_radius + 2]
        support_threshold = max(2.0 * noise, 0.1 * peak_excess)
        if int(np.count_nonzero(core_slice >= support_threshold)) < required_support:
            continue
        weights = np.maximum(patch - 0.5 * noise, 0.0)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= np.finfo(np.float64).eps:
            continue
        yy, xx = np.indices(patch.shape, dtype=np.float64)
        center_x = float((weights * xx).sum() / weight_sum)
        center_y = float((weights * yy).sum() / weight_sum)
        variance_x = float((weights * (xx - center_x) ** 2).sum() / weight_sum)
        variance_y = float((weights * (yy - center_y) ** 2).sum() / weight_sum)
        covariance = float((weights * (xx - center_x) * (yy - center_y)).sum() / weight_sum)
        eigenvalues = np.linalg.eigvalsh(
            np.asarray(((variance_x, covariance), (covariance, variance_y)), dtype=np.float64)
        )
        major = max(float(eigenvalues[-1]), 0.0)
        minor = max(float(eigenvalues[0]), 0.0)
        fwhm = 2.35482 * float(np.sqrt(max((major + minor) / 2.0, 0.0)))
        ellipticity = 1.0 - np.sqrt(minor) / max(np.sqrt(major), np.finfo(np.float64).eps)
        if not np.isfinite(fwhm) or not 0.5 <= fwhm <= 12.0:
            continue
        if not np.isfinite(ellipticity) or ellipticity > 0.75:
            continue
        reference_rows.append((float(x), float(y), float(response_snr_map[y, x])))
    if not reference_rows:
        return empty, 0

    reference_coordinates = np.asarray(
        [(row[0], row[1]) for row in reference_rows],
        dtype=np.float32,
    )
    # 用多尺度 Gaussian 权重在每帧原图的预测位置附近做廉价强制响应。
    # 参考图经过注册和四舍五入后，真实 PSF 中心可能落在预测整数点的
    # 相邻像素；若只在一个像素上测量，会把亚像素配准误差误判成形状不符。
    # 因此只在 ±1 px 的 3×3 小窗口内寻找最高匹配响应，再把同一个中心
    # 用于 3×3 支持和后续 PSF 相关。搜索半径保持很小，避免把空间邻域中
    # 随机噪声的最大值当成星点坐标。
    local_search_radius = 1
    kernel_radius = max(3, int(np.ceil(2.75 * max(temporal_sigmas))))
    patch_radius = kernel_radius + local_search_radius
    offset_y, offset_x = np.mgrid[-patch_radius : patch_radius + 1, -patch_radius : patch_radius + 1]
    offset_y = offset_y.reshape(-1)
    offset_x = offset_x.reshape(-1)
    patch_side = 2 * patch_radius + 1
    kernel_offset_y, kernel_offset_x = np.mgrid[
        -kernel_radius : kernel_radius + 1,
        -kernel_radius : kernel_radius + 1,
    ]
    kernel_offset_y = kernel_offset_y.reshape(-1)
    kernel_offset_x = kernel_offset_x.reshape(-1)
    search_offset_y, search_offset_x = np.mgrid[
        -local_search_radius : local_search_radius + 1,
        -local_search_radius : local_search_radius + 1,
    ]
    search_offset_y = search_offset_y.reshape(-1)
    search_offset_x = search_offset_x.reshape(-1)
    search_patch_indices = np.asarray(
        [
            (kernel_offset_y + dy + patch_radius) * patch_side
            + (kernel_offset_x + dx + patch_radius)
            for dy, dx in zip(search_offset_y, search_offset_x, strict=True)
        ],
        dtype=np.intp,
    )
    core_offset_y, core_offset_x = np.mgrid[-1:2, -1:2]
    core_offset_y = core_offset_y.reshape(-1)
    core_offset_x = core_offset_x.reshape(-1)
    search_core_indices = np.asarray(
        [
            (core_offset_y + dy + patch_radius) * patch_side
            + (core_offset_x + dx + patch_radius)
            for dy, dx in zip(search_offset_y, search_offset_x, strict=True)
        ],
        dtype=np.intp,
    )
    search_center_indices = np.asarray(
        [
            (dy + patch_radius) * patch_side + (dx + patch_radius)
            for dy, dx in zip(search_offset_y, search_offset_x, strict=True)
        ],
        dtype=np.intp,
    )
    kernels = tuple(
        np.exp(
            -0.5
            * ((kernel_offset_x**2 + kernel_offset_y**2) / sigma**2)
        ).astype(np.float32)
        for sigma in temporal_sigmas
    )
    output: list[np.ndarray] = []
    frame_gate = max(4.0, 0.5 * float(min_candidate_snr))
    for frame_index, (analysis, shift) in enumerate(zip(frame_analyses, cumulative_shifts, strict=True)):
        image = np.asarray(analysis.frame.data, dtype=np.float32)
        frame_background = float(analysis.detection.background)
        frame_noise = max(float(analysis.detection.noise), np.finfo(np.float32).eps)
        x_integer = np.rint(reference_coordinates[:, 0] + float(shift[0])).astype(np.intp)
        y_integer = np.rint(reference_coordinates[:, 1] + float(shift[1])).astype(np.intp)
        inside = (
            (x_integer >= patch_radius)
            & (x_integer < image.shape[1] - patch_radius)
            & (y_integer >= patch_radius)
            & (y_integer < image.shape[0] - patch_radius)
        )
        frame_rows = np.empty((0, 5), dtype=np.float32)
        valid_indices = np.flatnonzero(inside)
        if valid_indices.size:
            patches = image[
                y_integer[valid_indices, None] + offset_y[None, :],
                x_integer[valid_indices, None] + offset_x[None, :],
            ]
            pixel_valid = np.isfinite(patches)
            parameters = analysis.detection.parameters
            if int(parameters.get("mask_zero_pixels", 0)):
                pixel_valid &= patches != 0.0
            saturation_level = float(parameters.get("saturation_level", -1.0))
            if saturation_level > 0:
                pixel_valid &= patches < saturation_level
            absolute_y = y_integer[valid_indices, None] + offset_y[None, :]
            absolute_x = x_integer[valid_indices, None] + offset_x[None, :]
            pixel_valid &= ~((absolute_y == 0) & (absolute_x < 104))
            residual_patch = patches - frame_background
            scores = np.full(
                (valid_indices.size, search_offset_x.size),
                -np.inf,
                dtype=np.float32,
            )
            for kernel in kernels:
                kernel_sq = kernel**2
                for search_index, patch_indices in enumerate(search_patch_indices):
                    sampled = patches[:, patch_indices]
                    sampled_valid = pixel_valid[:, patch_indices]
                    denominator = frame_noise * np.sqrt(
                        np.maximum(
                            np.sum(kernel_sq[None, :] * sampled_valid, axis=1),
                            np.finfo(np.float32).eps,
                        )
                    )
                    scale_scores = np.sum(
                        np.where(sampled_valid, sampled - frame_background, 0.0)
                        * kernel[None, :],
                        axis=1,
                    ) / denominator
                    scores[:, search_index] = np.maximum(
                        scores[:, search_index], scale_scores
                    )
            best_search = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(valid_indices.size), best_search]
            selected_core_indices = search_core_indices[best_search]
            row_indices = np.arange(valid_indices.size, dtype=np.intp)[:, None]
            core = residual_patch[row_indices, selected_core_indices]
            core_valid = pixel_valid[row_indices, selected_core_indices]
            selected_centers = search_center_indices[best_search]
            center_valid = pixel_valid[np.arange(valid_indices.size), selected_centers]
            core_peak = np.maximum(residual_patch[np.arange(valid_indices.size), selected_centers], 0.0)
            support_threshold = np.maximum(2.0 * frame_noise, 0.1 * core_peak)
            support = np.count_nonzero(
                (core >= support_threshold[:, None]) & core_valid,
                axis=1,
            )
            keep = (
                np.isfinite(best_scores)
                & (best_scores >= frame_gate)
                & center_valid
                & (support >= required_support)
            )
            if np.any(keep):
                kept_indices = valid_indices[keep]
                kept_search = best_search[keep]
                frame_rows = np.column_stack(
                    (
                        x_integer[kept_indices] + search_offset_x[kept_search],
                        y_integer[kept_indices] + search_offset_y[kept_search],
                        best_scores[keep],
                        np.ones(kept_indices.size, dtype=np.float32),
                        response_snr_map[y_integer[kept_indices], x_integer[kept_indices]],
                    )
                ).astype(np.float32, copy=False)
        output.append(frame_rows)
    return tuple(output), len(reference_rows)


def _coherent_raw_motion_point_ids(
    raw_feature_frames: Sequence[Sequence[MotionFeaturePoint]],
    *,
    psf_fwhm: float,
    min_presence: int = 3,
    frame_times: Sequence[float] | np.ndarray | None = None,
    exposure_times_s: Sequence[float] | np.ndarray | None = None,
) -> set[int]:
    """Return raw single-frame line points that form a short motion chain.

    The temporal median is useful for a wide proposal pass, but it can absorb
    a faint line whose center moves only a few pixels between adjacent frames.
    This helper provides a second proposal source for the *15-frame* route:
    it links raw line-shape points in adjacent frames only when their aligned
    centers move by a plausible non-zero amount and keep a consistent
    direction.  Static stars and fixed detector streaks normally have a
    sub-pixel aligned displacement and therefore do not enter this set.

    It is deliberately not used by the single-frame API.  A chain is only a
    proposal; the normal association, multi-frame shape gate, displacement,
    and fit checks still decide whether it is ``moving``.
    """

    if len(raw_feature_frames) < int(min_presence):
        return set()
    resolved_times = _normalise_motion_times(frame_times, len(raw_feature_frames))
    reference_interval_s = _reference_time_interval_seconds(resolved_times)
    if exposure_times_s is None:
        resolved_exposures = (reference_interval_s,) * len(raw_feature_frames)
    else:
        resolved_exposures = tuple(float(value) for value in exposure_times_s)
        if len(resolved_exposures) != len(raw_feature_frames):
            raise ValueError("exposure_times_s length must match raw_feature_frames")
        if any(not np.isfinite(value) or value <= 0.0 for value in resolved_exposures):
            raise ValueError("exposure_times_s must contain positive finite values")
    min_step_px = max(2.0, 1.0 * float(psf_fwhm))
    max_step_px = max(18.0, 12.0 * float(psf_fwhm))
    # Keep the historical spatial gates but scale them by the actual elapsed
    # time.  A 1.529 s gap should allow about 3.8% more displacement than a
    # 1.473 s gap for the same physical angular rate.
    min_speed_px_s = min_step_px / reference_interval_s
    max_speed_px_s = max_step_px / reference_interval_s
    states: dict[tuple[int, int], tuple[int, tuple[int, int] | None]] = {}

    for frame_index, current in enumerate(raw_feature_frames):
        for point_index, point in enumerate(current):
            key = (frame_index, point_index)
            best_length = 1
            best_previous: tuple[int, int] | None = None
            if frame_index > 0:
                for previous_index, previous in enumerate(raw_feature_frames[frame_index - 1]):
                    elapsed_s = float(resolved_times[frame_index] - resolved_times[frame_index - 1])
                    if elapsed_s <= 0.0:
                        continue
                    distance = float(
                        np.hypot(
                            point.aligned_x - previous.aligned_x,
                            point.aligned_y - previous.aligned_y,
                        )
                    )
                    # A line's measured length is an exposure-integrated
                    # displacement proxy.  It is not an exact speed because
                    # thresholding shortens the visible segment, but it gives
                    # the wide proposal chain a data-derived upper gate.  The
                    # four-times tolerance covers PSF broadening, clipped
                    # endpoints and the gap between exposure and DATE-OBS
                    # intervals; the final association/fitting gates remain
                    # responsible for rejecting accidental chains.
                    exposure_s = max(
                        np.finfo(np.float64).eps,
                        0.5 * (
                            resolved_exposures[frame_index]
                            + resolved_exposures[frame_index - 1]
                        ),
                    )
                    expected_line_step = (
                        max(float(point.length_px), float(previous.length_px))
                        * elapsed_s
                        / exposure_s
                    )
                    max_step_for_pair = max(
                        max_speed_px_s * elapsed_s,
                        4.0 * expected_line_step,
                    )
                    if distance < min_speed_px_s * elapsed_s or distance > max_step_for_pair:
                        continue
                    previous_key = (frame_index - 1, previous_index)
                    previous_length, previous_previous = states.get(previous_key, (1, None))
                    if previous_previous is not None:
                        previous_previous_point = raw_feature_frames[frame_index - 2][previous_previous[1]]
                        first_step = np.asarray(
                            (
                                previous.aligned_x - previous_previous_point.aligned_x,
                                previous.aligned_y - previous_previous_point.aligned_y,
                            ),
                            dtype=np.float64,
                        )
                        current_step = np.asarray(
                            (
                                point.aligned_x - previous.aligned_x,
                                point.aligned_y - previous.aligned_y,
                            ),
                            dtype=np.float64,
                        )
                        denominator = float(np.linalg.norm(first_step) * np.linalg.norm(current_step))
                        if denominator <= 0.0 or float(np.dot(first_step, current_step)) / denominator < 0.25:
                            continue
                    candidate_length = int(previous_length) + 1
                    if candidate_length > best_length:
                        best_length = candidate_length
                        best_previous = previous_key
            states[key] = (best_length, best_previous)

    selected: set[int] = set()
    for key, (chain_length, _previous) in states.items():
        if chain_length < int(min_presence):
            continue
        chain: list[tuple[int, int]] = []
        cursor: tuple[int, int] | None = key
        while cursor is not None:
            chain.append(cursor)
            cursor = states[cursor][1]
        # At least two strong shape points are required; a weak point may
        # bridge a threshold dip, but a chain of weak fragments alone is not
        # enough to be admitted into the sequence association stage.
        strong_count = sum(
            _sequence_line_shape_is_supported(raw_feature_frames[frame][index])
            for frame, index in chain
        )
        if strong_count < 2:
            continue
        selected.update(
            id(raw_feature_frames[frame][index])
            for frame, index in chain
        )
    return selected


def _motion_frame_residual_inputs(
    frame_analysis: FrameAnalysis,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build the current-frame-only inputs shared by motion line proposals.

    This helper deliberately reads one ``FrameAnalysis`` only.  It mirrors the
    validity, zero-pixel and saturation rules used by the temporal residual
    detector, so the multiscale spatial proposal does not silently inspect a
    different set of pixels from the later cross-frame confirmation stage.
    """

    values = np.asarray(frame_analysis.frame.data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    valid = np.isfinite(values)
    valid &= ~auxiliary_mask(values.shape)
    parameters = frame_analysis.detection.parameters
    if int(parameters.get("mask_zero_pixels", 0)):
        valid &= values != 0.0
    saturation_level = float(parameters.get("saturation_level", -1.0))
    if saturation_level > 0:
        valid &= values < saturation_level
    background = float(frame_analysis.detection.background)
    noise = max(float(frame_analysis.detection.noise), np.finfo(np.float32).eps)
    residual = np.where(valid, values - background, 0.0)
    line_artifact_mask = _line_artifact_mask_from_coordinates(
        frame_analysis.detection.line_artifact_coordinates,
        values.shape,
    )
    return residual, valid, line_artifact_mask, values, background, noise


def _extract_multiscale_motion_frame_features(
    frame_analysis: FrameAnalysis,
    *,
    frame_index: int,
    shift: tuple[float, float],
    psf_fwhm: float,
    min_length_px: float,
    temporal_reference: np.ndarray | None = None,
    model_proposal_budget: int | None = DEFAULT_SEQUENCE_LINE_MODEL_PROPOSAL_BUDGET,
) -> tuple[MotionFeaturePoint, ...]:
    """Propose line-shaped structures from one frame for sequence association.

    The underlying detector is the same spatial, multiscale matcher used by
    the single-frame display route, but this function returns only per-frame
    observations.  When a registered temporal reference is supplied, the
    matcher runs on the current-frame residual so fixed stars and detector
    streaks do not dominate the proposal budget.  It never assigns ``moving``;
    cross-frame displacement, continuity and fit gates remain the sole source
    of a sequence-level motion classification.
    """

    residual, valid, line_artifact_mask, values, background, noise = _motion_frame_residual_inputs(
        frame_analysis
    )
    if temporal_reference is not None:
        reference = ndimage.shift(
            np.asarray(temporal_reference, dtype=np.float32),
            shift=(float(shift[1]), float(shift[0])),
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        reference_valid = valid & np.isfinite(reference)
        residual = np.where(reference_valid, values - reference, 0.0)
        background = 0.0
        # Current frame and registered temporal reference contribute independent
        # noise terms to the difference.  Keep the conservative sqrt(2) factor
        # used by the primary residual detector.
        noise *= np.sqrt(2.0)
        del reference
    proposal_tracks = _augment_single_frame_trails_with_multiscale_proposals(
        (),
        residual=residual,
        valid=valid,
        line_artifact_mask=line_artifact_mask,
        background=background,
        noise=noise,
        psf_fwhm=float(psf_fwhm),
        min_length_px=float(min_length_px),
        progress=None,
        # Sequence mode supplements short/medium trails.  The primary
        # connected-component layer already handles very long structures; a
        # bounded bank and coarser angular grid avoid rediscovering one long
        # fixed detector streak at every scale.
        # 1.5 s 曝光下，前端候选下限可能低于 8 px。把这个下限显式
        # 放进模板库；它只增加短线 proposal，不改变后面的原图模型、
        # 跨帧覆盖、速度一致性和 moving 确认门。
        length_bank_px=tuple(
            sorted(
                {
                    float(min_length_px),
                    8.0,
                    12.0,
                    16.0,
                    24.0,
                    32.0,
                    48.0,
                    64.0,
                }
            )
        ),
        angle_step_deg=20.0,
        max_proposals=96,
        model_proposal_budget=model_proposal_budget,
        # 15 帧主残差层已经在原图处理长结构；这里保留原生像素短线
        # 和伪影区复核，避免对同一帧再做一次 1024² 的长线全图扫描。
        # 单帧入口不经过此调用，仍保留降采样长线支路。
        include_reduced_scale=False,
    )
    shift_x, shift_y = shift
    return tuple(
        replace(
            track.points[0],
            frame_index=int(frame_index),
            aligned_x=float(track.points[0].x) - float(shift_x),
            aligned_y=float(track.points[0].y) - float(shift_y),
        )
        for track in proposal_tracks
        if track.points
    )


def _merge_motion_feature_points(
    base_points: Sequence[MotionFeaturePoint],
    supplemental_points: Sequence[MotionFeaturePoint],
) -> list[MotionFeaturePoint]:
    """Merge two current-frame proposal sets without duplicating one line.

    The connected-component detector remains first in the list because it
    provides the original pixel-supported area.  A multiscale proposal is
    added only when its finite-line geometry does not overlap an existing
    proposal; this keeps association deterministic and prevents one physical
    line from competing with itself in the Hungarian linker.
    """

    merged = list(base_points)
    for candidate in supplemental_points:
        candidate_geometry = (
            float(candidate.x),
            float(candidate.y),
            float(candidate.length_px),
            float(candidate.width_px),
            float(candidate.angle_deg),
        )
        if any(
            _single_frame_line_geometries_overlap(
                candidate_geometry,
                (
                    float(existing.x),
                    float(existing.y),
                    float(existing.length_px),
                    float(existing.width_px),
                    float(existing.angle_deg),
                ),
            )
            for existing in merged
        ):
            continue
        merged.append(candidate)
    return merged


def detect_motion_features(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    psf_fwhm: float,
    residual_sigma: float = 15.0,
    min_residual_adu: float = 100.0,
    min_feature_area: int = 40,
    min_axis_ratio: float = 4.0,
    min_length_px: float | None = None,
    bridge_gaps: bool = DEFAULT_SEQUENCE_TRAIL_BRIDGE_GAPS,
    temporal_reference: np.ndarray | None = None,
    progress: Callable[[float, str], None] | None = None,
    audit_sink: list[MotionFrameAudit] | None = None,
    raw_fallback_workers: int = 1,
    multiscale_line_proposals: bool = False,
    line_link_radius_px: float | None = None,
) -> tuple[tuple[MotionFeatureTrack, ...], float]:
    """从每帧原图中提取长线，再用配准坐标做跨帧关联。

    当前数据中的运动目标不是一个每帧只移动几像素的点，而是高亮、细长的
    拖影/线状结构。旧的点源质量筛选会正确地拒绝它，但也因此把它完全从
    运动层抹掉。这里不把它重新塞回星点列表，而是单独建立“线状候选”层：
    先用原图背景和 RMS 找显著正残差，再用连通域 PCA 排除普通圆点和孤立
    热像素，最后在已经估计的全局平移坐标中关联。只有跨帧形成稳定轨迹的
    线状候选才标成 ``moving``。单帧曝光候选由独立的
    ``detect_single_frame_long_trails`` 路径仍然负责单帧按钮的独立结果；
    当 ``multiscale_line_proposals`` 开启时，这里只复用其“当前帧空间
    线段提议器”，不调用单帧入口，也不把单帧结果直接升级为运动。
    """

    if not frame_analyses:
        return (), float(min_residual_adu)
    if len(frame_analyses) < 2:
        raise ValueError(
            "detect_motion_features is a multi-frame detector; "
            "use detect_long_trails for one frame"
        )
    if len(cumulative_shifts) != len(frame_analyses):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    if psf_fwhm <= 0:
        raise ValueError("psf_fwhm must be positive")
    if min_length_px is not None and min_length_px <= 0:
        raise ValueError("min_length_px must be positive")
    if raw_fallback_workers < 1:
        raise ValueError("raw_fallback_workers must be positive")

    frame_times = _observation_times_seconds(frame_analyses)
    reference_interval_s = _reference_time_interval_seconds(frame_times)
    # 只有完整的 DATE-OBS + EXPOSURE 元数据才启用“线内速度 ↔ 帧间
    # 位移”的关联前门。旧合成样例和不完整 FITS 继续走几何关联，不把
    # 缺失元数据误当成静止目标。
    declared_exposures_s = tuple(
        (
            float(exposure_ms) / 1000.0
            if (
                (exposure_ms := exposure_milliseconds(analysis.frame.header)) is not None
                and np.isfinite(float(exposure_ms))
                and float(exposure_ms) > 0.0
            )
            else None
        )
        for analysis in frame_analyses
    )
    exposure_link_metadata_supported = bool(
        all(
            _parse_observation_timestamp(analysis.frame.header.get("DATE-OBS")) is not None
            for analysis in frame_analyses
        )
        and all(value is not None for value in declared_exposures_s)
        and np.all(np.isfinite(frame_times))
        and np.all(np.diff(frame_times) > 0.0)
    )

    if progress is not None:
        progress(0.0, "准备线状残差筛选")
    if temporal_reference is None:
        temporal_reference = _registered_median_reference(
            frame_analyses,
            cumulative_shifts,
            progress=progress,
        )
    elif tuple(np.asarray(temporal_reference).shape) != tuple(int(value) for value in frame_analyses[0].frame.data.shape):
        # 预计算参考图尺寸不一致时安全回退，不让一个旧结果改变当前
        # 线状候选的检测坐标。
        temporal_reference = _registered_median_reference(
            frame_analyses,
            cumulative_shifts,
            progress=progress,
        )
    if progress is not None:
        progress(20.0, "完成时间中值参考")
    difference_noise_factor = np.sqrt(2.0) if temporal_reference is not None else 1.0
    min_pixels = max(int(min_feature_area), int(np.ceil(4.0 * psf_fwhm**2)))
    min_length = _sequence_motion_candidate_min_length(
        psf_fwhm=float(psf_fwhm),
        min_length_px=min_length_px,
        exposure_s=declared_exposures_s,
        metadata_supported=exposure_link_metadata_supported,
    )
    feature_frames: list[list[MotionFeaturePoint]] = []
    thresholds: list[float] = []
    frame_audits: list[MotionFrameAudit] = []

    for frame_index, (analysis, shift) in enumerate(zip(frame_analyses, cumulative_shifts, strict=True)):
        values = np.asarray(analysis.frame.data, dtype=np.float32)
        valid = np.isfinite(values)
        valid &= ~auxiliary_mask(values.shape)
        parameters = analysis.detection.parameters
        if int(parameters.get("mask_zero_pixels", 0)):
            valid &= values != 0.0
        saturation_level = float(parameters.get("saturation_level", -1.0))
        if saturation_level > 0:
            valid &= values < saturation_level
        line_artifact_mask = _line_artifact_mask_from_coordinates(
            analysis.detection.line_artifact_coordinates,
            values.shape,
        )
        background = float(analysis.detection.background)
        noise = max(float(analysis.detection.noise), np.finfo(np.float32).eps)
        frame_exposure_ms = exposure_milliseconds(analysis.frame.header)
        frame_exposure_s = (
            float(frame_exposure_ms) / 1000.0
            if frame_exposure_ms is not None
            and np.isfinite(float(frame_exposure_ms))
            and float(frame_exposure_ms) > 0.0
            else None
        )
        # 对已知的开运一号 1.5 s 曝光，把面积门从历史的 20 px
        # 降到保守的 12 px 提议门；未知曝光仍保持 20 px。这个门只
        # 负责让短线进入后续原图模型和跨帧关联，不能直接把它标成 moving。
        short_min_pixels = max(
            _exposure_aware_short_trail_min_area(
                psf_fwhm=float(psf_fwhm),
                exposure_s=frame_exposure_s,
            ),
            int(np.ceil(2.0 * psf_fwhm**2)),
        )
        residual_noise = noise * difference_noise_factor
        threshold = max(float(min_residual_adu), float(residual_sigma) * residual_noise)
        thresholds.append(threshold)
        if temporal_reference is None:
            residual = np.where(valid, values - background, 0.0)
        else:
            shift_x, shift_y = cumulative_shifts[frame_index]
            raw_reference = ndimage.shift(
                temporal_reference,
                shift=(float(shift_y), float(shift_x)),
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            residual = np.where(valid & np.isfinite(raw_reference), values - raw_reference, 0.0)
            del raw_reference
        support = valid & (residual >= threshold)
        valid_pixel_count = int(np.count_nonzero(valid))
        support_pixel_count = int(np.count_nonzero(support))
        # 单帧短拖影可能因像素量化或局部背景起伏在一条直线上断开一个
        # 像素。单帧预检可以先用一次 3×3 邻域膨胀做“相邻缺口”分组，
        # 但后续
        # 面积、PCA 和权重一律回到原始 support 像素，绝不把补出来的
        # 像素当成信号。序列确认层也允许显式开启缺口桥接；桥接只用于
        # 连通域分组，不会把补出来的像素计入面积、PCA 或通量。
        geometry_support = (
            ndimage.binary_dilation(support, structure=np.ones((3, 3), dtype=bool))
            if bridge_gaps
            else support
        )
        labels, component_count = ndimage.label(geometry_support, structure=np.ones((3, 3), dtype=bool))
        slices = ndimage.find_objects(labels)
        sizes = ndimage.sum(support, labels, index=np.arange(1, component_count + 1))
        frame_features: list[MotionFeaturePoint] = []
        area_pass_count = 0
        geometry_pass_count = 0
        edge_rejected_count = 0
        short_area_rejected_count = 0
        length_rejected_count = 0
        axis_ratio_rejected_count = 0
        large_area_rejected_count = 0
        for component_id, size in enumerate(sizes, start=1):
            if float(size) < short_min_pixels:
                short_area_rejected_count += 1
                continue
            area_pass_count += 1
            component_slice = slices[component_id - 1]
            if component_slice is None:
                continue
            ys, xs = component_slice
            component = labels[component_slice] == component_id
            component_support = support[component_slice] & component
            local_y, local_x = np.nonzero(component_support)
            if local_x.size < 3:
                continue
            raw_x = local_x + xs.start
            raw_y = local_y + ys.start
            covariance = np.cov(np.column_stack((raw_x, raw_y)), rowvar=False, bias=True)
            eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
            major_variance = max(0.0, float(eigenvalues[-1]))
            minor_variance = max(0.0, float(eigenvalues[0]))
            length_px = 4.0 * np.sqrt(major_variance)
            width_px = 4.0 * np.sqrt(minor_variance)
            axis_ratio = np.sqrt(major_variance / max(minor_variance, np.finfo(np.float64).eps))
            axis_ratio_gate = float(min_axis_ratio)
            if length_px < MOTION_FEATURE_SHORT_TRAIL_LENGTH:
                # 短线先允许较低轴比进入候选池，但不会绕过后面的
                # 0.80/0.80 逐帧形态门以及跨帧位移/拟合门。
                axis_ratio_gate = min(
                    axis_ratio_gate,
                    MOTION_FEATURE_SHORT_TRAIL_MIN_AXIS_RATIO,
                )
            if length_px < min_length:
                length_rejected_count += 1
                continue
            if axis_ratio < axis_ratio_gate:
                axis_ratio_rejected_count += 1
                continue
            if length_px >= MOTION_FEATURE_SHORT_TRAIL_LENGTH and float(size) < min_pixels:
                large_area_rejected_count += 1
                continue
            geometry_pass_count += 1
            x_min, x_max = int(raw_x.min()), int(raw_x.max())
            y_min, y_max = int(raw_y.min()), int(raw_y.max())
            touches_edge = x_min == 0 or y_min == 0 or x_max == values.shape[1] - 1 or y_max == values.shape[0] - 1
            # 小型边缘横线通常是数组边界/截断异常；保留足够长的真实拖影，
            # 但不把几十个像素的边缘噪声当成目标。
            if touches_edge and length_px < max(min_length, 8.0 * psf_fwhm):
                edge_rejected_count += 1
                continue
            component_values = residual[component_slice][component_support]
            component_artifact = line_artifact_mask[component_slice][component_support]
            artifact_overlap_fraction = (
                float(np.count_nonzero(component_artifact) / component_values.size)
                if component_values.size
                else 0.0
            )
            weights = np.maximum(component_values.astype(np.float64), 0.0)
            weight_sum = float(weights.sum())
            center_x = float(np.average(raw_x, weights=weights)) if weight_sum > 0 else float(raw_x.mean())
            center_y = float(np.average(raw_y, weights=weights)) if weight_sum > 0 else float(raw_y.mean())
            major_vector = eigenvectors[:, -1]
            angle_deg = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])))
            residual_peak = float(component_values.max())
            continuity_ratio, profile_fill_ratio = _line_profile_metrics(
                residual,
                center_x=center_x,
                center_y=center_y,
                length_px=length_px,
                width_px=width_px,
                angle_deg=angle_deg,
                threshold_adu=threshold,
            )
            point = MotionFeaturePoint(
                frame_index=frame_index,
                x=center_x,
                y=center_y,
                aligned_x=center_x - float(shift[0]),
                aligned_y=center_y - float(shift[1]),
                residual_snr=residual_peak / residual_noise,
                area_pixels=int(size),
                length_px=float(length_px),
                width_px=float(width_px),
                angle_deg=angle_deg,
                bbox=(x_min, y_min, x_max, y_max),
                touches_edge=touches_edge,
                continuity_ratio=continuity_ratio,
                profile_fill_ratio=profile_fill_ratio,
                line_artifact_overlap_fraction=artifact_overlap_fraction,
            )
            # The temporal residual is only a proposal image.  Its raw-frame
            # model evidence is attached after the per-frame raw fallback has
            # finished, so the same local point/double/ellipse fit is not
            # repeated once per temporal candidate.
            frame_features.append(point)
        feature_frames.append(frame_features)
        frame_audits.append(
            MotionFrameAudit(
                frame_index=frame_index,
                threshold_adu=float(threshold),
                residual_noise_adu=float(residual_noise),
                valid_pixel_count=valid_pixel_count,
                support_pixel_count=support_pixel_count,
                component_count=int(component_count),
                area_pass_count=area_pass_count,
                geometry_pass_count=geometry_pass_count,
                edge_rejected_count=edge_rejected_count,
                feature_count=len(frame_features),
                max_feature_residual_snr=(
                    max((float(point.residual_snr) for point in frame_features), default=None)
                ),
                short_area_rejected_count=short_area_rejected_count,
                length_rejected_count=length_rejected_count,
                axis_ratio_rejected_count=axis_ratio_rejected_count,
                large_area_rejected_count=large_area_rejected_count,
                short_min_pixels=int(short_min_pixels),
            )
        )
        if progress is not None:
            progress(
                20.0 + 60.0 * (frame_index + 1) / len(frame_analyses),
                f"线状残差筛选 F{frame_index + 1:02d}/{len(frame_analyses)}",
            )

    # 时间中值参考对“每帧移动距离不大、但 1.5 s 曝光内已经拉成短线”
    # 的目标可能过度吸收：它们在某些帧的线形残差会消失。用独立单帧
    # 形态检测做补充。补充点有两种准入方式：
    # 1) 靠近时间残差锚点；或 2) 自身与相邻帧组成非零位移链。
    # 第二种方式专门救回“整条短线都被时间中值吸收”的目标，同时要求
    # 至少三帧的相邻位移和至少两个强形态点，避免静态长线/双星只凭形状进入。
    raw_fallback_point_ids: set[int] = set()
    raw_feature_frames: list[list[MotionFeaturePoint]] = [[] for _ in frame_analyses]
    if progress is not None:
        progress(80.0, "补充单帧线形候选")

    def extract_raw_frame_features(
        item: tuple[int, FrameAnalysis, tuple[float, float]],
    ) -> tuple[int, list[MotionFeaturePoint]]:
        frame_index, analysis, shift = item
        raw_exposure_ms = exposure_milliseconds(analysis.frame.header)
        raw_exposure_s = (
            float(raw_exposure_ms) / 1000.0
            if raw_exposure_ms is not None
            and np.isfinite(float(raw_exposure_ms))
            and float(raw_exposure_ms) > 0.0
            else None
        )
        raw_tracks, _raw_threshold = _extract_single_frame_trail_candidates(
            analysis,
            psf_fwhm=psf_fwhm,
            residual_sigma=DEFAULT_SINGLE_FRAME_TRAIL_RESIDUAL_SIGMA,
            min_residual_adu=DEFAULT_SINGLE_FRAME_TRAIL_MIN_RESIDUAL_ADU,
            min_feature_area=DEFAULT_SINGLE_FRAME_TRAIL_MIN_AREA,
            min_axis_ratio=DEFAULT_SINGLE_FRAME_TRAIL_MIN_AXIS_RATIO,
            # 与序列线残差入口使用同一个曝光感知下限。旧代码这里固定
            # 为 8 px，会让 1.5 s/FWHM 2 px 时已经允许进入序列候选池的
            # 6--7 px 短线，在原图补检阶段再次消失。
            min_length_px=min_length,
            bridge_gaps=DEFAULT_SINGLE_FRAME_TRAIL_BRIDGE_GAPS,
            progress=None,
            audit_sink=None,
            # 这条补充通道读取的是当前帧原图，不是时间差分。模型竞争
            # 会在与多尺度提案合并后统一做一次；这里先只提取几何候选，
            # 避免原图候选在“提取”和“合并后回到原图复核”两个位置重复
            # 运行单星/双星/椭圆/有限线段模型。最终进入关联器的点仍然
            # 必须通过下面同一套原图模型门，不会放宽相邻双星的负例控制。
            fit_line_model=False,
            short_min_feature_area=_exposure_aware_short_trail_min_area(
                psf_fwhm=float(psf_fwhm),
                exposure_s=raw_exposure_s,
            ),
        )
        raw_points = [
            replace(
                raw_track.points[0],
                frame_index=frame_index,
                aligned_x=float(raw_track.points[0].x) - float(shift[0]),
                aligned_y=float(raw_track.points[0].y) - float(shift[1]),
            )
            for raw_track in raw_tracks
            if raw_track.points
        ]
        if multiscale_line_proposals:
            multiscale_points = _extract_multiscale_motion_frame_features(
                analysis,
                frame_index=frame_index,
                shift=shift,
                psf_fwhm=psf_fwhm,
                min_length_px=min_length,
                temporal_reference=temporal_reference,
            )
            raw_points = _merge_motion_feature_points(raw_points, multiscale_points)
        # 这里只返回几何候选。原图提案和多尺度提案合并后，统一由下方
        # ``score_raw_frame_points`` 回到当前帧原图做一次单星/双星/椭圆/
        # 线段模型竞争；不能在这里先复核、合并后再复核一次。
        return frame_index, raw_points

    raw_work_items = tuple(
        (frame_index, analysis, shift)
        for frame_index, (analysis, shift) in enumerate(
            zip(frame_analyses, cumulative_shifts, strict=True)
        )
    )
    raw_worker_count = min(int(raw_fallback_workers), len(raw_work_items))
    if raw_worker_count == 1:
        completed_raw_items = map(extract_raw_frame_features, raw_work_items)
        for frame_index, raw_points in completed_raw_items:
            raw_feature_frames[frame_index] = raw_points
            if progress is not None:
                progress(
                    80.0 + 2.0 * (frame_index + 1) / len(frame_analyses),
                    f"补充单帧线形候选 F{frame_index + 1:02d}/{len(frame_analyses)}",
                )
    else:
        # 这一步只补充每帧原图的线形 proposal，之前在这里按帧串行运行，
        # 会把 15 张 4096² 图重新做一遍昂贵的连通域/PCA。它没有共享可变
        # 状态，按帧并行后再按 frame_index 写回，结果顺序和单线程完全一致。
        # worker 数由调用方控制；序列默认使用不超过 2 个，避免与前面的
        # 源级检测同时占满内存带宽。
        with ThreadPoolExecutor(max_workers=raw_worker_count, thread_name_prefix="rst19-trail") as executor:
            completed_raw_items = executor.map(extract_raw_frame_features, raw_work_items)
            for frame_index, raw_points in completed_raw_items:
                raw_feature_frames[frame_index] = raw_points
                if progress is not None:
                    progress(
                        80.0 + 2.0 * (frame_index + 1) / len(frame_analyses),
                        f"补充单帧线形候选 F{frame_index + 1:02d}/{len(frame_analyses)}",
                    )

    # 把时间残差候选对回同一帧原图的模型结果。原图补充候选的几何扫描
    # 很便宜，但局部单星/双星/椭圆/线段竞争很贵；先完成几何候选与多
    # 尺度提案合并，再按帧只做一次原图模型复核。这样保留同一套双星
    # 负例门，同时避免候选在“几何提取”和“合并后回原图”两个位置重复
    # 建立局部模板。
    def score_raw_frame_points(
        frame_index: int,
        frame_points: Sequence[MotionFeaturePoint],
    ) -> list[MotionFeaturePoint]:
        """Score raw points once after all proposal sources have been merged.

        Build the per-frame residual once.  The previous implementation built
        model evidence for every geometric point and then scored the same
        points again after the multiscale merge; the deferred path keeps the
        identical raw-pixel model while avoiding the previous duplicate pass.
        """

        analysis = frame_analyses[frame_index]
        raw_values = np.asarray(analysis.frame.data, dtype=np.float32)
        raw_parameters = analysis.detection.parameters
        raw_valid = np.isfinite(raw_values)
        raw_valid &= ~auxiliary_mask(raw_values.shape)
        if int(raw_parameters.get("mask_zero_pixels", 0)):
            raw_valid &= raw_values != 0.0
        raw_saturation = float(raw_parameters.get("saturation_level", -1.0))
        if raw_saturation > 0:
            raw_valid &= raw_values < raw_saturation
        raw_residual = np.where(
            raw_valid,
            raw_values - float(analysis.detection.background),
            0.0,
        )
        raw_noise = max(float(analysis.detection.noise), np.finfo(np.float32).eps)
        return [
            _apply_single_frame_model_evidence(
                point,
                residual=raw_residual,
                valid=raw_valid,
                psf_fwhm=float(psf_fwhm),
                noise=raw_noise,
            )
            for point in frame_points
        ]

    # 保留未复核的几何点作为审计负例，但不给它们伪造模型字段；下面的
    # model-supported 过滤会自然丢弃这些点，靠近锚点的点才做昂贵复核。
    # 每帧评分只读当前帧图像和候选列表，和前面的原图几何提取一样可以
    # 按帧并行；结果按 frame_index 写回，保证候选顺序和审计口径不变。
    if raw_worker_count == 1:
        raw_model_feature_frames = []
        for completed, (frame_index, frame_points) in enumerate(
            enumerate(raw_feature_frames),
            start=1,
        ):
            raw_model_feature_frames.append(score_raw_frame_points(frame_index, frame_points))
            if progress is not None:
                progress(
                    82.0 + 2.0 * completed / len(raw_feature_frames),
                    f"原图线形模型复核 F{frame_index + 1:02d}/{len(raw_feature_frames)}",
                )
    else:
        score_items = tuple(enumerate(raw_feature_frames))
        with ThreadPoolExecutor(max_workers=raw_worker_count, thread_name_prefix="rst19-trail-score") as executor:
            raw_model_feature_frames = []
            for completed, (frame_index, frame_points) in enumerate(
                zip(
                    (item[0] for item in score_items),
                    executor.map(
                        lambda item: score_raw_frame_points(item[0], item[1]),
                        score_items,
                    ),
                    strict=True,
                ),
                start=1,
            ):
                raw_model_feature_frames.append(frame_points)
                if progress is not None:
                    progress(
                        82.0 + 2.0 * completed / len(raw_feature_frames),
                        f"原图线形模型复核 F{frame_index + 1:02d}/{len(raw_feature_frames)}",
                    )

    # 把时间残差候选对回同一帧原图的模型结果。这里只做空间 NMS 式
    # 匹配和字段复制，不再对每个 temporal point 重新拟合局部模板。
    def raw_model_match(
        point: MotionFeaturePoint,
        raw_points: Sequence[MotionFeaturePoint],
    ) -> MotionFeaturePoint | None:
        point_geometry = (
            float(point.x),
            float(point.y),
            float(point.length_px),
            float(point.width_px),
            float(point.angle_deg),
        )
        matches = []
        for raw_point in raw_points:
            raw_geometry = (
                float(raw_point.x),
                float(raw_point.y),
                float(raw_point.length_px),
                float(raw_point.width_px),
                float(raw_point.angle_deg),
            )
            if not _single_frame_line_geometries_overlap(point_geometry, raw_geometry):
                continue
            center_distance = float(np.hypot(point.x - raw_point.x, point.y - raw_point.y))
            length_difference = abs(float(point.length_px) - float(raw_point.length_px))
            matches.append((center_distance, length_difference, raw_point))
        if not matches:
            return None
        return min(matches, key=lambda item: (item[0], item[1]))[2]

    # 送入关联器的 raw feature 保留“强线”以及有明确弱线证据的 review
    # 点；双星/点源不会进入后续运动拟合。review 点只是为了不在关联前
    # 丢失短线，最终 moving 仍由轨迹级门控决定。
    for frame_index, frame_points in enumerate(feature_frames):
        checked_points: list[MotionFeaturePoint] = []
        for point in frame_points:
            model_point = raw_model_match(point, raw_model_feature_frames[frame_index])
            if model_point is None:
                # 时间差分中存在、但原图没有可回溯线形证据的结构，不能
                # 仅凭差分 PCA 进入运动关联；它会留在逐帧审计的几何计数中。
                continue
            checked_point = replace(
                point,
                line_template_snr=model_point.line_template_snr,
                line_vs_point_delta_bic=model_point.line_vs_point_delta_bic,
                line_vs_double_delta_bic=model_point.line_vs_double_delta_bic,
                line_vs_background_delta_bic=model_point.line_vs_background_delta_bic,
                line_vs_elliptical_delta_bic=model_point.line_vs_elliptical_delta_bic,
                single_frame_model=model_point.single_frame_model,
                single_frame_model_margin_bic=model_point.single_frame_model_margin_bic,
                # 这些三个字段必须来自同一张当前帧原图。时间差分的
                # 连续度/伪影重叠会受到配准残差影响，不能一边用原图
                # 做双星负例，一边继续用差分图的形态指标放行。
                continuity_ratio=model_point.continuity_ratio,
                profile_fill_ratio=model_point.profile_fill_ratio,
                line_artifact_overlap_fraction=model_point.line_artifact_overlap_fraction,
            )
            if _single_frame_sequence_line_model_is_review_supported(
                checked_point,
                psf_fwhm=float(psf_fwhm),
            ):
                checked_points.append(checked_point)
        feature_frames[frame_index] = checked_points

    # 原图负例仍保留在 raw_model_feature_frames 做解释，但只有通过强线
    # 或 review 模型门槛的线形进入“补充候选/共识/跨帧关联”。
    raw_feature_frames = [
        [
            point
            for point in frame_points
            if _single_frame_sequence_line_model_is_review_supported(
                point,
                psf_fwhm=float(psf_fwhm),
            )
        ]
        for frame_points in raw_model_feature_frames
    ]
    # 过滤掉 temporal-only 的未模型化候选后再计算锚点，避免被淘汰的
    # 双星残差继续作为宽距离补充准入的“锚”。
    temporal_anchors = [
        point
        for frame_points in feature_frames
        for point in frame_points
    ]
    coherent_raw_point_ids = _coherent_raw_motion_point_ids(
        raw_feature_frames,
        psf_fwhm=psf_fwhm,
        frame_times=frame_times,
        exposure_times_s=tuple(
            exposure_seconds(
                analysis.frame.header,
                default=reference_interval_s,
            )
            for analysis in frame_analyses
        ),
    )

    # 保留那些已经通过原图线形模型、但没有时序锚点的点。它们不能进入
    # 15 帧关联器，否则静态仪器线/单帧噪声可能被误解释成轨迹；但也不
    # 应该在序列结果中消失。后面会把它们作为独立的
    # ``single_frame_trail`` 审计候选返回，供当前帧叠加和悬停说明使用。
    unassociated_single_frame_points: list[MotionFeaturePoint] = []

    for frame_index, (raw_points, _shift) in enumerate(zip(raw_feature_frames, cumulative_shifts, strict=True)):
        for raw_point in raw_points:
            review_model = _single_frame_sequence_line_model_is_review_supported(
                raw_point,
                psf_fwhm=float(psf_fwhm),
            )
            anchor_supported = False
            if temporal_anchors:
                # temporal_anchors 已经在配准坐标中；24 px 只作“同一
                # 局部运动证据”的宽搜索，不直接当成轨迹关联半径。
                anchor_supported = min(
                    float(np.hypot(raw_point.aligned_x - anchor.aligned_x, raw_point.aligned_y - anchor.aligned_y))
                    for anchor in temporal_anchors
                ) <= max(24.0, 6.0 * psf_fwhm)
            if not review_model and id(raw_point) not in coherent_raw_point_ids:
                continue
            if not anchor_supported and id(raw_point) not in coherent_raw_point_ids:
                unassociated_single_frame_points.append(raw_point)
                continue
            # 同一帧的时间残差候选优先；补充点只填空缺，不复制
            # 一条已有的时序候选。
            duplicate = any(
                float(np.hypot(raw_point.aligned_x - existing.aligned_x, raw_point.aligned_y - existing.aligned_y))
                <= max(6.0, 0.25 * min(float(raw_point.length_px), float(existing.length_px)))
                for existing in feature_frames[frame_index]
            )
            if duplicate:
                continue
            feature_frames[frame_index].append(raw_point)
            raw_fallback_point_ids.add(id(raw_point))

    # 审计中的 feature_count 表示实际送入关联器的候选数。单帧形态
    # 补充层只填补时间参考漏掉的候选，因此在补充完成后回写数量；
    # 阈值、连通域和 geometry_pass_count 仍保留时间残差层的原始口径。
    for frame_index, frame_points in enumerate(feature_frames):
        if not frame_points:
            continue
        audit = frame_audits[frame_index]
        frame_audits[frame_index] = replace(
            audit,
            feature_count=len(frame_points),
            max_feature_residual_snr=max(
                (float(point.residual_snr) for point in frame_points),
                default=None,
            ),
        )

    # 当前真实数据中线状目标的帧间位移约为几十像素；半径过小会漏掉，
    # 过大又会把不同边缘伪迹串成一条轨迹。这个半径必须显式进入结果，
    # 不能再用点源的 ``link_radius_px`` 覆盖后静默改变。
    resolved_line_link_radius_px = (
        max(48.0, 32.0 * float(psf_fwhm))
        if line_link_radius_px is None
        else float(line_link_radius_px)
    )
    if not np.isfinite(resolved_line_link_radius_px) or resolved_line_link_radius_px <= 0.0:
        raise ValueError("line_link_radius_px must be positive")
    if progress is not None:
        progress(82.0, "关联线状候选")
    tracks: list[list[MotionFeaturePoint]] = []
    # 短曝光线在时间中值残差里可能连续漏掉两帧；真实 F01--F15 的
    # T3(F05--F08) 与 T5(F11--F14) 就是同一条物理路径被 F09--F10
    # 吸收后的两段。允许到 3 的 frame_gap 只意味着最多跳过两帧，且
    # 在完整头部元数据存在时还必须通过曝光速度门；缺少元数据时仍有
    # 常速度拟合、形态多数和最终持续性门兜底。
    max_frame_gap = 3
    for frame_index, current in enumerate(feature_frames):
        if not current:
            if progress is not None:
                progress(
                    82.0 + 12.0 * (frame_index + 1) / len(feature_frames),
                    f"关联线状候选 F{frame_index + 1:02d}/{len(feature_frames)}（无候选）",
                )
            continue
        predicted_tracks: list[tuple[int, float, float]] = []
        # 每个可预测轨迹保存“上一个曝光中心”和由上一帧线长推导出的
        # 期望帧间位移。回调只在半径邻域内运行，所以不会把 4096² 源场
        # 变成全量 pairwise 计算。
        exposure_link_gates: dict[int, tuple[float, float, float, float]] = {}
        for track_index, history in enumerate(tracks):
            last = history[-1]
            frame_gap = frame_index - last.frame_index
            if frame_gap < 1 or frame_gap > max_frame_gap:
                continue
            elapsed_s = float(frame_times[frame_index] - frame_times[last.frame_index])
            if elapsed_s <= 0.0:
                continue
            if len(history) >= 2 and history[-2].frame_index < last.frame_index:
                previous = history[-2]
                previous_elapsed_s = float(frame_times[last.frame_index] - frame_times[previous.frame_index])
                if previous_elapsed_s <= 0.0:
                    continue
                step_x = (last.aligned_x - previous.aligned_x) / previous_elapsed_s
                step_y = (last.aligned_y - previous.aligned_y) / previous_elapsed_s
                predicted_x = last.aligned_x + step_x * elapsed_s
                predicted_y = last.aligned_y + step_y * elapsed_s
            else:
                predicted_x, predicted_y = last.aligned_x, last.aligned_y
            predicted_tracks.append((track_index, predicted_x, predicted_y))
        current_points = np.asarray(
            [(point.aligned_x, point.aligned_y) for point in current],
            dtype=np.float64,
        )
        # 线状候选也可能在端点或相邻结构处靠近；使用同一局部全局
        # 分配器，避免“先到先得”让一条真实线被拆成多条候选轨迹。
        if exposure_link_metadata_supported:
            for track_index, history in enumerate(tracks):
                if not history:
                    continue
                last = history[-1]
                last_index = int(last.frame_index)
                if last_index < 0 or last_index >= len(declared_exposures_s):
                    continue
                exposure_s = declared_exposures_s[last_index]
                if exposure_s is None:
                    continue
                elapsed_s = float(frame_times[frame_index] - frame_times[last_index])
                motion_length = _deconvolved_exposure_motion_length(
                    last.length_px,
                    float(psf_fwhm),
                )
                if (
                    not np.isfinite(elapsed_s)
                    or elapsed_s <= 0.0
                    or motion_length is None
                    or not np.isfinite(float(motion_length))
                    or float(motion_length) <= 0.0
                ):
                    continue
                expected_step = float(motion_length) * elapsed_s / float(exposure_s)
                if not np.isfinite(expected_step) or expected_step <= 0.0:
                    continue
                exposure_link_gates[track_index] = (
                    float(last.aligned_x),
                    float(last.aligned_y),
                    expected_step,
                    elapsed_s,
                )

        def pair_allowed(track_index: int, source_index: int, _distance: float) -> bool:
            gate = exposure_link_gates.get(int(track_index))
            if gate is None:
                return True
            last_x, last_y, expected_step, _elapsed_s = gate
            observed_step = float(
                np.hypot(
                    float(current_points[source_index, 0]) - last_x,
                    float(current_points[source_index, 1]) - last_y,
                )
            )
            if not np.isfinite(observed_step):
                return False
            ratio = observed_step / max(expected_step, np.finfo(np.float64).eps)
            return bool(
                MOTION_FEATURE_LINK_EXPOSURE_RATIO_MIN <= ratio <= MOTION_FEATURE_LINK_EXPOSURE_RATIO_MAX
            )

        def pair_geometry_cost(
            track_index: int,
            source_index: int,
            _distance: float,
        ) -> float:
            history = tracks[int(track_index)]
            if not history:
                return 0.0
            return _motion_feature_pair_geometry_penalty(
                history[-1],
                current[int(source_index)],
                psf_fwhm=float(psf_fwhm),
                expected_step_px=(
                    float(exposure_link_gates[int(track_index)][2])
                    if int(track_index) in exposure_link_gates
                    else None
                ),
            )

        possible = _global_frame_assignments(
            predicted_tracks,
            current_points,
            link_radius_px=resolved_line_link_radius_px,
            pair_allowed=pair_allowed,
            pair_cost=pair_geometry_cost,
        )
        assigned_tracks: set[int] = set()
        assigned_sources: set[int] = set()
        for _distance, track_index, source_index in possible:
            if track_index in assigned_tracks or source_index in assigned_sources:
                continue
            tracks[track_index].append(current[source_index])
            assigned_tracks.add(track_index)
            assigned_sources.add(source_index)
        for source_index, point in enumerate(current):
            if source_index not in assigned_sources:
                tracks.append([point])
        if progress is not None:
            progress(
                82.0 + 12.0 * (frame_index + 1) / len(feature_frames),
                f"关联线状候选 F{frame_index + 1:02d}/{len(feature_frames)} · {len(tracks)} 条轨迹",
            )

    # 局部关联完成后，把严格可预测的单帧短线接回已有轨迹。单帧补充
    # 点有两种来源：一类已经在 ``unassociated_single_frame_points`` 中，
    # 另一类在首轮关联时形成了 singleton track。两类都必须进入同一个
    # 回接器；此前只处理前者，会让“已经通过原图线模型、但首轮没有邻
    # 点可接”的 singleton 永远停在审计层。
    stitch_points = list(unassociated_single_frame_points)
    retained_tracks: list[list[MotionFeaturePoint]] = []
    for history in tracks:
        if len(history) == 1:
            singleton = history[0]
            singleton_is_stitchable = (
                float(singleton.length_px) < max(32.0, 8.0 * float(psf_fwhm))
                and
                _single_frame_sequence_line_model_is_review_supported(
                    singleton,
                    psf_fwhm=float(psf_fwhm),
                )
                and (
                    singleton.line_artifact_overlap_fraction is None
                    or float(singleton.line_artifact_overlap_fraction)
                    < SINGLE_FRAME_ARTIFACT_REVIEW_OVERLAP_FRACTION
                )
            )
            if singleton_is_stitchable:
                stitch_points.append(singleton)
                continue
        retained_tracks.append(history)
    before_stitch_ids = {id(point) for point in stitch_points}
    tracks, remaining_single_frame_points = _stitch_unassociated_single_frame_points(
        retained_tracks,
        stitch_points,
        psf_fwhm=float(psf_fwhm),
        frame_times=frame_times,
    )
    stitched_ids = before_stitch_ids - {
        id(point) for point in remaining_single_frame_points
    }
    raw_fallback_point_ids.update(stitched_ids)
    unassociated_single_frame_points = list(remaining_single_frame_points)

    # 已有两段强短线轨迹后，只在缺帧的预测中心附近回到原始像素复核。
    # 这一步不把外推位置当成观测；只有局部线模型通过才会追加一个
    # ``recovered_by_predicted_roi`` 点。
    recovered_point_ids = _recover_predicted_roi_motion_feature_points(
        tracks,
        frame_analyses,
        cumulative_shifts,
        frame_times,
        psf_fwhm=float(psf_fwhm),
    )
    raw_fallback_point_ids.update(recovered_point_ids)

    # 质心关联适合逐帧位移不大的点/短线；高速目标可能在每帧留下
    # 相隔数百像素的线段，前面的邻近门控会把它们各自保留为 singleton。
    # 只对足够长的 singleton 做一次几何补救，短斜线和边缘小结构不会
    # 因为“方向看起来相似”而被升级。已被中心关联形成多点轨迹的候选
    # 不重复进入这一层。
    rescued_groups = _rescue_collinear_motion_feature_groups(
        tracks,
        psf_fwhm=psf_fwhm,
        frame_times=frame_times,
        reference_interval_s=reference_interval_s,
    )
    rescued_point_ids = {
        id(point)
        for group in rescued_groups
        for point in group
    }
    if rescued_point_ids:
        tracks = [
            history
            for history in tracks
            if not (len(history) == 1 and id(history[0]) in rescued_point_ids)
        ]
        rescued_track_start = len(tracks)
        tracks.extend([list(group) for group in rescued_groups])
        rescued_track_indices = set(range(rescued_track_start, len(tracks)))
    else:
        rescued_track_indices = set()

    rendered: list[MotionFeatureTrack] = []
    # ``min_presence`` in the sequence summary is the stable-source门槛，
    # 不能拿来代替线状运动确认门。对 15 帧数据，moving 至少覆盖一半
    # 观测帧；3--7 帧的共线结果仍返回，但只标为 candidate，避免把
    # 偶然的短时段直线或仪器状态变化说成已经确认的运动目标。
    feature_min_presence = _motion_feature_confirmation_presence(len(frame_analyses))
    feature_min_displacement = max(20.0, 6.0 * psf_fwhm)
    feature_max_rms = max(4.0, 1.5 * psf_fwhm)
    frame_shapes = tuple(
        tuple(int(value) for value in analysis.frame.data.shape)
        for analysis in frame_analyses
    )
    if progress is not None:
        progress(94.0, "拟合运动轨迹")
    for track_id, points in enumerate(tracks):
        shape_supported = _motion_feature_track_shape_is_supported(points)
        model_supported = _motion_feature_track_model_is_supported(
            points,
            psf_fwhm=float(psf_fwhm),
        )
        motion_axis_supported = _motion_feature_motion_axis_is_supported(points)
        edge_truncated = _motion_feature_edge_truncation_is_supported(
            points,
            frame_shapes,
        )
        presence_supported = len(points) >= feature_min_presence or (
            edge_truncated
            and len(points)
            >= max(
                3,
                feature_min_presence - MOTION_FEATURE_EDGE_MIN_PRESENCE_RELAXATION,
            )
        )
        temporal_continuity_supported = _motion_feature_temporal_continuity_is_supported(
            points,
            min_presence=feature_min_presence,
            edge_truncated=edge_truncated,
        )
        exposure_velocity_ratio, exposure_velocity_log_mad, exposure_consistency_fraction = (
            _motion_feature_exposure_metrics(
                points,
                frame_analyses,
                frame_times,
                psf_fwhm=float(psf_fwhm),
            )
        )
        exposure_supported = _motion_feature_exposure_is_supported(
            exposure_velocity_ratio,
            exposure_velocity_log_mad,
            exposure_consistency_fraction,
        )
        artifact_supported = _motion_feature_artifact_is_supported_as_moving(points)
        internal_gap_bridge_supported = _motion_feature_internal_gap_bridge_is_supported(points)
        if track_id in rescued_track_indices:
            displacement, speed, geometry_rms, _angle = _fit_collinear_motion_geometry(
                points,
                frame_times=frame_times,
                reference_interval_s=reference_interval_s,
            )
            fit_rms = geometry_rms
            classification = (
                "moving"
                if presence_supported
                and displacement >= feature_min_displacement
                and geometry_rms is not None
                and geometry_rms <= 12.0
                and shape_supported
                and model_supported
                and motion_axis_supported
                and temporal_continuity_supported
                and exposure_supported
                and artifact_supported
                else "candidate"
            )
            evidence_level = "collinear_trail"
        else:
            displacement, speed, fit_rms = _fit_motion_feature_track(
                points,
                frame_times=frame_times,
                reference_interval_s=reference_interval_s,
            )
            classification = (
                "moving"
                if presence_supported
                and displacement >= feature_min_displacement
                and fit_rms is not None
                and fit_rms <= feature_max_rms
                and shape_supported
                and model_supported
                and motion_axis_supported
                and temporal_continuity_supported
                and exposure_supported
                and artifact_supported
                else "candidate"
            )
            geometry_rms = None
            evidence_level = "center_motion"
        strict_gap_bridge = bool(
            not temporal_continuity_supported
            and internal_gap_bridge_supported
            and presence_supported
            and shape_supported
            and motion_axis_supported
            and exposure_supported
            and artifact_supported
            and displacement >= feature_min_displacement
            and fit_rms is not None
            and fit_rms <= (12.0 if track_id in rescued_track_indices else feature_max_rms)
            and all(
                _single_frame_sequence_line_model_is_review_supported(
                    point,
                    psf_fwhm=float(psf_fwhm),
                )
                for point in points
            )
        )
        artifact_motion_override = (
            not artifact_supported
            and _motion_feature_artifact_overlap_is_compatible_with_motion(
                points,
                displacement_px=displacement,
                fit_rms_px=fit_rms,
                max_fit_rms_px=(12.0 if track_id in rescued_track_indices else feature_max_rms),
                presence_supported=presence_supported,
                min_displacement_px=feature_min_displacement,
                shape_supported=shape_supported,
                motion_axis_supported=motion_axis_supported,
                temporal_continuity_supported=temporal_continuity_supported,
                exposure_supported=exposure_supported,
                model_supported=model_supported,
            )
        )
        # 通用直线掩膜会命中真实曝光拖影本身。只有完整运动证据已经通过
        # 时，才允许覆盖“高重叠”这一项；单帧线、低拟合质量或缺少曝光
        # 元数据的轨迹仍不能靠这个例外升级。
        if artifact_motion_override:
            classification = "moving"
        if strict_gap_bridge:
            # 两段观测虽然中间缺了帧，但只有在上面的全部严格条件都
            # 通过时才把它当作同一条曝光运动轨迹；这里不能只追加说明
            # 字段，否则会出现“证据说已桥接、分类仍是 candidate”的矛盾。
            classification = "moving"
            evidence_level = f"{evidence_level}+interior_gap_bridged"
        if edge_truncated:
            # 这是“边界截断”证据标签，而不是把边界外的未知位置补成
            # 观测。报告/UI 仍应把它与完整视场内的 moving 区分开。
            evidence_level = f"{evidence_level}+edge_truncated"
        if not artifact_supported:
            evidence_level = (
                f"{evidence_level}+line_artifact_overlap_reviewed"
                if artifact_motion_override
                else f"{evidence_level}+line_artifact_overlap"
            )
        if any(id(point) in raw_fallback_point_ids for point in points):
            # 这条轨迹中至少有一个观测来自“单帧形态 + 时序锚点”的
            # 补充层；保留来源标签，避免报告把它误说成纯时间差分结果。
            evidence_level = f"{evidence_level}+single_frame_shape"
        if any(id(point) in stitched_ids for point in points):
            # 这些点是在初次局部关联结束后由严格常速度预测回接的；
            # 保留独立标签，避免报告把“回接”误读成首轮逐帧关联。
            evidence_level = f"{evidence_level}+single_frame_stitched"
        if any(id(point) in recovered_point_ids for point in points):
            # 这些点来自已拟合轨迹预测中心的小窗口原图复核；它们不是
            # 轨迹外推，也不能在报告里伪装成首轮逐帧检测。
            evidence_level = f"{evidence_level}+recovered_by_predicted_roi"
        rejection_reasons = _motion_feature_rejection_reasons(
            classification=classification,
            evidence_level=evidence_level,
            points=points,
            presence_supported=presence_supported,
            displacement_px=displacement,
            min_displacement_px=feature_min_displacement,
            fit_rms_px=fit_rms,
            max_fit_rms_px=(
                12.0 if track_id in rescued_track_indices else feature_max_rms
            ),
            shape_supported=shape_supported,
            model_supported=model_supported,
            motion_axis_supported=motion_axis_supported,
            temporal_continuity_supported=temporal_continuity_supported,
            exposure_supported=exposure_supported,
            artifact_supported=artifact_supported,
            edge_truncated=edge_truncated,
        )
        rendered.append(
            MotionFeatureTrack(
                track_id=track_id,
                classification=classification,
                points=tuple(points),
                displacement_px=displacement,
                speed_px_per_frame=speed,
                fit_rms_px=fit_rms,
                evidence_level=evidence_level,
                geometry_rms_px=geometry_rms,
                exposure_velocity_ratio=exposure_velocity_ratio,
                exposure_velocity_log_mad=exposure_velocity_log_mad,
                exposure_consistency_fraction=exposure_consistency_fraction,
                rejection_reasons=rejection_reasons,
            )
        )
    # 单帧线形证据和跨帧运动证据必须同时可见、但不能混为一种结论。
    # 这些点已经通过同一套原图线模型 review gate，仍然保持 candidate，
    # 位移/速度/RMS 明确置零/缺失，防止下游把它们算进 moving。
    next_track_id = len(rendered)
    rendered_point_geometries = [
        (
            int(point.frame_index),
            (
                float(point.x),
                float(point.y),
                float(point.length_px),
                float(point.width_px),
                float(point.angle_deg),
            ),
        )
        for track in rendered
        for point in track.points
    ]
    review_only_points: list[MotionFeaturePoint] = []
    for point in unassociated_single_frame_points:
        point_geometry = (
            float(point.x),
            float(point.y),
            float(point.length_px),
            float(point.width_px),
            float(point.angle_deg),
        )
        # A raw model point can miss the association object's identity even
        # though the same frame/geometry is already represented by a temporal
        # track.  Do not draw that proposal twice; this is display/result NMS,
        # not a relaxed admission rule.
        if any(
            frame_index == int(point.frame_index)
            and _single_frame_line_geometries_overlap(point_geometry, geometry)
            for frame_index, geometry in rendered_point_geometries
        ):
            continue
        review_only_points.append(point)
    for offset, point in enumerate(review_only_points):
        rendered.append(
            MotionFeatureTrack(
                track_id=next_track_id + offset,
                classification="candidate",
                points=(point,),
                displacement_px=0.0,
                speed_px_per_frame=0.0,
                fit_rms_px=None,
                evidence_level="single_frame_trail",
                rejection_reasons=("SINGLE_FRAME_ONLY", "NO_CROSS_FRAME_ASSOCIATION"),
            )
        )
    if progress is not None:
        progress(100.0, "线状目标筛选完成")
    if audit_sink is not None:
        audit_sink.extend(frame_audits)
    return tuple(rendered), float(np.median(thresholds)) if thresholds else float(min_residual_adu)


def _extract_single_frame_trail_candidates(
    frame_analysis: FrameAnalysis,
    *,
    psf_fwhm: float,
    residual_sigma: float,
    min_residual_adu: float,
    min_feature_area: int,
    min_axis_ratio: float,
    min_length_px: float | None,
    bridge_gaps: bool,
    progress: Callable[[float, str], None] | None,
    audit_sink: list[MotionFrameAudit] | None,
    fit_line_model: bool = False,
    short_min_feature_area: int | None = None,
) -> tuple[tuple[MotionFeatureTrack, ...], float]:
    """只对一张图做曝光内线形检测。

    这个函数刻意不接受帧序列、累计配准位移或时间参考图。它是单帧
    按钮的实际算法边界：当前图的 ADU → 当前图背景/噪声 → 正残差连通域
    → PCA 线形与剖面连续性 → ``candidate``。任何“moving”结论都不在
    这里产生。
    """

    if psf_fwhm <= 0 or residual_sigma <= 0 or min_residual_adu <= 0:
        raise ValueError("single-frame trail parameters must be positive")
    if min_feature_area < 3 or min_axis_ratio <= 0:
        raise ValueError("single-frame trail geometry parameters are invalid")
    if min_length_px is not None and min_length_px <= 0:
        raise ValueError("min_length_px must be positive")

    values = np.asarray(frame_analysis.frame.data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if progress is not None:
        # 保留既有进度标签，避免外部进度面板因实现拆分而失去兼容性；
        # 这里实际已经进入独立的单帧函数，并不代表调用多帧路线。
        progress(0.0, "准备线状残差筛选")

    valid = np.isfinite(values)
    valid &= ~auxiliary_mask(values.shape)
    parameters = frame_analysis.detection.parameters
    if int(parameters.get("mask_zero_pixels", 0)):
        valid &= values != 0.0
    saturation_level = float(parameters.get("saturation_level", -1.0))
    if saturation_level > 0:
        valid &= values < saturation_level
    line_artifact_mask = _line_artifact_mask_from_coordinates(
        frame_analysis.detection.line_artifact_coordinates,
        values.shape,
    )

    background = float(frame_analysis.detection.background)
    noise = max(float(frame_analysis.detection.noise), np.finfo(np.float32).eps)
    threshold = max(float(min_residual_adu), float(residual_sigma) * noise)
    residual = np.where(valid, values - background, 0.0)
    support = valid & (residual >= threshold)
    # 只用膨胀后的掩膜做“相邻缺口分组”；面积、PCA、峰值和剖面测量
    # 仍全部回到原始 support，避免人为补像素被当成真实信号。
    geometry_support = (
        ndimage.binary_dilation(support, structure=np.ones((3, 3), dtype=bool))
        if bridge_gaps
        else support
    )
    labels, component_count = ndimage.label(geometry_support, structure=np.ones((3, 3), dtype=bool))
    slices = ndimage.find_objects(labels)
    min_pixels = max(int(min_feature_area), int(np.ceil(4.0 * psf_fwhm**2)))
    short_area_floor = (
        MOTION_FEATURE_SHORT_TRAIL_MIN_AREA
        if short_min_feature_area is None
        else int(short_min_feature_area)
    )
    if short_area_floor < 3:
        raise ValueError("short_min_feature_area must be at least 3")
    short_min_pixels = max(
        short_area_floor,
        int(np.ceil(2.0 * psf_fwhm**2)),
    )
    min_length = (
        max(DEFAULT_SEQUENCE_TRAIL_MIN_LENGTH, 3.0 * psf_fwhm)
        if min_length_px is None
        else float(min_length_px)
    )
    frame_features: list[MotionFeaturePoint] = []
    area_pass_count = 0
    geometry_pass_count = 0
    edge_rejected_count = 0
    short_area_rejected_count = 0
    length_rejected_count = 0
    axis_ratio_rejected_count = 0
    large_area_rejected_count = 0

    sizes = ndimage.sum(support, labels, index=np.arange(1, component_count + 1))
    for component_id, size in enumerate(sizes, start=1):
        if float(size) < short_min_pixels:
            short_area_rejected_count += 1
            continue
        area_pass_count += 1
        component_slice = slices[component_id - 1]
        if component_slice is None:
            continue
        ys, xs = component_slice
        component = labels[component_slice] == component_id
        component_support = support[component_slice] & component
        local_y, local_x = np.nonzero(component_support)
        if local_x.size < 3:
            continue
        raw_x = local_x + xs.start
        raw_y = local_y + ys.start
        covariance = np.cov(np.column_stack((raw_x, raw_y)), rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
        major_variance = max(0.0, float(eigenvalues[-1]))
        minor_variance = max(0.0, float(eigenvalues[0]))
        length_px = 4.0 * np.sqrt(major_variance)
        width_px = 4.0 * np.sqrt(minor_variance)
        axis_ratio = np.sqrt(major_variance / max(minor_variance, np.finfo(np.float64).eps))
        axis_ratio_gate = float(min_axis_ratio)
        if length_px < MOTION_FEATURE_SHORT_TRAIL_LENGTH:
            axis_ratio_gate = min(axis_ratio_gate, MOTION_FEATURE_SHORT_TRAIL_MIN_AXIS_RATIO)
        if length_px < min_length:
            length_rejected_count += 1
            continue
        if axis_ratio < axis_ratio_gate:
            axis_ratio_rejected_count += 1
            continue
        if length_px >= MOTION_FEATURE_SHORT_TRAIL_LENGTH and float(size) < min_pixels:
            large_area_rejected_count += 1
            continue
        geometry_pass_count += 1
        x_min, x_max = int(raw_x.min()), int(raw_x.max())
        y_min, y_max = int(raw_y.min()), int(raw_y.max())
        touches_edge = (
            x_min == 0
            or y_min == 0
            or x_max == values.shape[1] - 1
            or y_max == values.shape[0] - 1
        )
        if touches_edge and length_px < max(min_length, 8.0 * psf_fwhm):
            edge_rejected_count += 1
            continue
        component_values = residual[component_slice][component_support]
        component_artifact = line_artifact_mask[component_slice][component_support]
        artifact_overlap_fraction = (
            float(np.count_nonzero(component_artifact) / component_values.size)
            if component_values.size
            else 0.0
        )
        weights = np.maximum(component_values.astype(np.float64), 0.0)
        weight_sum = float(weights.sum())
        center_x = float(np.average(raw_x, weights=weights)) if weight_sum > 0 else float(raw_x.mean())
        center_y = float(np.average(raw_y, weights=weights)) if weight_sum > 0 else float(raw_y.mean())
        major_vector = eigenvectors[:, -1]
        angle_deg = float(np.degrees(np.arctan2(major_vector[1], major_vector[0])))
        residual_peak = float(component_values.max())
        continuity_ratio, profile_fill_ratio = _line_profile_metrics(
            residual,
            center_x=center_x,
            center_y=center_y,
            length_px=length_px,
            width_px=width_px,
            angle_deg=angle_deg,
            threshold_adu=threshold,
        )
        if fit_line_model:
            # 短/中线进入完整的点源、双星、椭圆和有限线段竞争；超出
            # 局部窗口的结构走整段剖面分支，不能再用 ``not_fit`` 静默
            # 抹掉 F01 这种曝光内超长线。
            model_evidence = (
                _single_frame_long_line_profile_evidence(
                    residual,
                    valid,
                    center_x=center_x,
                    center_y=center_y,
                    length_px=length_px,
                    width_px=width_px,
                    angle_deg=angle_deg,
                    psf_fwhm=psf_fwhm,
                    noise=noise,
                )
                if length_px > SINGLE_FRAME_MODEL_MAX_LENGTH_PX
                else _single_frame_line_model_evidence(
                    residual,
                    valid,
                    center_x=center_x,
                    center_y=center_y,
                    length_px=length_px,
                    width_px=width_px,
                    angle_deg=angle_deg,
                    psf_fwhm=psf_fwhm,
                    noise=noise,
                )
            )
        else:
            model_evidence = {}
        frame_features.append(
            MotionFeaturePoint(
                frame_index=0,
                x=center_x,
                y=center_y,
                aligned_x=center_x,
                aligned_y=center_y,
                residual_snr=residual_peak / noise,
                area_pixels=int(size),
                length_px=float(length_px),
                width_px=float(width_px),
                angle_deg=angle_deg,
                bbox=(x_min, y_min, x_max, y_max),
                touches_edge=touches_edge,
                continuity_ratio=continuity_ratio,
                profile_fill_ratio=profile_fill_ratio,
                line_artifact_overlap_fraction=artifact_overlap_fraction,
                line_template_snr=model_evidence.get("line_template_snr"),
                line_vs_point_delta_bic=model_evidence.get("line_vs_point_delta_bic"),
                line_vs_double_delta_bic=model_evidence.get("line_vs_double_delta_bic"),
                line_vs_background_delta_bic=model_evidence.get("line_vs_background_delta_bic"),
                line_vs_elliptical_delta_bic=model_evidence.get("line_vs_elliptical_delta_bic"),
                single_frame_model=model_evidence.get("single_frame_model"),
                single_frame_model_margin_bic=model_evidence.get("single_frame_model_margin_bic"),
            )
        )

    if progress is not None:
        progress(85.0, f"单帧线形候选 {len(frame_features)} 条")
    audit = MotionFrameAudit(
        frame_index=0,
        threshold_adu=float(threshold),
        residual_noise_adu=float(noise),
        valid_pixel_count=int(np.count_nonzero(valid)),
        support_pixel_count=int(np.count_nonzero(support)),
        component_count=int(component_count),
        area_pass_count=area_pass_count,
        geometry_pass_count=geometry_pass_count,
        edge_rejected_count=edge_rejected_count,
        feature_count=len(frame_features),
        max_feature_residual_snr=max(
            (float(point.residual_snr) for point in frame_features),
            default=None,
        ),
        short_area_rejected_count=short_area_rejected_count,
        length_rejected_count=length_rejected_count,
        axis_ratio_rejected_count=axis_ratio_rejected_count,
        large_area_rejected_count=large_area_rejected_count,
        short_min_pixels=int(short_min_pixels),
    )
    if audit_sink is not None:
        audit_sink.append(audit)
    tracks = tuple(
        MotionFeatureTrack(
            track_id=track_id,
            classification="candidate",
            points=(point,),
            displacement_px=0.0,
            speed_px_per_frame=0.0,
            fit_rms_px=None,
            evidence_level="single_frame_trail",
            rejection_reasons=("SINGLE_FRAME_ONLY", "NO_CROSS_FRAME_ASSOCIATION"),
        )
        for track_id, point in enumerate(frame_features)
    )
    if progress is not None:
        progress(100.0, "单帧线状候选筛选完成")
    return tracks, float(threshold)


def detect_long_trails(
    frame_analysis: FrameAnalysis,
    *,
    psf_fwhm: float,
    residual_sigma: float = 15.0,
    min_residual_adu: float = 100.0,
    min_feature_area: int = 40,
    min_axis_ratio: float = 4.0,
    min_length_px: float | None = None,
    bridge_gaps: bool = False,
    fit_line_model: bool = False,
    progress: Callable[[float, str], None] | None = None,
    audit_sink: list[MotionFrameAudit] | None = None,
) -> tuple[MotionFeatureTrack, ...]:
    """对已经完成背景测量的单帧提取曝光内长线候选。

    这是单帧 API 的独立入口，不再调用 ``detect_motion_features``。后者
    专门负责 15 帧时间参考、配准坐标、候选关联和速度拟合；本函数只
    处理当前 ``frame_analysis``，返回的分类永远是 ``candidate``。

    ``fit_line_model`` 只对需要完整单帧负例复核的审计调用开启。默认值
    保持轻量几何 API 的兼容性；开启后会把当前局部结构与单星、双星、
    静态椭圆星像和有限线段模型比较，避免相邻点源仅因 PCA 拉长就被写
    成可接受的线状证据。
    """

    tracks, _threshold = _extract_single_frame_trail_candidates(
        frame_analysis,
        psf_fwhm=psf_fwhm,
        residual_sigma=residual_sigma,
        min_residual_adu=min_residual_adu,
        min_feature_area=min_feature_area,
        min_axis_ratio=min_axis_ratio,
        min_length_px=min_length_px,
        bridge_gaps=bridge_gaps,
        progress=progress,
        audit_sink=audit_sink,
        fit_line_model=fit_line_model,
    )
    return tracks


def detect_single_frame_long_trails(
    frame: FitsFrame,
    *,
    psf_fwhm: float,
    residual_sigma: float = DEFAULT_SINGLE_FRAME_TRAIL_RESIDUAL_SIGMA,
    min_residual_adu: float = DEFAULT_SINGLE_FRAME_TRAIL_MIN_RESIDUAL_ADU,
    min_feature_area: int = DEFAULT_SINGLE_FRAME_TRAIL_MIN_AREA,
    min_axis_ratio: float = DEFAULT_SINGLE_FRAME_TRAIL_MIN_AXIS_RATIO,
    min_length_px: float | None = None,
    bridge_gaps: bool = DEFAULT_SINGLE_FRAME_TRAIL_BRIDGE_GAPS,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[MotionFeatureTrack, ...]:
    """在不等待全量星点测光的情况下提取单帧曝光线形候选。

    单帧长线只需要全图稳健背景、有效像素和连通域几何，不需要先完成
    数万候选的孔径测光。这里构造一个只含背景基线的轻量 ``FrameAnalysis``
    后直接进入单帧提取器；它不创建时间参考图，也不调用跨帧关联。单帧默认
    使用比序列确认层略宽的几何门槛，但默认不桥接像素缺口；这样两颗相邻恒星
    不会因为一次膨胀被合成一条线。调用方只有在明确知道要做宽筛时才传入
    ``bridge_gaps=True``，且合并只用于连通域分组，面积、PCA 和测量权重仍来自
    原始残差像素。结果仍
    严格是 ``candidate``，但它代表“曝光期间可能发生位移”的形态候选，不是
    已经确认的运动目标。只有 15 帧路径完成跨帧关联后，结果才可能升级成
    ``moving``。
    """

    if psf_fwhm <= 0 or residual_sigma <= 0 or min_residual_adu <= 0:
        raise ValueError("single-frame trail parameters must be positive")
    if min_feature_area < 3 or min_axis_ratio <= 0:
        raise ValueError("single-frame trail geometry parameters are invalid")
    if min_length_px is not None and min_length_px <= 0:
        raise ValueError("single-frame trail length parameter must be positive")
    # Keep the standalone single-frame view on the same exposure-aware
    # proposal floor as the sequence fallback.  The underlying spatial
    # proposal module remains time-agnostic; only this FITS-facing caller
    # resolves the physical metadata.  With no usable EXPOSURE header the
    # historical 8 px / 3×FWHM floor is preserved.
    exposure_ms = exposure_milliseconds(frame.header)
    exposure_s = (
        (float(exposure_ms) / 1000.0,)
        if exposure_ms is not None and np.isfinite(float(exposure_ms)) and float(exposure_ms) > 0.0
        else (None,)
    )
    resolved_min_length_px = _sequence_motion_candidate_min_length(
        psf_fwhm=float(psf_fwhm),
        min_length_px=min_length_px,
        exposure_s=exposure_s,
        metadata_supported=exposure_s[0] is not None,
    )
    values = np.asarray(frame.data)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    if progress is not None:
        progress(1.0, "估计单帧背景与噪声")
    # 这里是“先显示”的候选预览，不承担单图星等最终测量；对当前 16
    # 位 FITS，float32 可以精确表示每个 ADU，避免为这次轻量预览复制
    # 一份 128 MiB 的 float64 全图。
    calculation_dtype = np.float32 if (
        np.issubdtype(values.dtype, np.integer) or values.dtype == np.float32
    ) else np.float64
    numeric = np.asarray(values, dtype=calculation_dtype)
    auxiliary = auxiliary_mask(values.shape)
    base_mask = ~np.isfinite(numeric) | auxiliary
    global_background, _global_noise = sigma_clipped_stats(numeric, mask=base_mask)
    effective_mask, saturation_level, used_zero_mask = _working_mask(
        values,
        auxiliary,
        background=global_background,
        mask_zero_pixels=None,
        saturation_level=None,
        numeric=numeric,
    )
    background, noise = sigma_clipped_stats(numeric, mask=effective_mask)
    valid = ~np.asarray(effective_mask, dtype=bool)
    residual = np.where(valid, numeric - float(background), 0.0)
    line_artifact_mask = _linear_artifact_mask(
        residual,
        np.full(values.shape, max(float(noise), np.finfo(np.float32).eps), dtype=np.float32),
        valid,
        detection_sigma=4.0,
        psf_fwhm=float(psf_fwhm),
    )
    line_artifact_coordinates = (
        np.column_stack(np.nonzero(line_artifact_mask)[::-1]).astype(np.int32, copy=False)
        if np.any(line_artifact_mask)
        else np.empty((0, 2), dtype=np.int32)
    )
    detection = DetectionResult(
        image_shape=(int(values.shape[0]), int(values.shape[1])),
        background=float(background),
        noise=float(noise),
        threshold=float(background + 4.0 * noise),
        candidate_count=0,
        sources=(),
        parameters={
            "mask_zero_pixels": int(used_zero_mask),
            "saturation_level": -1.0 if saturation_level is None else float(saturation_level),
        },
        quality_count=0,
        line_artifact_coordinates=line_artifact_coordinates,
    )
    analysis = FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=None)
    if progress is not None:
        progress(2.0, "开始单帧长线几何筛选")

        def trail_progress(value: float, label: str) -> None:
            progress(2.0 + 0.98 * max(0.0, min(100.0, float(value))), label)

    else:
        trail_progress = None
    # 这里故意直接进入单帧提取器，不经过任何序列包装入口。这样调用
    # 链本身就能保证：单帧不会建立时间中值参考、不会做帧间关联，也
    # 不会因为后续重构把单帧任务误接到 ``detect_motion_features``。
    tracks, _threshold = _extract_single_frame_trail_candidates(
        analysis,
        psf_fwhm=psf_fwhm,
        residual_sigma=residual_sigma,
        min_residual_adu=min_residual_adu,
        min_feature_area=min_feature_area,
        min_axis_ratio=min_axis_ratio,
        min_length_px=resolved_min_length_px,
        bridge_gaps=bridge_gaps,
        fit_line_model=True,
        progress=trail_progress,
        audit_sink=None,
        short_min_feature_area=_exposure_aware_short_trail_min_area(
            psf_fwhm=float(psf_fwhm),
            exposure_s=exposure_s[0],
        ),
    )
    # 连通域/PCA 是亮线的快速提案，但会漏掉“单像素不够高、沿线积分
    # 才显著”的短拖影。对较大的真实 FITS，再用当前这一张图的降采样
    # 多尺度线段匹配做第二条提案支路；它不接收其它帧，也不参与序列
    # 关联。小图/单元测试保留原来的轻量路径，避免改变旧 API 的行为。
    if min(values.shape) >= 512:
        if trail_progress is not None:
            trail_progress(88.0, "单帧多尺度线段宽筛")
        tracks = _augment_single_frame_trails_with_multiscale_proposals(
            tracks,
            residual=residual,
            valid=valid,
            line_artifact_mask=line_artifact_mask,
            background=float(background),
            noise=max(float(noise), np.finfo(np.float32).eps),
            psf_fwhm=float(psf_fwhm),
            min_length_px=float(resolved_min_length_px),
            progress=trail_progress,
        )
        if trail_progress is not None:
            trail_progress(100.0, f"单帧线状候选筛选完成 · 共 {len(tracks)} 条")
    # 连通域主路线和原图/降采样提案路线都可能在同一条亮线的不同位置
    # 返回几何重叠结果。最终单帧结果只保留每个共线结构的一个代表，避免
    # 默认图把一条白线画成多个目标；不重叠的短线不受影响。
    tracks = _deduplicate_single_frame_line_tracks(tracks)
    if trail_progress is not None and min(values.shape) >= 512:
        trail_progress(100.0, f"单帧线状候选筛选完成 · 去重后 {len(tracks)} 条")
    return tracks


def _block_reduce_single_frame(
    values: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用块均值把当前帧缩小，返回降采样图和有效块掩膜。"""

    if factor <= 1:
        return np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool)
    height, width = values.shape
    reduced_height = height // factor
    reduced_width = width // factor
    if reduced_height < 8 or reduced_width < 8:
        return np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool)
    trimmed = np.asarray(values[: reduced_height * factor, : reduced_width * factor], dtype=np.float64)
    trimmed_valid = np.asarray(valid[: reduced_height * factor, : reduced_width * factor], dtype=bool)
    blocks = trimmed.reshape(reduced_height, factor, reduced_width, factor)
    valid_blocks = trimmed_valid.reshape(reduced_height, factor, reduced_width, factor)
    counts = valid_blocks.sum(axis=(1, 3))
    totals = np.where(valid_blocks, blocks, 0.0).sum(axis=(1, 3))
    reduced = np.zeros((reduced_height, reduced_width), dtype=np.float64)
    np.divide(totals, counts, out=reduced, where=counts > 0)
    # 允许少量坏像素，但不让一条线跨过大块无效区后被重新连起来。
    reduced_valid = counts >= max(1, int(np.ceil(0.75 * factor * factor)))
    return reduced, reduced_valid


def _single_frame_proposal_artifact_overlap(
    artifact_mask: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    width_px: float,
    angle_deg: float,
) -> float:
    """估计一条单帧线段走廊与线状掩膜的重叠比例。"""

    if artifact_mask.ndim != 2 or length_px <= 0.0:
        return 0.0
    sample_count = max(9, int(np.ceil(length_px)) + 1)
    along = np.linspace(-0.5 * length_px, 0.5 * length_px, sample_count)
    across = np.linspace(-max(1.0, min(4.0, 0.5 * width_px + 1.0)), max(1.0, min(4.0, 0.5 * width_px + 1.0)), 5)
    angle = np.deg2rad(float(angle_deg))
    unit_x, unit_y = float(np.cos(angle)), float(np.sin(angle))
    normal_x, normal_y = -unit_y, unit_x
    sample_x = center_x + along[:, None] * unit_x + across[None, :] * normal_x
    sample_y = center_y + along[:, None] * unit_y + across[None, :] * normal_y
    sampled = ndimage.map_coordinates(
        np.asarray(artifact_mask, dtype=np.float32),
        [sample_y.T, sample_x.T],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return float(np.mean(sampled > 0.5)) if sampled.size else 0.0


def _single_frame_proposal_peak_residual(
    residual: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length_px: float,
    width_px: float,
    angle_deg: float,
) -> float:
    """从当前线段走廊提取峰值残差，用于兼容旧的悬浮信息字段。"""

    sample_count = max(9, int(np.ceil(length_px)) + 1)
    along = np.linspace(-0.5 * length_px, 0.5 * length_px, sample_count)
    across = np.linspace(-max(1.0, min(4.0, 0.5 * width_px + 1.0)), max(1.0, min(4.0, 0.5 * width_px + 1.0)), 5)
    angle = np.deg2rad(float(angle_deg))
    unit_x, unit_y = float(np.cos(angle)), float(np.sin(angle))
    normal_x, normal_y = -unit_y, unit_x
    sample_x = center_x + along[:, None] * unit_x + across[None, :] * normal_x
    sample_y = center_y + along[:, None] * unit_y + across[None, :] * normal_y
    sampled = ndimage.map_coordinates(
        np.asarray(residual, dtype=np.float32),
        [sample_y.T, sample_x.T],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    finite = sampled[np.isfinite(sampled)]
    return float(np.max(finite)) if finite.size else 0.0


def _single_frame_line_geometries_overlap(
    first: tuple[float, float, float, float, float],
    second: tuple[float, float, float, float, float],
) -> bool:
    """判断两条当前帧线段是否是同一条几何结构的重复提案。

    单帧现在有两条互补的提案支路：阈值连通域会给出完整亮线，降采样
    匹配滤波则能找出低峰值但沿线积分显著的结构。两条支路的中心可能
    落在同一条长线的不同位置，单纯按中心距离去重会把它们都留下。这里
    使用无向角度、垂直距离和沿线重叠比例做线段级 NMS；不重叠的平行线
    仍然可以分别保留。该函数只比较当前图像中的空间几何，不涉及帧间
    关联或时间信息。

    元组格式为 ``(center_x, center_y, length_px, width_px, angle_deg)``。
    """

    first_x, first_y, first_length, first_width, first_angle = first
    second_x, second_y, second_length, second_width, second_angle = second
    if first_length <= 0.0 or second_length <= 0.0:
        return False
    # 这是一条“线段是否落在另一条线段走廊内”的关系，必须与参数
    # 顺序无关。短段的 PCA 方向在只有十几个像素时更容易量化偏转；
    # 若直接用短段作基准，位于长段末端的同一条结构会因两个中心相距
    # 较远而被判成不重叠。始终以较长段作方向/横向距离基准，既保持
    # 共线段去重，又不会改变真正平行分离线的判定。
    if second_length > first_length:
        first_x, second_x = second_x, first_x
        first_y, second_y = second_y, first_y
        first_length, second_length = second_length, first_length
        first_width, second_width = second_width, first_width
        first_angle, second_angle = second_angle, first_angle
    angle_difference = abs((float(first_angle) - float(second_angle) + 90.0) % 180.0 - 90.0)
    if angle_difference > 15.0:
        return False

    angle = np.deg2rad(float(first_angle))
    direction_x = float(np.cos(angle))
    direction_y = float(np.sin(angle))
    delta_x = float(second_x) - float(first_x)
    delta_y = float(second_y) - float(first_y)
    perpendicular_distance = abs(delta_x * direction_y - delta_y * direction_x)
    perpendicular_gate = max(
        2.5,
        0.5 * (float(first_width) + float(second_width)) + 1.5,
    )
    if perpendicular_distance > perpendicular_gate:
        return False

    parallel_distance = abs(delta_x * direction_x + delta_y * direction_y)
    overlap = 0.5 * (float(first_length) + float(second_length)) - parallel_distance
    shorter_length = min(float(first_length), float(second_length))
    return overlap / shorter_length >= 0.35


def _deduplicate_single_frame_line_tracks(
    tracks: Sequence[MotionFeatureTrack],
) -> tuple[MotionFeatureTrack, ...]:
    """Collapse repeated collinear pieces from the current-frame detector.

    The threshold-connected-component pass can split one bright exposure trail
    into a long component plus shorter pieces at local gaps.  The native-pixel
    proposal pass can then rediscover the same geometry at another length.  The
    old merge logic protected only the *new* proposal batch, so the already
    existing pieces were all sent to the GUI and visually looked like several
    targets.  Keep the strongest representative of each overlapping geometry;
    non-overlapping short trails remain independent candidates.

    This is deliberately a spatial NMS operation on one image.  It does not
    associate frames, estimate a direction, or promote a candidate to
    ``moving``.
    """

    if len(tracks) < 2:
        return tuple(tracks)

    def priority(track: MotionFeatureTrack) -> tuple[int, int, int, float, float]:
        if not track.points:
            return (0, 0, 0, 0.0, 0.0)
        point = track.points[0]
        artifact_overlap = point.line_artifact_overlap_fraction
        artifact_clear = int(
            artifact_overlap is None
            or float(artifact_overlap) < SINGLE_FRAME_ARTIFACT_REVIEW_OVERLAP_FRACTION
        )
        return (
            int(_single_frame_trail_shape_is_supported(point)),
            int(_single_frame_model_is_supported(point)),
            artifact_clear,
            float(point.length_px),
            float(point.residual_snr),
        )

    ordered = sorted(
        (track for track in tracks if track.points),
        key=priority,
        reverse=True,
    )
    kept: list[MotionFeatureTrack] = []
    for track in ordered:
        point = track.points[0]
        geometry = (
            float(point.x),
            float(point.y),
            float(point.length_px),
            float(point.width_px),
            float(point.angle_deg),
        )
        if any(
            _single_frame_line_geometries_overlap(
                geometry,
                (
                    float(existing.points[0].x),
                    float(existing.points[0].y),
                    float(existing.points[0].length_px),
                    float(existing.points[0].width_px),
                    float(existing.points[0].angle_deg),
                ),
            )
            for existing in kept
        ):
            continue
        kept.append(track)

    # Keep original IDs/order for hover state, cache readability, and
    # deterministic screenshots.  Empty tracks are not expected from the
    # detector, but preserving them makes this helper safe for old callers.
    empty_tracks = [track for track in tracks if not track.points]
    return tuple(sorted((*kept, *empty_tracks), key=lambda track: track.track_id))


def _augment_single_frame_trails_with_multiscale_proposals(
    base_tracks: Sequence[MotionFeatureTrack],
    *,
    residual: np.ndarray,
    valid: np.ndarray,
    line_artifact_mask: np.ndarray,
    background: float,
    noise: float,
    psf_fwhm: float,
    min_length_px: float,
    progress: Callable[[float, str], None] | None,
    length_bank_px: Sequence[float] | None = None,
    angle_step_deg: float = 15.0,
    max_proposals: int = 256,
    model_proposal_budget: int | None = None,
    include_reduced_scale: bool = True,
) -> tuple[MotionFeatureTrack, ...]:
    """把当前帧的原图短线和多尺度线段宽筛并入单帧候选。

    原图短线支路必须先于降采样支路运行：4×块平均适合找长结构，
    却会把 8--20 px 拖影压成几个像素。两条支路都只读当前帧；这里仍
    不接收序列、时间参考或帧间位移。
    """

    values = np.asarray(residual, dtype=np.float64) + float(background)
    if model_proposal_budget is not None and int(model_proposal_budget) < 1:
        raise ValueError("model_proposal_budget must be positive or None")
    # 这是单帧短线的主补检支路。它在原始像素上做有界局部扫描，避免把
    # 用户关心的短拖影先压缩掉。局部扫描的候选仍会在下面回到原图做
    # 有限线段/点源/双点模型复核。
    local_proposals = ()
    artifact_proposals = ()
    local_length_bank = tuple(
        sorted(
            {
                float(min_length_px),
                8.0,
                10.0,
                12.0,
                16.0,
                20.0,
                24.0,
                32.0,
                48.0,
                64.0,
            }
        )
    )
    try:
        local_proposals = detect_local_pixel_trail_proposals(
            values,
            # 普通像素单独占主要预算，避免最长固定亮线抢走所有种子。
            valid & ~np.asarray(line_artifact_mask, dtype=bool),
            float(background),
            max(float(noise), np.finfo(np.float64).eps),
            float(psf_fwhm),
            length_bank_px=length_bank_px if length_bank_px is not None else local_length_bank,
            # 序列补充通道已经有时间残差和连通域主路线兜底；在保留
            # 1.5 s 短线方向召回的前提下，15° 网格配合后面的局部复核
            # 即可，避免每帧重复扫描 18 个方向。
            angle_step_deg=(
                min(15.0, float(angle_step_deg))
                if model_proposal_budget is not None
                else min(10.0, float(angle_step_deg))
            ),
            max_proposals=min(128, int(max_proposals)),
            # 序列中每个 256 px 网格保留 1 个种子，仍覆盖整幅视场；
            # 单帧入口保持历史 512 个种子，不改变其召回口径。
            max_seeds=256 if model_proposal_budget is not None else 512,
            seed_threshold=2.25,
            sample_step_px=2.0 if model_proposal_budget is not None else 1.0,
        )
    except (TypeError, ValueError, FloatingPointError):
        # 短线补检失败不能抹掉连通域主路线；下面仍尝试既有的长结构支路。
        local_proposals = ()
    # 伪影掩膜不能成为真实短线的绝对禁区。保留一个更小的独立预算，
    # 让“落在固定线附近的短线”先进入 candidate/review，再由原图模型、
    # 跨帧速度和伪影审计决定能否升级；不在宽筛阶段静默删除它。
    artifact_mask = np.asarray(line_artifact_mask, dtype=bool)
    if np.any(artifact_mask):
        try:
            artifact_proposals = detect_local_pixel_trail_proposals(
                values,
                valid & artifact_mask,
                float(background),
                max(float(noise), np.finfo(np.float64).eps),
                float(psf_fwhm),
                length_bank_px=length_bank_px if length_bank_px is not None else local_length_bank,
                angle_step_deg=(
                    min(15.0, float(angle_step_deg))
                    if model_proposal_budget is not None
                    else min(10.0, float(angle_step_deg))
                ),
                max_proposals=min(48, int(max_proposals)),
                max_seeds=64 if model_proposal_budget is not None else 128,
                seed_threshold=2.25,
                sample_step_px=2.0 if model_proposal_budget is not None else 1.0,
            )
        except (TypeError, ValueError, FloatingPointError):
            artifact_proposals = ()
    if progress is not None:
        progress(
            92.0,
            f"原图像素短线提案 {len(local_proposals)} 条 · 伪影区复核 {len(artifact_proposals)} 条",
        )

    reduced_proposals = ()
    factor = max(1, int(np.ceil(min(values.shape) / 1024.0)))
    if include_reduced_scale:
        reduced, reduced_valid = _block_reduce_single_frame(values, valid, factor)
        reduced_fwhm = max(0.5, float(psf_fwhm) / factor)
        original_length_bank = tuple(
            float(value)
            for value in (
                length_bank_px
                if length_bank_px is not None
                else (8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0)
            )
        )
        reduced_lengths = tuple(
            sorted(
                {
                    max(2.0, round(length / factor, 2))
                    for length in original_length_bank
                    if length / factor < min(reduced.shape) - 1.0
                }
            )
        )
        reduced_background = float(background)
        reduced_noise = max(float(noise) / np.sqrt(float(factor * factor)), np.finfo(np.float64).eps)
        if reduced_lengths:
            try:
                reduced_proposals = detect_single_frame_trail_proposals(
                    reduced,
                    reduced_valid,
                    reduced_background,
                    reduced_noise,
                    reduced_fwhm,
                    length_bank_px=reduced_lengths,
                    angle_step_deg=float(angle_step_deg),
                    max_proposals=int(max_proposals),
                )
            except (TypeError, ValueError, FloatingPointError):
                # 宽筛是补充证据；输入掩膜或数值异常时不阻塞原有单帧路线。
                reduced_proposals = ()
    if progress is not None:
        progress(
            96.0,
            f"多尺度线段提案 {len(reduced_proposals)} 条，正在回到原图复核",
        )

    existing_points = [point for track in base_tracks for point in track.points]
    added: list[MotionFeatureTrack] = []
    # 多尺度 proposal 与连通域路线共用 sequence review 门。尤其是
    # 量化成很窄像素带的短拖影，局部模型可能返回 elliptical_point；
    # 这不等于静态椭圆星像已经被接受，后面的 review helper 还会检查
    # 宽度、连续性以及 line-vs-double/point/ellipse 的弱证据。
    allowed_models = {"line", "long_line_profile", "elliptical_point", "not_fit"}
    accepted_geometries = [
        (
            float(point.x),
            float(point.y),
            float(point.length_px),
            float(point.width_px),
            float(point.angle_deg),
        )
        for point in existing_points
        if point.single_frame_model in allowed_models
    ]
    proposal_batches = (
        (local_proposals, 1.0, 0.0),
        (artifact_proposals, 1.0, 0.0),
        (reduced_proposals, float(factor), 0.5 * (factor - 1)),
    )
    if model_proposal_budget is not None:
        # Native-pixel proposals are the important recall path for the 1.5 s
        # short trails.  Keep a separate artifact reserve because a detector
        # line can overlap the generic artifact mask, and use the remaining
        # budget for the reduced-image long-line proposals.  Sorting by the
        # proposal response is deterministic and happens before any expensive
        # local model is built; it therefore changes cost, not the model gate.
        batch_budgets = (24, 8, max(0, int(model_proposal_budget) - 32))
        bounded_batches: list[tuple[Sequence[object], float, float]] = []
        for (proposals, coordinate_scale, coordinate_offset), budget in zip(
            proposal_batches,
            batch_budgets,
            strict=True,
        ):
            if budget <= 0 or len(proposals) <= budget:
                selected = proposals
            else:
                selected = tuple(
                    proposal
                    for _index, proposal in sorted(
                        enumerate(proposals),
                        key=lambda item: (
                            -float(item[1].response),
                            -float(item[1].length),
                            float(item[1].y),
                            float(item[1].x),
                            int(item[0]),
                        ),
                    )[:budget]
                )
            bounded_batches.append((selected, coordinate_scale, coordinate_offset))
        proposal_batches = tuple(bounded_batches)
    for proposals, coordinate_scale, coordinate_offset in proposal_batches:
        for proposal in proposals:
            center_x = float(proposal.x) * coordinate_scale + coordinate_offset
            center_y = float(proposal.y) * coordinate_scale + coordinate_offset
            length_px = float(proposal.length) * coordinate_scale
            angle_deg = float(proposal.angle) % 180.0
            width_px = max(float(psf_fwhm), float(proposal.width) * coordinate_scale)
            if length_px < float(min_length_px):
                continue
            proposal_geometry = (center_x, center_y, length_px, width_px, angle_deg)
            if any(
                _single_frame_line_geometries_overlap(proposal_geometry, existing_geometry)
                for existing_geometry in accepted_geometries
            ):
                continue
            model_evidence = (
                _single_frame_long_line_profile_evidence(
                    residual,
                    valid,
                    center_x=center_x,
                    center_y=center_y,
                    length_px=length_px,
                    width_px=width_px,
                    angle_deg=angle_deg,
                    psf_fwhm=psf_fwhm,
                    noise=noise,
                )
                if length_px > SINGLE_FRAME_MODEL_MAX_LENGTH_PX
                else _single_frame_line_model_evidence(
                    residual,
                    valid,
                    center_x=center_x,
                    center_y=center_y,
                    length_px=length_px,
                    width_px=width_px,
                    angle_deg=angle_deg,
                    psf_fwhm=psf_fwhm,
                    noise=noise,
                )
            )
            angle = np.deg2rad(angle_deg)
            half_width = max(0.5, 0.5 * width_px)
            x_extent = 0.5 * length_px * abs(float(np.cos(angle))) + half_width * abs(float(np.sin(angle))) + 1.0
            y_extent = 0.5 * length_px * abs(float(np.sin(angle))) + half_width * abs(float(np.cos(angle))) + 1.0
            x_min = max(0, int(np.floor(center_x - x_extent)))
            x_max = min(residual.shape[1] - 1, int(np.ceil(center_x + x_extent)))
            y_min = max(0, int(np.floor(center_y - y_extent)))
            y_max = min(residual.shape[0] - 1, int(np.ceil(center_y + y_extent)))
            peak_residual = _single_frame_proposal_peak_residual(
                residual,
                center_x=center_x,
                center_y=center_y,
                length_px=length_px,
                width_px=width_px,
                angle_deg=angle_deg,
            )
            continuity_ratio, profile_fill_ratio = _line_profile_metrics(
                residual,
                center_x=center_x,
                center_y=center_y,
                length_px=length_px,
                width_px=width_px,
                angle_deg=angle_deg,
                threshold_adu=max(3.0 * float(noise), np.finfo(np.float64).eps),
            )
            point = MotionFeaturePoint(
                frame_index=0,
                x=center_x,
                y=center_y,
                aligned_x=center_x,
                aligned_y=center_y,
                residual_snr=peak_residual / max(float(noise), np.finfo(np.float64).eps),
                area_pixels=max(1, int(round(length_px * max(width_px, 1.0)))),
                length_px=length_px,
                width_px=width_px,
                angle_deg=angle_deg,
                bbox=(x_min, y_min, x_max, y_max),
                touches_edge=x_min == 0 or y_min == 0 or x_max == residual.shape[1] - 1 or y_max == residual.shape[0] - 1,
                continuity_ratio=continuity_ratio,
                profile_fill_ratio=profile_fill_ratio,
                line_artifact_overlap_fraction=_single_frame_proposal_artifact_overlap(
                    line_artifact_mask,
                    center_x=center_x,
                    center_y=center_y,
                    length_px=length_px,
                    width_px=width_px,
                    angle_deg=angle_deg,
                ),
                line_template_snr=model_evidence.get("line_template_snr"),
                line_vs_point_delta_bic=model_evidence.get("line_vs_point_delta_bic"),
                line_vs_double_delta_bic=model_evidence.get("line_vs_double_delta_bic"),
                line_vs_background_delta_bic=model_evidence.get("line_vs_background_delta_bic"),
                line_vs_elliptical_delta_bic=model_evidence.get("line_vs_elliptical_delta_bic"),
                single_frame_model=model_evidence.get("single_frame_model"),
                single_frame_model_margin_bic=model_evidence.get("single_frame_model_margin_bic"),
            )
            if not _single_frame_sequence_line_model_is_review_supported(
                point,
                psf_fwhm=float(psf_fwhm),
            ):
                continue
            added.append(
                MotionFeatureTrack(
                    track_id=len(base_tracks) + len(added),
                    classification="candidate",
                    points=(point,),
                    displacement_px=0.0,
                    speed_px_per_frame=0.0,
                    fit_rms_px=None,
                    evidence_level="single_frame_trail",
                    rejection_reasons=("SINGLE_FRAME_ONLY", "NO_CROSS_FRAME_ASSOCIATION"),
                )
            )
            accepted_geometries.append(proposal_geometry)
    return tuple(base_tracks) + tuple(added)


def split_single_frame_trail_blends(
    trails: Sequence[MotionFeatureTrack],
    frame_analysis: FrameAnalysis,
    *,
    max_length_px: float = DEFAULT_SINGLE_FRAME_BLEND_GUARD_MAX_LENGTH,
    margin_px: float = DEFAULT_SINGLE_FRAME_BLEND_GUARD_MARGIN,
) -> tuple[tuple[MotionFeatureTrack, ...], tuple[MotionFeatureTrack, ...]]:
    """兼容旧接口，但不再用静态近邻星点自动排除单帧线状候选。

    之前的实现只要在线段主轴走廊内找到两个通过质量门控的点源，就把
    整条线降为紫色“点源混合”。这并不能证明线是两个点源拼出来的：一条
    真实短拖影同样可能穿过恒星附近，而且单张曝光没有时间维度，无法仅凭
    邻近关系判断运动与点源混合。错误排除的代价是直接让用户看不到真正的
    线形证据，因此默认路径现在保守地保留全部候选，交给 15 帧关联确认。

    参数和返回值保留，是为了兼容已经存在的调用方和缓存结果；第二项始终
    为空，不再生成紫色审计线。
    """

    if max_length_px <= 0 or margin_px < 0:
        raise ValueError("blend guard geometry parameters are invalid")
    # ``frame_analysis``、``max_length_px`` 和 ``margin_px`` 仍作为参数接受，
    # 避免第三方脚本升级后直接报错；它们不再参与默认单帧候选的删除。
    del frame_analysis, max_length_px, margin_px
    return tuple(trails), ()


def filter_single_frame_trail_blends(
    trails: Sequence[MotionFeatureTrack],
    frame_analysis: FrameAnalysis,
    *,
    max_length_px: float = DEFAULT_SINGLE_FRAME_BLEND_GUARD_MAX_LENGTH,
    margin_px: float = DEFAULT_SINGLE_FRAME_BLEND_GUARD_MARGIN,
) -> tuple[tuple[MotionFeatureTrack, ...], int]:
    """兼容旧调用：只返回运动显示候选和点源混合排除数量。"""

    kept, blend_tracks = split_single_frame_trail_blends(
        trails,
        frame_analysis,
        max_length_px=max_length_px,
        margin_px=margin_px,
    )
    return kept, len(blend_tracks)


def track_detections(
    frame_sources: Sequence[Sequence[Detection]],
    *,
    frame_names: Sequence[str] | None = None,
    frame_times: Sequence[float] | np.ndarray | None = None,
    link_radius_px: float = 4.0,
    min_presence: int | None = None,
    motion_min_displacement_px: float = 2.0,
    registration_radius_px: float = 8.0,
    max_motion_fit_rms_px: float = 0.75,
    persistent_min_presence: int | None = None,
) -> SequenceResult:
    """对已经检测的质量源做平移配准、仿射残差校正和轨迹关联。

    ``frame_sources`` 应传入 ``DetectionResult.quality_sources``，这样被
    局部 SNR、形状或坏像素规则拒绝的候选不会污染运动轨迹。为了便于
    单元测试和缓存复用，本函数不强制重新读取 FITS。轨迹点中的
    ``aligned_x/y`` 仍是可审计的平移配准坐标；运动位移和拟合 RMS 会在
    可用的稳健仿射模型校正后计算，并在结果中写出实际生效的运动下限。
    ``frame_times`` 是相对秒数；传入时只影响运动拟合，不改变帧间关联
    的 frame index 语义。省略或格式无效时回退到帧号，兼容旧测试和旧
    调用方。
    """

    if not frame_sources:
        raise ValueError("frame_sources cannot be empty")
    if link_radius_px <= 0 or registration_radius_px <= 0:
        raise ValueError("link and registration radii must be positive")
    frame_count = len(frame_sources)
    resolved_frame_times = _normalise_motion_times(frame_times, frame_count)
    if max_motion_fit_rms_px <= 0:
        raise ValueError("max_motion_fit_rms_px must be positive")
    # 三帧以上才有资格形成“持续运动”证据，但一/两帧仍应能完成
    # 配准和基础序列结果；不能因为默认的三帧门槛把短序列直接打成参数错误。
    required_presence = min(
        frame_count,
        min_presence if min_presence is not None else max(3, int(np.ceil(frame_count * 0.8))),
    )
    if required_presence < 1 or required_presence > frame_count:
        raise ValueError("min_presence must be within the frame count")
    persistent_required = (
        persistent_min_presence
        if persistent_min_presence is not None
        else min(required_presence, max(3, int(np.ceil(frame_count * 0.5))))
    )
    if persistent_required < 1 or persistent_required > required_presence:
        raise ValueError("persistent_min_presence must be within 1..min_presence")
    names = tuple(frame_names or (str(index) for index in range(frame_count)))
    if len(names) != frame_count:
        raise ValueError("frame_names length must match frame_sources")

    cumulative: list[tuple[float, float]] = [(0.0, 0.0)]
    for index in range(1, frame_count):
        delta = _estimate_translation(
            frame_sources[index - 1],
            frame_sources[index],
            radius_px=registration_radius_px,
        )
        previous = cumulative[-1]
        cumulative.append((previous[0] + delta[0], previous[1] + delta[1]))

    aligned_frames: list[list[tuple[Detection, float, float]]] = []
    for index, sources in enumerate(frame_sources):
        shift_x, shift_y = cumulative[index]
        aligned_frames.append([(source, source.x - shift_x, source.y - shift_y) for source in sources])

    # 每条轨迹保存最后一个点和其历史；未匹配源会在当前帧新建轨迹。
    tracks: list[list[TrackPoint]] = []
    if aligned_frames:
        for source, x, y in aligned_frames[0]:
            tracks.append(
                [
                    TrackPoint(
                        0,
                        source.detection_id,
                        source.x,
                        source.y,
                        x,
                        y,
                        source.flux_snr,
                        quality_passed=bool(source.quality_passed),
                    )
                ]
            )
    for frame_index in range(1, frame_count):
        current = aligned_frames[frame_index]
        if not current:
            continue
        points = np.array([(x, y) for _, x, y in current], dtype=np.float64)
        eligible_tracks: list[tuple[int, float, float]] = []
        for track_index, history in enumerate(tracks):
            last = history[-1]
            # 严格的连续观测：漏掉一帧就结束旧轨迹，避免把两个不相邻
            # 的偶然亮点重新串起来；运动候选的独立检测层负责处理短暂
            # 出现或拖线目标。
            if last.frame_index != frame_index - 1:
                continue
            eligible_tracks.append((track_index, last.aligned_x, last.aligned_y))
        # 先按局部连通分量做全局一对一分配，避免近邻/交叉轨迹因贪心
        # 抢占同一个当前源而被错误拆断。
        possible = _global_frame_assignments(
            eligible_tracks,
            points,
            link_radius_px=link_radius_px,
        )
        assigned_tracks: set[int] = set()
        assigned_sources: set[int] = set()
        for _distance, track_index, source_index in possible:
            if track_index in assigned_tracks or source_index in assigned_sources:
                continue
            source, x, y = current[source_index]
            tracks[track_index].append(
                TrackPoint(
                    frame_index,
                    source.detection_id,
                    source.x,
                    source.y,
                    x,
                    y,
                    source.flux_snr,
                    quality_passed=bool(source.quality_passed),
                )
            )
            assigned_tracks.add(track_index)
            assigned_sources.add(source_index)
        for source_index, (source, x, y) in enumerate(current):
            if source_index not in assigned_sources:
                tracks.append(
                    [
                        TrackPoint(
                            frame_index,
                            source.detection_id,
                            source.x,
                            source.y,
                            x,
                            y,
                            source.flux_snr,
                            quality_passed=bool(source.quality_passed),
                        )
                    ]
                )

    # 平移后仍可能残留很小的视场旋转/尺度漂移。先用全部已关联轨迹
    # 估计稳健的低自由度模型，再只在“运动判定指标”中扣除该系统项；
    # ``TrackPoint.aligned_x/y`` 和 ``cumulative_shifts`` 仍保留原始平移
    # 配准坐标，方便审计、复现和下游候选层使用同一坐标约定。
    affine_models, affine_frame_count, affine_residual_median = _estimate_sequence_affine_models(
        tracks,
        frame_count,
    )
    track_metrics: list[tuple[int, float, float, float | None]] = []
    for points in tracks:
        aligned_coordinates = np.asarray(
            [(point.aligned_x, point.aligned_y) for point in points],
            dtype=np.float64,
        )
        corrected_coordinates = aligned_coordinates.copy()
        if points:
            point_frames = np.asarray([point.frame_index for point in points], dtype=np.intp)
            for frame_index in np.unique(point_frames):
                frame_mask = point_frames == int(frame_index)
                corrected_coordinates[frame_mask] = _inverse_affine_coordinates(
                    aligned_coordinates[frame_mask],
                    affine_models[int(frame_index)],
                )
        displacement, speed, fit_rms = _fit_track_coordinates(
            points,
            corrected_coordinates,
            frame_times=resolved_frame_times,
        )
        track_metrics.append((len(points), displacement, speed, fit_rms))
    effective_motion_min_displacement_px = _derive_motion_displacement_floor(
        track_metrics,
        required_presence=required_presence,
        user_floor=float(motion_min_displacement_px),
        max_motion_fit_rms_px=float(max_motion_fit_rms_px),
        affine_frame_count=affine_frame_count,
    )

    rendered_tracks: list[SourceTrack] = []
    motion_supported = frame_count >= 3
    for track_id, (points, metrics) in enumerate(zip(tracks, track_metrics, strict=True)):
        _presence, displacement, speed, fit_rms = metrics
        if (
            len(points) >= required_presence
            and displacement >= effective_motion_min_displacement_px
            and fit_rms is not None
            and fit_rms <= max_motion_fit_rms_px
        ):
            classification = "moving" if motion_supported else "static"
        elif len(points) >= required_presence:
            classification = "static"
        elif (
            len(points) >= persistent_required
            and displacement < effective_motion_min_displacement_px
            and fit_rms is not None
            and fit_rms <= max_motion_fit_rms_px
        ):
            classification = "persistent"
        else:
            classification = "transient"
        rendered_tracks.append(
            SourceTrack(
                track_id=track_id,
                classification=classification,
                points=tuple(points),
                displacement_px=displacement,
                speed_px_per_frame=speed,
                fit_rms_px=fit_rms,
            )
        )
    frames = tuple(
        FrameSequenceSummary(
            frame_index=index,
            path=names[index],
            candidate_count=len(frame_sources[index]),
            returned_count=len(frame_sources[index]),
            quality_count=len(frame_sources[index]),
        )
        for index in range(frame_count)
    )
    return SequenceResult(
        frames=frames,
        cumulative_shifts=tuple(cumulative),
        tracks=tuple(rendered_tracks),
        link_radius_px=link_radius_px,
        min_presence=required_presence,
        motion_min_displacement_px=motion_min_displacement_px,
        max_motion_fit_rms_px=max_motion_fit_rms_px,
        persistent_min_presence=persistent_required,
        registration_model=("translation+robust_affine" if affine_frame_count else "translation"),
        registration_affine_frame_count=affine_frame_count,
        registration_residual_median_px=affine_residual_median,
        effective_motion_min_displacement_px=effective_motion_min_displacement_px,
    )


def _candidate_consensus_tracks(
    candidate_peak_frames: Sequence[object],
    cumulative_shifts: Sequence[tuple[float, float]],
    quality_tracks: Sequence[SourceTrack],
    *,
    frame_analyses: Sequence[FrameAnalysis] | None = None,
    frame_times: Sequence[float] | np.ndarray | None = None,
    link_radius_px: float,
    min_presence: int,
    persistent_min_presence: int,
    motion_min_displacement_px: float,
    max_motion_fit_rms_px: float,
    min_candidate_snr: float,
    temporal_candidate_min_snr: float | None = None,
    temporal_reference_min_snr: float | None = None,
    temporal_min_psf_correlation: float = 0.8,
    line_artifact_frames: Sequence[object] | None = None,
    audit_sink: MutableMapping[str, int] | None = None,
) -> tuple[SourceTrack, ...]:
    """从全量滤波峰中提取跨帧稳定、但尚未通过单帧质量的候选。

    15 帧快速路径会为了速度只测量每帧前 ``N`` 个源；如果稳定星较暗，
    它们可能在单帧质量层之外，却仍然在完整匹配滤波候选层中反复出现。
    这里不把这些峰直接当作恒星，而是用首帧候选作为锚点，在配准坐标中
    做互相最近的一对一匹配，并要求至少达到持续性门槛、位置拟合稳定、
    且普通候选峰 SNR 不低于 ``min_candidate_snr``。如果提供了原始帧分析，
    还要求候选在多数出现帧中满足 3×3 核心 PSF 支持，并排除边缘、无效
    像素、线状伪迹和单像素尖峰。时间中值/稳健叠加补提案使用独立的较低强制响应
    门槛；其参考 SNR 默认达到 ``min_candidate_snr``，也可用
    ``temporal_reference_min_snr`` 单独降低做召回抽测。已经由质量源轨迹解释的位置会被去重；返回轨迹
    统一标为 ``persistent``，供 UI 以低置信颜色显示，不能与严格单帧
    质量源混写。

    ``candidate_peak_frames`` 是检测器的瞬时内存结果，不是缓存格式的一
    部分。支持 ``numpy`` 数组和测试用的三元组序列，缺少该证据时安全地
    返回空元组。``frame_analyses`` 只用于廉价的时序核心支持审计，不会
    重新执行逐源环形测光。
    """

    if not candidate_peak_frames or len(candidate_peak_frames) != len(cumulative_shifts):
        return ()
    try:
        if all(np.asarray(raw).size == 0 for raw in candidate_peak_frames):
            return ()
    except (TypeError, ValueError):
        return ()
    if link_radius_px <= 0 or min_presence < 1 or persistent_min_presence < 1:
        raise ValueError("candidate consensus thresholds are invalid")
    if persistent_min_presence > min_presence or min_candidate_snr <= 0:
        raise ValueError("candidate consensus presence/SNR thresholds are invalid")
    temporal_reference_gate = (
        float(min_candidate_snr)
        if temporal_reference_min_snr is None
        else float(temporal_reference_min_snr)
    )
    if temporal_reference_gate <= 0 or temporal_reference_gate > float(min_candidate_snr):
        raise ValueError("temporal reference SNR threshold is invalid")
    temporal_min_snr = (
        max(5.0, 0.5 * float(min_candidate_snr))
        if temporal_candidate_min_snr is None
        else float(temporal_candidate_min_snr)
    )
    if temporal_min_snr <= 0 or temporal_min_snr > float(min_candidate_snr):
        raise ValueError("temporal candidate SNR threshold is invalid")
    if not 0.0 < float(temporal_min_psf_correlation) <= 1.0:
        raise ValueError("temporal PSF correlation threshold is invalid")

    normalized: list[np.ndarray] = []
    input_row_count = 0
    gated_row_count = 0
    for raw in candidate_peak_frames:
        try:
            array = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            return ()
        if array.size == 0:
            normalized.append(np.empty((0, 5), dtype=np.float64))
            continue
        if array.ndim != 2 or array.shape[1] < 3:
            return ()
        input_row_count += int(array.shape[0])
        finite = np.isfinite(array[:, :3]).all(axis=1)
        temporal_channel = np.zeros(array.shape[0], dtype=bool)
        reference_snr = np.full(array.shape[0], np.nan, dtype=np.float64)
        if array.shape[1] >= 4:
            temporal_channel = np.isfinite(array[:, 3]) & (array[:, 3] >= 0.5)
        if array.shape[1] >= 5:
            reference_snr = np.asarray(array[:, 4], dtype=np.float64)
        score_gate = np.where(temporal_channel, temporal_min_snr, float(min_candidate_snr))
        finite &= array[:, 2] >= score_gate
        if np.any(temporal_channel):
            finite &= (~temporal_channel) | (
                np.isfinite(reference_snr) & (reference_snr >= temporal_reference_gate)
            )
        filtered = np.column_stack(
            (
                np.asarray(array[:, :3], dtype=np.float64),
                temporal_channel.astype(np.float64),
                reference_snr,
            )
        )[finite]
        gated_row_count += int(filtered.shape[0])
        normalized.append(filtered)
    if audit_sink is not None:
        audit_sink["input_rows"] = int(input_row_count)
        audit_sink["rows_after_snr_gate"] = int(gated_row_count)
    if not normalized or not any(array.size for array in normalized):
        return ()
    if line_artifact_frames is not None and len(line_artifact_frames) != len(normalized):
        raise ValueError("line_artifact_frames length must match candidate_peak_frames")
    resolved_frame_times = (
        _normalise_motion_times(frame_times, len(normalized))
        if frame_times is not None
        else _observation_times_seconds(frame_analyses)
        if frame_analyses is not None and len(frame_analyses) == len(normalized)
        else None
    )

    # 不再把第 1 帧硬编码成唯一锚点。真实序列中某一帧可能被坏像素、
    # 亮线、重叠源或局部低响应遮住；如果只从第 1 帧起轨迹，这类源即使
    # 在后续 14 帧稳定出现，也没有机会进入共识。先在配准坐标中建立各帧
    # 候选的空间并集，再对并集逐帧匹配。新增锚点只来自距离现有锚点超过
    # link_radius_px 的位置，避免同一源被重复播种；最终仍要求 presence、
    # 形状、线状、静态拟合和质量轨迹去重等细筛条件。
    anchor_aligned_parts: list[np.ndarray] = []
    anchor_metadata_parts: list[np.ndarray] = []
    for frame_index, current in enumerate(normalized):
        if current.size == 0:
            continue
        shift_x, shift_y = cumulative_shifts[frame_index]
        current_aligned = current[:, :2] - np.asarray((shift_x, shift_y), dtype=np.float64)
        if anchor_aligned_parts:
            existing_aligned = np.vstack(anchor_aligned_parts)
            anchor_tree = cKDTree(existing_aligned)
            distances, _existing_indices = anchor_tree.query(
                current_aligned,
                k=1,
                distance_upper_bound=float(link_radius_px),
            )
            new_mask = ~np.isfinite(distances)
        else:
            new_mask = np.ones(current.shape[0], dtype=bool)
        if np.any(new_mask):
            anchor_aligned_parts.append(current_aligned[new_mask])
            anchor_metadata_parts.append(current[new_mask, 2:5])
    if not anchor_aligned_parts:
        return ()
    anchor_aligned = np.vstack(anchor_aligned_parts).astype(np.float64, copy=False)
    # anchors 的 x/y 仅保留配准坐标；源级元数据列仍是候选分数、通道和
    # 时间中值参考分数。后续输出点始终从具体帧的 current 行读取原始坐标。
    anchors = np.column_stack(
        (anchor_aligned, np.vstack(anchor_metadata_parts).astype(np.float64, copy=False))
    )
    anchor_tree = cKDTree(anchor_aligned)
    match_indices = np.full((anchors.shape[0], len(normalized)), -1, dtype=np.int32)
    match_scores = np.full((anchors.shape[0], len(normalized)), np.nan, dtype=np.float64)

    for frame_index in range(len(normalized)):
        current = normalized[frame_index]
        if current.size == 0:
            continue
        shift_x, shift_y = cumulative_shifts[frame_index]
        current_aligned = current[:, :2] - np.asarray((shift_x, shift_y), dtype=np.float64)
        current_tree = cKDTree(current_aligned)
        distances, current_indices = current_tree.query(
            anchor_aligned,
            k=1,
            distance_upper_bound=float(link_radius_px),
        )
        reverse_distances, reverse_indices = anchor_tree.query(
            current_aligned,
            k=1,
            distance_upper_bound=float(link_radius_px),
        )
        del reverse_distances
        valid = np.isfinite(distances) & (current_indices < len(current))
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            continue
        selected_current = current_indices[valid_indices].astype(np.intp, copy=False)
        reciprocal = (selected_current < len(reverse_indices)) & (
            reverse_indices[selected_current] == valid_indices
        )
        valid_indices = valid_indices[reciprocal]
        if valid_indices.size == 0:
            continue
        selected_current = current_indices[valid_indices].astype(np.intp, copy=False)
        # reciprocal 最近邻已经排除了大部分多对一情况；这里再按距离做一
        # 次显式的一对一选择，避免两个并集锚点在密集源区共享同一个峰。
        order = np.argsort(distances[valid_indices], kind="stable")
        used_current: set[int] = set()
        selected_anchors: list[int] = []
        selected_sources: list[int] = []
        for ordered_index in order:
            anchor_index = int(valid_indices[ordered_index])
            source_index = int(selected_current[ordered_index])
            if source_index in used_current:
                continue
            used_current.add(source_index)
            selected_anchors.append(anchor_index)
            selected_sources.append(source_index)
        if selected_anchors:
            selected_anchor_array = np.asarray(selected_anchors, dtype=np.intp)
            selected_source_array = np.asarray(selected_sources, dtype=np.intp)
            match_indices[selected_anchor_array, frame_index] = selected_source_array
            match_scores[selected_anchor_array, frame_index] = current[selected_source_array, 2]

    presence = np.count_nonzero(match_indices >= 0, axis=1)
    candidate_anchor_indices = np.flatnonzero(presence >= int(persistent_min_presence))
    if audit_sink is not None:
        audit_sink["anchor_total"] = int(anchors.shape[0])
        audit_sink["anchor_presence_ge_persistent"] = int(candidate_anchor_indices.size)
    if candidate_anchor_indices.size == 0:
        return ()

    # 对候选补充一个跨帧的 3×3 核心支持审计。候选峰分数只描述匹配滤波
    # 响应，固定读出纹理也可能在多帧重复；要求核心周围有多个同量级正
    # 像素，才能把“重复出现”与“重复单像素尖峰”分开。这里按帧向量化，
    # 不调用逐候选的完整测光函数，以免稳定候选补检重新变成分钟级。
    temporal_support: np.ndarray | None = None
    temporal_valid: np.ndarray | None = None
    temporal_line_overlap: np.ndarray | None = None
    temporal_shape_support: np.ndarray | None = None
    temporal_psf_fit_support: np.ndarray | None = None
    if frame_analyses is not None:
        if len(frame_analyses) != len(normalized):
            raise ValueError("frame_analyses length must match candidate_peak_frames")
        temporal_support = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_valid = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_line_overlap = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_shape_support = np.zeros(anchors.shape[0], dtype=np.int16)
        temporal_psf_fit_support = np.zeros(anchors.shape[0], dtype=np.int16)
        offsets_y, offsets_x = np.mgrid[-1:2, -1:2]
        offsets_y = offsets_y.reshape(-1)
        offsets_x = offsets_x.reshape(-1)
        required_support = max(
            3,
            int(frame_analyses[0].detection.parameters.get("min_psf_support_pixels", 3)),
        )
        aperture_radius = max(1, int(frame_analyses[0].detection.parameters.get("aperture_radius", 4)))
        aperture_offsets_y, aperture_offsets_x = np.mgrid[
            -aperture_radius : aperture_radius + 1,
            -aperture_radius : aperture_radius + 1,
        ]
        aperture_offsets_y = aperture_offsets_y.reshape(-1)
        aperture_offsets_x = aperture_offsets_x.reshape(-1)
        aperture_geometry = (
            (aperture_offsets_x.astype(np.float64) ** 2)
            + (aperture_offsets_y.astype(np.float64) ** 2)
            <= float(aperture_radius**2)
        )
        annulus_inner_radius = max(1.5, 0.75 * float(aperture_radius))
        annulus_geometry = aperture_geometry & (
            (aperture_offsets_x.astype(np.float64) ** 2)
            + (aperture_offsets_y.astype(np.float64) ** 2)
            >= annulus_inner_radius**2
        )
        minimum_annulus_pixels = max(8, int(np.ceil(0.25 * int(np.count_nonzero(aperture_geometry)))))
        try:
            temporal_psf_fwhm = float(frame_analyses[0].detection.parameters.get("psf_fwhm", 3.0))
        except (AttributeError, TypeError, ValueError):
            temporal_psf_fwhm = 3.0
        psf_fit_sigmas = tuple(
            max(fwhm / 2.35482, 0.5)
            for fwhm in _temporal_psf_fwhm_bank(temporal_psf_fwhm)
        )
        psf_fit_kernels = tuple(
            np.exp(
                -0.5
                * (
                    (aperture_offsets_x.astype(np.float64) ** 2)
                    + (aperture_offsets_y.astype(np.float64) ** 2)
                )
                / sigma**2
            )
            for sigma in psf_fit_sigmas
        )
        for frame_index, (current, analysis) in enumerate(zip(normalized, frame_analyses, strict=True)):
            active = candidate_anchor_indices[match_indices[candidate_anchor_indices, frame_index] >= 0]
            if active.size == 0:
                continue
            current_indices = match_indices[active, frame_index].astype(np.intp, copy=False)
            points = current[current_indices]
            integer_x = np.rint(points[:, 0]).astype(np.intp)
            integer_y = np.rint(points[:, 1]).astype(np.intp)
            height, width = analysis.frame.data.shape
            inside = (
                (integer_x >= 1)
                & (integer_x < width - 1)
                & (integer_y >= 1)
                & (integer_y < height - 1)
            )
            edge_margin = aperture_radius + 4
            inside &= (
                (integer_x >= edge_margin)
                & (integer_x < width - edge_margin)
                & (integer_y >= edge_margin)
                & (integer_y < height - edge_margin)
            )
            if not np.any(inside):
                continue
            active = active[inside]
            integer_x = integer_x[inside]
            integer_y = integer_y[inside]
            pixel_values = np.asarray(analysis.frame.data)[
                integer_y[:, None] + offsets_y[None, :],
                integer_x[:, None] + offsets_x[None, :],
            ].astype(np.float64, copy=False)
            aperture_values = np.asarray(analysis.frame.data)[
                integer_y[:, None] + aperture_offsets_y[None, :],
                integer_x[:, None] + aperture_offsets_x[None, :],
            ]
            global_background = float(analysis.detection.background)
            global_noise = max(float(analysis.detection.noise), np.finfo(np.float64).eps)
            # 零值/NaN 是像素级坏点，不能因为孔径中恰好有一个坏像素就
            # 把整帧观测判成无效。中心必须有效，圆形孔径仍需保留足够的
            # 有效像素；后续支持和形状统计只消费对应的有效像素。
            pixel_valid = np.isfinite(pixel_values)
            aperture_valid = np.isfinite(aperture_values)
            if int(analysis.detection.parameters.get("mask_zero_pixels", 0)):
                pixel_valid &= pixel_values != 0.0
                aperture_valid &= aperture_values != 0.0
            saturation = float(analysis.detection.parameters.get("saturation_level", -1.0))
            if saturation > 0:
                pixel_valid &= pixel_values < saturation
                aperture_valid &= aperture_values < saturation
            aperture_y = integer_y[:, None] + aperture_offsets_y[None, :]
            aperture_x = integer_x[:, None] + aperture_offsets_x[None, :]
            aperture_valid &= ~((aperture_y == 0) & (aperture_x < 104))
            core_y = integer_y[:, None] + offsets_y[None, :]
            core_x = integer_x[:, None] + offsets_x[None, :]
            pixel_valid &= ~((core_y == 0) & (core_x < 104))
            aperture_valid_count = np.count_nonzero(
                aperture_valid & aperture_geometry[None, :],
                axis=1,
            )
            minimum_aperture_valid = max(
                9,
                int(np.ceil(0.8 * int(np.count_nonzero(aperture_geometry)))),
            )
            finite = (
                pixel_valid[:, 4]
                & (aperture_valid_count >= minimum_aperture_valid)
            )
            # 候选共识阶段不能只减每帧一个全局背景。当前 FITS 存在明显的
            # 空间纹理/亮度梯度；若直接用全局背景，宽纹理会在 3×3 支持和
            # 二阶矩中伪装成“重复的星点”。用孔径外环估计每个候选的局部
            # 中位数和 MAD，并保留全局噪声作为无效/样本不足时的回退。
            annulus_valid = aperture_valid & annulus_geometry[None, :]
            annulus_count = np.count_nonzero(annulus_valid, axis=1)
            annulus_values = np.where(annulus_valid, aperture_values, np.nan).astype(np.float64, copy=False)
            with np.errstate(invalid="ignore"):
                local_background = np.nanmedian(annulus_values, axis=1)
                local_deviation = np.abs(annulus_values - local_background[:, None])
                local_noise = 1.4826 * np.nanmedian(local_deviation, axis=1)
            usable_annulus = (
                (annulus_count >= minimum_annulus_pixels)
                & np.isfinite(local_background)
                & np.isfinite(local_noise)
                & (local_noise > np.finfo(np.float64).eps)
            )
            local_background = np.where(usable_annulus, local_background, global_background)
            local_noise = np.where(usable_annulus, local_noise, global_noise)
            # MAD 在非常平坦的局部块中可能低估真实读出噪声；不让它低于
            # 全局噪声的 50%，避免把个别像素波动夸大成高显著性点源。
            local_noise = np.maximum(local_noise, 0.5 * global_noise)
            if line_artifact_frames is not None:
                raw_line_coordinates = np.asarray(line_artifact_frames[frame_index])
                if raw_line_coordinates.size:
                    if raw_line_coordinates.ndim != 2 or raw_line_coordinates.shape[1] < 2:
                        raise ValueError("line artifact coordinates must be an Nx2 array")
                    line_tree = cKDTree(
                        np.asarray(raw_line_coordinates[:, :2], dtype=np.float64)
                    )
                    line_distances, _line_indices = line_tree.query(
                        np.column_stack((integer_x, integer_y)),
                        k=1,
                        distance_upper_bound=float(aperture_radius),
                    )
                    line_overlap = np.isfinite(line_distances)
                    temporal_line_overlap[active[line_overlap]] += 1
                    finite &= ~line_overlap
            temporal_valid[active[finite]] += 1
            if not np.any(finite):
                continue
            valid_pixels = pixel_values[finite]
            valid_core = pixel_valid[finite]
            local_background_valid = local_background[finite]
            local_noise_valid = local_noise[finite]
            peak_excess = np.maximum(valid_pixels[:, 4] - local_background_valid, 0.0)
            support_threshold = np.maximum(2.0 * local_noise_valid, 0.1 * peak_excess)
            support_pixels = np.count_nonzero(
                (valid_pixels - local_background_valid[:, None] >= support_threshold[:, None]) & valid_core,
                axis=1,
            )
            temporal_support[active[finite]] += support_pixels >= required_support

            # 3×3 支持能排除单像素尖峰，但不能排除“宽而连续”的读出纹理。
            # 用与单帧源级测量相同的正残差二阶矩和 peak/positive_sum 指标
            # 做一个廉价形状细筛；这里只计数证据，不重算孔径通量。
            shape_values = np.where(
                aperture_valid[finite],
                aperture_values[finite].astype(np.float64, copy=False) - local_background_valid[:, None],
                0.0,
            )
            shape_geometry = aperture_geometry[None, :] & aperture_valid[finite]
            positive = np.where(
                shape_geometry,
                np.maximum(shape_values, 0.0),
                0.0,
            )
            positive_sum = positive.sum(axis=1)
            shape_peak = np.max(
                np.where(shape_geometry, shape_values, -np.inf),
                axis=1,
            )
            safe_sum = np.maximum(positive_sum, np.finfo(np.float64).eps)
            center_x = (positive * aperture_offsets_x[None, :]).sum(axis=1) / safe_sum
            center_y = (positive * aperture_offsets_y[None, :]).sum(axis=1) / safe_sum
            variance_x = (
                positive
                * (aperture_offsets_x[None, :] - center_x[:, None]) ** 2
            ).sum(axis=1) / safe_sum
            variance_y = (
                positive
                * (aperture_offsets_y[None, :] - center_y[:, None]) ** 2
            ).sum(axis=1) / safe_sum
            fwhm = 2.35482 * np.sqrt(np.maximum((variance_x + variance_y) / 2.0, 0.0))
            ellipticity = np.abs(np.sqrt(variance_x) - np.sqrt(variance_y)) / np.maximum(
                np.maximum(np.sqrt(variance_x), np.sqrt(variance_y)),
                np.finfo(np.float64).eps,
            )
            sharpness = np.maximum(shape_peak, 0.0) / safe_sum
            shape_pass = (
                np.isfinite(fwhm)
                & (fwhm >= 0.8)
                & (fwhm <= 4.5)
                & np.isfinite(ellipticity)
                & (ellipticity <= 0.65)
                & np.isfinite(sharpness)
                & (sharpness >= 0.08)
            )
            temporal_shape_support[active[finite]] += shape_pass
            # 仅靠 FWHM/椭圆率仍可能让低幅度宽纹理通过。计算局部孔径
            # 与窄/基准/宽 Gaussian 模板的归一化相关系数，取最相符尺度；
            # 这不是完整 PSF 拟合，只作为时序候选的点源形状证据。
            fit_values = np.where(shape_geometry, shape_values, 0.0)
            data_norm = np.sqrt(np.sum(fit_values**2, axis=1))
            best_correlation = np.full(fit_values.shape[0], -np.inf, dtype=np.float64)
            for kernel in psf_fit_kernels:
                template = np.where(shape_geometry, kernel[None, :], 0.0)
                template_norm = np.sqrt(np.sum(template**2, axis=1))
                denominator = data_norm * template_norm
                correlation = np.divide(
                    np.sum(fit_values * template, axis=1),
                    denominator,
                    out=np.full_like(denominator, -np.inf, dtype=np.float64),
                    where=denominator > np.finfo(np.float64).eps,
                )
                best_correlation = np.maximum(best_correlation, correlation)
            temporal_psf_fit_support[active[finite]] += (
                best_correlation >= float(temporal_min_psf_correlation)
            )

    # 只有已经达到持续门槛的质量轨迹才足以“解释掉”补检位置。此前
    # 任何只出现 1--3 帧的瞬态质量轨迹都会让整条 15 帧候选补检被去重，
    # 这会遮蔽真实但偶尔被局部峰/坏像素打断的星。短质量轨迹保留为
    # 独立瞬态证据，不得压制更强的跨帧共识。
    quality_by_frame: list[list[tuple[float, float]]] = [[] for _ in normalized]
    for track in quality_tracks:
        if track.presence < int(persistent_min_presence):
            continue
        for point in track.points:
            if 0 <= point.frame_index < len(quality_by_frame):
                quality_by_frame[point.frame_index].append((point.aligned_x, point.aligned_y))
    quality_trees = [cKDTree(points) if points else None for points in quality_by_frame]
    merge_radius = max(1.25, min(2.0, float(link_radius_px) * 0.5))

    rendered: list[SourceTrack] = []
    for anchor_index in candidate_anchor_indices:
        if audit_sink is not None:
            audit_sink["anchors_evaluated"] = audit_sink.get("anchors_evaluated", 0) + 1
        frame_indices = np.flatnonzero(match_indices[anchor_index] >= 0)
        points: list[TrackPoint] = []
        overlaps_quality = False
        for frame_index in frame_indices:
            current = normalized[frame_index]
            current_index = int(match_indices[anchor_index, frame_index])
            raw_x, raw_y, score = current[current_index, :3]
            shift_x, shift_y = cumulative_shifts[frame_index]
            point = TrackPoint(
                frame_index=int(frame_index),
                detection_id=-1 - int(anchor_index),
                x=float(raw_x),
                y=float(raw_y),
                aligned_x=float(raw_x - shift_x),
                aligned_y=float(raw_y - shift_y),
                flux_snr=None,
                quality_passed=False,
                candidate_snr=float(score),
            )
            points.append(point)
            tree = quality_trees[frame_index]
            if tree is not None:
                distance, _index = tree.query((point.aligned_x, point.aligned_y), k=1)
                if np.isfinite(distance) and float(distance) <= merge_radius:
                    overlaps_quality = True
        if overlaps_quality:
            if audit_sink is not None:
                audit_sink["reject_quality_overlap"] = audit_sink.get("reject_quality_overlap", 0) + 1
            continue
        if len(points) < int(persistent_min_presence):
            if audit_sink is not None:
                audit_sink["reject_presence"] = audit_sink.get("reject_presence", 0) + 1
            continue
        median_candidate_snr = float(np.nanmedian(match_scores[anchor_index, frame_indices]))
        anchor_is_temporal = bool(anchors[anchor_index, 3] >= 0.5)
        score_threshold = temporal_min_snr if anchor_is_temporal else float(min_candidate_snr)
        if not np.isfinite(median_candidate_snr) or median_candidate_snr < score_threshold:
            if audit_sink is not None:
                audit_sink["reject_median_candidate_snr"] = audit_sink.get("reject_median_candidate_snr", 0) + 1
            continue
        if anchor_is_temporal and (
            not np.isfinite(anchors[anchor_index, 4])
            or anchors[anchor_index, 4] < temporal_reference_gate
        ):
            if audit_sink is not None:
                audit_sink["reject_temporal_reference_snr"] = audit_sink.get("reject_temporal_reference_snr", 0) + 1
            continue
        if temporal_support is not None and temporal_valid is not None:
            if audit_sink is not None:
                for name, value in (
                    ("presence", len(points)),
                    ("temporal_valid", int(temporal_valid[anchor_index])),
                    ("temporal_support", int(temporal_support[anchor_index])),
                    (
                        "temporal_shape",
                        int(temporal_shape_support[anchor_index])
                        if temporal_shape_support is not None
                        else -1,
                    ),
                    (
                        "temporal_psf_correlation",
                        int(temporal_psf_fit_support[anchor_index])
                        if temporal_psf_fit_support is not None
                        else -1,
                    ),
                    (
                        "line_overlap",
                        int(temporal_line_overlap[anchor_index])
                        if temporal_line_overlap is not None
                        else 0,
                    ),
                ):
                    key = f"{name}_{value}"
                    audit_sink[key] = audit_sink.get(key, 0) + 1
            # 单个坏像素可能只影响一帧的孔径；要求大多数而非所有
            # 观测孔径可用，避免把真实但偶尔被坏点覆盖的暗星全部漏掉。
            minimum_valid = max(1, int(np.ceil(0.8 * len(points))))
            if temporal_valid[anchor_index] < minimum_valid:
                if audit_sink is not None:
                    audit_sink["reject_temporal_valid"] = audit_sink.get("reject_temporal_valid", 0) + 1
                continue
            if temporal_support[anchor_index] < max(1, int(np.ceil(0.6 * temporal_valid[anchor_index]))):
                if audit_sink is not None:
                    audit_sink["reject_temporal_support"] = audit_sink.get("reject_temporal_support", 0) + 1
                    support_key = f"support_reject_valid_{int(temporal_valid[anchor_index])}_support_{int(temporal_support[anchor_index])}"
                    audit_sink[support_key] = audit_sink.get(support_key, 0) + 1
                continue
            if temporal_line_overlap is not None:
                # 亮轨迹只擦过一两帧时，不能把其余 13--14 帧的稳定
                # 点源证据一并丢掉；但短序列保持保守，且整条轨迹被
                # 亮线覆盖时仍必须拒绝。允许的污染比例最多为 20%，
                # 最少需要 5 帧才启用“偶发擦过”例外。
                line_overlap_count = int(temporal_line_overlap[anchor_index])
                allowed_line_overlap = (
                    max(1, int(np.floor(0.2 * len(points))))
                    if len(points) >= 5
                    else 0
                )
                if line_overlap_count > allowed_line_overlap:
                    if audit_sink is not None:
                        audit_sink["reject_line_overlap"] = audit_sink.get("reject_line_overlap", 0) + 1
                    continue
            if temporal_shape_support is not None and temporal_valid[anchor_index] > 0:
                if temporal_shape_support[anchor_index] < max(
                    1,
                    int(np.ceil(0.6 * temporal_valid[anchor_index])),
                ):
                    if audit_sink is not None:
                        audit_sink["reject_temporal_shape"] = audit_sink.get("reject_temporal_shape", 0) + 1
                    continue
            if anchor_is_temporal and temporal_psf_fit_support is not None and temporal_valid[anchor_index] > 0:
                if temporal_psf_fit_support[anchor_index] < max(
                    1,
                    int(np.ceil(0.6 * temporal_valid[anchor_index])),
                ):
                    if audit_sink is not None:
                        audit_sink["reject_temporal_psf_correlation"] = audit_sink.get(
                            "reject_temporal_psf_correlation",
                            0,
                        ) + 1
                    continue
        displacement, speed, fit_rms = _fit_track(
            points,
            frame_times=resolved_frame_times,
        )
        if (
            fit_rms is None
            or fit_rms > float(max_motion_fit_rms_px)
            or displacement >= float(motion_min_displacement_px)
        ):
            # 候选峰也可能来自随帧移动的伪迹；不把它们塞进稳定场。
            if audit_sink is not None:
                audit_sink["reject_motion_fit"] = audit_sink.get("reject_motion_fit", 0) + 1
            continue
        rendered.append(
            SourceTrack(
                track_id=-1,
                classification="persistent",
                points=tuple(points),
                displacement_px=float(displacement),
                speed_px_per_frame=float(speed),
                fit_rms_px=float(fit_rms),
                evidence_level=(
                    "temporal_reference"
                    if anchor_is_temporal
                    else "candidate_consensus"
                ),
            )
        )
        if audit_sink is not None:
            audit_sink["accepted"] = audit_sink.get("accepted", 0) + 1
            key = "accepted_temporal_reference" if anchor_is_temporal else "accepted_candidate_consensus"
            audit_sink[key] = audit_sink.get(key, 0) + 1
    return tuple(rendered)


def _stack_forced_frame_measure(
    image: np.ndarray,
    mask: np.ndarray,
    x: float,
    y: float,
    *,
    aperture_radius: int,
    min_psf_support_pixels: int,
    psf_fwhm: float,
    global_background: float,
    global_noise: float,
) -> tuple[float | None, int, bool, float | None, float | None]:
    """在单帧原图 (x, y) 处做紧凑强制测光。

    叠加参考图把噪声压到约 ``1/√N``，但固定热像素和静态纹理也会一起
    保留；因此任何叠加候选都必须回到每帧原始图像做局部环测光、3×3 支持
    和点源形状审计，才能算作“该帧确实出现”。这里只做最小必要测量，
    复用与 ``_source_from_peak`` 相同的局部环背景、孔径通量、flux_snr、
    ``3×3`` 支持与二阶矩形状口径，避免为叠加候选再走一遍全帧背景网格。
    返回 ``(flux_snr, support, center_valid, fwhm, ellipticity)``。
    """

    height, width = image.shape
    outer = aperture_radius + 4
    y0 = max(0, int(np.floor(y)) - outer)
    y1 = min(height, int(np.floor(y)) + outer + 1)
    x0 = max(0, int(np.floor(x)) - outer)
    x1 = min(width, int(np.floor(x)) + outer + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None, 0, False, None, None
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool) | ~np.isfinite(patch)
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    center_x = x - x0
    center_y = y - y0
    distance = np.hypot(xx - center_x, yy - center_y)
    aperture = (distance <= aperture_radius) & ~patch_mask
    annulus = (distance > aperture_radius + 1) & (distance <= outer) & ~patch_mask
    annulus_values = patch[annulus]
    if annulus_values.size < 8:
        local_background = float(global_background)
        local_noise = max(float(global_noise), np.finfo(np.float64).eps)
    else:
        local_background, local_noise = sigma_clipped_stats(
            annulus_values,
            sample_limit=100_000,
        )
        local_noise = max(float(local_noise), np.finfo(np.float64).eps)
    # 局部环可能落在亮星翼部里，噪声被系统性抬高；但相对全局噪声过高的
    # 环不可靠，直接退回全局噪声，避免把翼部纹理误判成高显著性暗星。
    if local_noise > 3.0 * max(float(global_noise), np.finfo(np.float64).eps):
        local_noise = max(float(global_noise), np.finfo(np.float64).eps)
    residual = patch - local_background
    valid_aperture = residual[aperture]
    net_flux = float(valid_aperture.sum()) if valid_aperture.size else 0.0
    aperture_pixels = int(aperture.sum())
    if aperture_pixels < 3:
        return None, 0, False, None, None
    background_pixels = max(1, int(annulus.sum()))
    background_variance = aperture_pixels * local_noise**2
    background_variance += (aperture_pixels**2 / background_pixels) * local_noise**2
    flux_error = float(np.sqrt(max(background_variance, np.finfo(np.float64).eps)))
    flux_snr = net_flux / flux_error

    positive = np.where(aperture, np.maximum(residual, 0.0), 0.0)
    positive_sum = float(positive.sum())
    if positive_sum > 0:
        shape_x = float((positive * (xx + x0)).sum() / positive_sum)
        shape_y = float((positive * (yy + y0)).sum() / positive_sum)
        variance_x = float((positive * ((xx + x0) - shape_x) ** 2).sum() / positive_sum)
        variance_y = float((positive * ((yy + y0) - shape_y) ** 2).sum() / positive_sum)
        fwhm_x = 2.35482 * float(np.sqrt(max(0.0, variance_x)))
        fwhm_y = 2.35482 * float(np.sqrt(max(0.0, variance_y)))
        fwhm = 2.35482 * float(np.sqrt(max(0.0, (variance_x + variance_y) / 2.0)))
        ellipticity = abs(fwhm_x - fwhm_y) / max(fwhm_x, fwhm_y, np.finfo(np.float64).eps)
        if not np.isfinite(fwhm) or fwhm <= 0:
            fwhm = None
            ellipticity = None
    else:
        fwhm = None
        ellipticity = None

    core_x = int(np.rint(center_x))
    core_y = int(np.rint(center_y))
    center_valid = bool(not patch_mask[core_y, core_x]) if 0 <= core_y < patch.shape[0] and 0 <= core_x < patch.shape[1] else False
    if not center_valid:
        return flux_snr, 0, False, fwhm, ellipticity
    peak_excess = max(float(patch[core_y, core_x] - local_background), 0.0)
    core_y0, core_y1 = max(0, core_y - 1), min(patch.shape[0], core_y + 2)
    core_x0, core_x1 = max(0, core_x - 1), min(patch.shape[1], core_x + 2)
    core_slice = residual[core_y0:core_y1, core_x0:core_x1]
    core_valid = ~patch_mask[core_y0:core_y1, core_x0:core_x1]
    support_threshold = max(2.0 * local_noise, 0.1 * peak_excess)
    support = int(
        np.count_nonzero(
            core_valid
            & np.isfinite(core_slice)
            & (core_slice >= support_threshold)
        )
    )
    return flux_snr, support, True, fwhm, ellipticity


def _stack_forced_frame_measure_batch(
    image: np.ndarray,
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    aperture_radius: int,
    min_psf_support_pixels: int,
    global_background: float,
    global_noise: float,
    padded_image: np.ndarray | None = None,
    padded_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """向量化的单帧强制测光，等价于对每个坐标调用 ``_stack_forced_frame_measure``。

    输入 ``xs/ys`` 为同长度的原始坐标数组；返回逐源的
    ``(flux_snr, support, center_valid, fwhm, ellipticity)`` 数组，口径与
    标量版本一致。相比逐个候选调用标量函数，这里把 4096² 背景环、孔径
    通量、二阶矩和 3×3 支持全部做成按坐标广播的数组运算，减少数万次
    Python 调用与重复的小数组分配。

    局部环背景用一次稳健中位数/MAD/标准差估计：对干净背景环（绝大多数
    弱源的情形）与 ``sigma_clipped_stats`` 的首轮结果一致；极少数被近邻
    污染时会退回全局背景/噪声，再由调用方的出现帧数门槛吸收。
    """

    count = int(xs.size)
    if count == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    outer = int(aperture_radius) + 4
    side = 2 * outer + 1
    pad = outer
    height, width = image.shape
    # 图像用 NaN 填充边界，掩膜用 True（无效）填充边界；这样边缘候选的
    # 孔径自然失去边界外的像素，无需逐源裁剪分支。叠加暗星阶段会把同一帧
    # 的候选分成多个块，这两个数组由调用方一次准备并复用，避免每个块重复
    # 分配并复制一张 4096² 图像。
    expected_padded_shape = (height + 2 * pad, width + 2 * pad)
    if padded_image is None:
        padded = np.pad(np.asarray(image, dtype=np.float64), pad, mode="constant", constant_values=np.nan)
    else:
        padded = np.asarray(padded_image, dtype=np.float64)
        if padded.shape != expected_padded_shape:
            raise ValueError("padded_image shape does not match image and aperture radius")
    if padded_mask is None:
        padded_mask_array = np.pad(np.asarray(mask, dtype=bool), pad, mode="constant", constant_values=True)
    else:
        padded_mask_array = np.asarray(padded_mask, dtype=bool)
        if padded_mask_array.shape != expected_padded_shape:
            raise ValueError("padded_mask shape does not match image and aperture radius")

    # 叠加参考图覆盖的坐标可能落在某一帧视场之外（例如累计漂移把
    # 参考图底部候选投回到当前帧之外）。先把用于数组索引的中心裁到
    # 安全范围，最后再用 ``in_frame`` 把这些源标成无效；不能让一个
    # 合法的“视场外候选”把整批向量化测光变成 IndexError。
    finite_coordinates = np.isfinite(xs) & np.isfinite(ys)
    in_frame = (
        finite_coordinates
        & (xs >= 0.0)
        & (xs <= float(width - 1))
        & (ys >= 0.0)
        & (ys <= float(height - 1))
    )
    safe_xs = np.where(finite_coordinates, xs, 0.0)
    safe_ys = np.where(finite_coordinates, ys, 0.0)
    raw_floor_x = np.floor(safe_xs).astype(np.intp)
    raw_floor_y = np.floor(safe_ys).astype(np.intp)
    floor_x = np.clip(raw_floor_x, 0, width - 1)
    floor_y = np.clip(raw_floor_y, 0, height - 1)
    dx_sub = np.where(finite_coordinates, safe_xs - raw_floor_x, 0.0)
    dy_sub = np.where(finite_coordinates, safe_ys - raw_floor_y, 0.0)

    # 相对整数中心的偏移网格：元素 (py, px) 相对中心的偏移为
    # (py - outer, px - outer)。
    oy = np.arange(side, dtype=np.float64)[:, None] - outer  # (side, 1)
    ox = np.arange(side, dtype=np.float64)[None, :] - outer  # (1, side)
    distance = np.hypot(
        ox[None, :, :] - dx_sub[:, None, None],
        oy[None, :, :] - dy_sub[:, None, None],
    )
    aperture = distance <= float(aperture_radius)
    annulus = (distance > float(aperture_radius) + 1.0) & (distance <= float(outer))

    row_base = floor_y[:, None, None] + pad  # (count,1,1) -> padded rows
    col_base = floor_x[:, None, None] + pad
    rows = row_base + (np.arange(side, dtype=np.intp)[None, :, None] - outer)
    cols = col_base + (np.arange(side, dtype=np.intp)[None, None, :] - outer)
    patches = padded[rows, cols]  # (count, side, side)
    patch_mask = padded_mask_array[rows, cols] | ~np.isfinite(patches)

    aperture &= ~patch_mask
    annulus &= ~patch_mask
    annulus_count = annulus.sum(axis=(1, 2))
    # 迭代式稳健裁剪，与标量版的 sigma_clipped_stats 对齐：首轮求中位数
    # 和 MAD，按 3×MAD 保留像素，再重算；对干净背景环，后续迭代无变化，
    # 对亮星翼部/近邻污染则逐步剔除异常值。固定三轮，向量化完成。
    annulus_values = np.where(annulus, patches, np.nan)
    working = annulus_values
    annulus_median = np.full(count, np.nan, dtype=np.float64)
    annulus_mad = np.full(count, np.nan, dtype=np.float64)
    annulus_std = np.full(count, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(3):
            center = np.nanmedian(working, axis=(1, 2))
            deviation = np.abs(working - center[:, None, None])
            scale = 1.4826 * np.nanmedian(deviation, axis=(1, 2))
            keep = np.isfinite(working) & (deviation <= 3.0 * scale[:, None, None])
            kept_all = keep.all(axis=(1, 2))
            annulus_median = np.where(np.isfinite(center), center, annulus_median)
            annulus_mad = np.where(np.isfinite(scale), scale, annulus_mad)
            std = np.sqrt(np.nanmean((working - center[:, None, None]) ** 2, axis=(1, 2)))
            annulus_std = np.where(np.isfinite(std), std, annulus_std)
            if np.all(kept_all):
                break
            working = np.where(keep, working, np.nan)
    local_noise = np.maximum.reduce(
        (
            annulus_mad,
            annulus_std,
            np.full(count, np.finfo(np.float64).eps),
        )
    )
    usable_annulus = annulus_count >= 8
    local_background = np.where(usable_annulus, annulus_median, float(global_background))
    local_noise = np.where(usable_annulus, local_noise, max(float(global_noise), np.finfo(np.float64).eps))
    # 局部环被亮星翼部抬高时退回全局噪声，与标量版本一致。
    inflated = local_noise > 3.0 * max(float(global_noise), np.finfo(np.float64).eps)
    local_noise = np.where(inflated, max(float(global_noise), np.finfo(np.float64).eps), local_noise)
    local_noise = np.maximum(local_noise, np.finfo(np.float64).eps)

    residual = patches - local_background[:, None, None]
    # 边缘候选的 patch 含 NaN 填充；NaN * False 仍为 NaN，会污染求和。
    # 用 np.where 只在有效孔径内取残差，等价于标量版的 ``residual[aperture]``。
    aperture_residual = np.where(aperture, residual, 0.0)
    net_flux = aperture_residual.sum(axis=(1, 2))
    aperture_pixels = aperture.sum(axis=(1, 2))
    background_pixels = np.maximum(annulus_count, 1).astype(np.float64)
    background_variance = aperture_pixels * local_noise**2
    background_variance += (aperture_pixels**2 / background_pixels) * local_noise**2
    flux_error = np.sqrt(np.maximum(background_variance, np.finfo(np.float64).eps))
    flux_snr = net_flux / flux_error

    positive = np.where(aperture, np.maximum(residual, 0.0), 0.0)
    positive_sum = positive.sum(axis=(1, 2))
    valid_shape = positive_sum > 0
    safe_sum = np.where(valid_shape, positive_sum, 1.0)
    shape_x = (positive * (cols - pad)).sum(axis=(1, 2)) / safe_sum
    shape_y = (positive * (rows - pad)).sum(axis=(1, 2)) / safe_sum
    variance_x = (positive * ((cols - pad) - shape_x[:, None, None]) ** 2).sum(axis=(1, 2)) / safe_sum
    variance_y = (positive * ((rows - pad) - shape_y[:, None, None]) ** 2).sum(axis=(1, 2)) / safe_sum
    fwhm_x = 2.35482 * np.sqrt(np.maximum(variance_x, 0.0))
    fwhm_y = 2.35482 * np.sqrt(np.maximum(variance_y, 0.0))
    fwhm = 2.35482 * np.sqrt(np.maximum((variance_x + variance_y) / 2.0, 0.0))
    ellipticity = np.abs(fwhm_x - fwhm_y) / np.maximum.reduce((fwhm_x, fwhm_y, np.full(count, np.finfo(np.float64).eps)))
    fwhm = np.where(valid_shape & np.isfinite(fwhm) & (fwhm > 0), fwhm, np.nan)
    ellipticity = np.where(valid_shape & np.isfinite(ellipticity), ellipticity, np.nan)

    # 中心有效性：整数舍入后的中心像素，与标量版本一致。
    center_ix = np.clip(np.rint(safe_xs).astype(np.intp), 0, width - 1)
    center_iy = np.clip(np.rint(safe_ys).astype(np.intp), 0, height - 1)
    center_valid = ~padded_mask_array[center_iy + pad, center_ix + pad] & np.isfinite(padded[center_iy + pad, center_ix + pad])
    center_valid &= in_frame
    center_patch_y = center_iy - floor_y + outer
    center_patch_x = center_ix - floor_x + outer
    peak_excess = np.maximum(patches[np.arange(count), center_patch_y, center_patch_x] - local_background, 0.0)
    support_threshold = np.maximum(2.0 * local_noise, 0.1 * peak_excess)
    core_offsets = np.array(
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)],
        dtype=np.intp,
    )
    support = np.zeros(count, dtype=np.int16)
    for dy, dx in core_offsets:
        core_y = center_patch_y + dy
        core_x = center_patch_x + dx
        valid_core = (
            (core_y >= 0)
            & (core_y < side)
            & (core_x >= 0)
            & (core_x < side)
            & ~patch_mask[np.arange(count), core_y, core_x]
        )
        support += (
            valid_core
            & (residual[np.arange(count), core_y, core_x] >= support_threshold)
        ).astype(np.int16)
    outside = ~in_frame
    flux_snr[outside] = np.nan
    support[outside] = 0
    center_valid[outside] = False
    fwhm[outside] = np.nan
    ellipticity[outside] = np.nan
    return flux_snr, support, center_valid, fwhm, ellipticity


def _stack_faint_tracks(
    frame_analyses: Sequence[FrameAnalysis],
    cumulative_shifts: Sequence[tuple[float, float]],
    reference: np.ndarray | None,
    *,
    threshold_sigma: float,
    min_distance: int,
    aperture_radius: int,
    psf_fwhm: float,
    min_flux_snr: float,
    min_psf_support_pixels: int,
    proposal_mode: str,
    stack_frame_min_flux_snr: float,
    min_presence: int,
    link_radius_px: float,
    existing_tracks: Sequence[SourceTrack],
    audit_sink: MutableMapping[str, int] | None = None,
    workers: int = DEFAULT_SEQUENCE_WORKERS,
    chunk_size: int = 8_192,
    max_reference_sources: int | None = DEFAULT_SEQUENCE_STACK_WORKING_LIMIT,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[SourceTrack, ...]:
    """从注册中值/稳健叠加参考图恢复单帧漏掉的暗星。

    叠加把噪声降到约 ``1/√N``，能把单帧 ``flux_snr<5`` 的暗星抬到参考图
    ``flux_snr≥5``；但叠加也保留固定热像素和静态纹理，所以这里不把参考图
    检测直接当结果。流程：参考图局部测光 → 与已有质量/共识轨迹按
    ``link_radius_px`` 去重 → 回到 15 帧原图做强制测光/3×3 支持/形状审计
    → 出现帧数达到 ``min_presence`` 且逐帧 ``flux_snr`` 达到放宽门槛。
    结果标为低置信 ``evidence_level="stack_faint"`` 轨迹，分类为
    ``persistent``，不与单帧质量源或严格静态源混写。

    参考图坐标已对齐到第 1 帧；第 i 帧的原始坐标是
    ``(x + shift_x, y + shift_y)``。
    """

    if reference is None or len(frame_analyses) < 2:
        return ()
    if max_reference_sources is not None and int(max_reference_sources) < 1:
        raise ValueError("max_reference_sources must be positive or None")
    if len(frame_analyses) != len(cumulative_shifts):
        raise ValueError("cumulative_shifts length must match frame_analyses")
    resolved_frame_times = _observation_times_seconds(frame_analyses)
    # 参考图本身是 float32（注册中值/稳健均值）。叠加上只剩“再筛一遍”，
    # 不追求单帧式环形精修；使用 float32 + 网格背景即可显著降低 4096²
    # 中间阵列的内存带宽，避免 GUI 后台在 15 帧之上再加一次 float64 全图
    # 检测而耗尽内存。参考图候选仍需逐帧原图强制测光，最终质量不依赖这
    # 一层测光的绝对精度。
    values = np.asarray(reference, dtype=np.float32)
    if values.ndim != 2:
        return ()
    reference_mask = ~np.isfinite(values)
    # 第一行辅助区在参考图中已经是 NaN；这里不再重复屏蔽，但防御性地
    # 保持与单帧一致的口径：仅当参考图没有 NaN 时才依赖 detect_sources
    # 自身的有限掩膜。
    try:
        stack_detection = detect_sources(
            values,
            mask=reference_mask,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            max_sources=None,
            psf_fwhm=psf_fwhm,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            proposal_mode=proposal_mode,
            refine_local_background=False,
            use_float32=True,
            fast_sequence=True,
            mask_zero_pixels=False,
            allow_partial_zero_mask=False,
        )
    except ValueError:
        return ()
    all_stack_candidates = tuple(stack_detection.quality_sources)
    # ``quality_sources`` 已按候选响应从高到低返回。这里不重新跑检测，
    # 只截取高 SNR 参考源进入逐帧强制测光；暗星恢复的最终证据仍来自
    # 原始 15 帧的出现帧数、通量 SNR、PSF 形状和掩膜审计。
    if max_reference_sources is None:
        stack_candidates = all_stack_candidates
    else:
        stack_candidates = all_stack_candidates[: int(max_reference_sources)]
    if audit_sink is not None:
        audit_sink["stack_reference_quality_sources"] = len(all_stack_candidates)
        audit_sink["stack_reference_working_sources"] = len(stack_candidates)
    if not stack_candidates:
        return ()

    # 去重：任何已经被质量轨迹或候选共识轨迹解释过的位置不再重复输出。
    explained: list[tuple[float, float]] = []
    for track in existing_tracks:
        for point in track.points:
            explained.append((point.aligned_x, point.aligned_y))
    if explained:
        tree = cKDTree(np.asarray(explained, dtype=np.float64))
    else:
        tree = None

    # 逐帧原始图、有效掩膜和全局背景/噪声只准备一次；若在候选循环内
    # 重复构造 4096² 的辅助掩膜或 ``image == 0`` 布尔阵列，数千候选会
    # 放大成数百万次整图分配，把这一层拖到不可接受的耗时。
    frame_images = tuple(np.asarray(analysis.frame.data) for analysis in frame_analyses)
    frame_masks = tuple(
        (
            auxiliary_mask(image.shape)
            | (image == 0)
            if int(analysis.detection.parameters.get("mask_zero_pixels", 0))
            else auxiliary_mask(image.shape)
        )
        for image, analysis in zip(frame_images, frame_analyses, strict=True)
    )
    frame_backgrounds = tuple(float(analysis.detection.background) for analysis in frame_analyses)
    frame_noises = tuple(float(analysis.detection.noise) for analysis in frame_analyses)

    ref_xs = np.asarray(
        [float(candidate.peak_x if candidate.peak_x is not None else candidate.x) for candidate in stack_candidates],
        dtype=np.float64,
    )
    ref_ys = np.asarray(
        [float(candidate.peak_y if candidate.peak_y is not None else candidate.y) for candidate in stack_candidates],
        dtype=np.float64,
    )
    filter_snrs = np.asarray(
        [float(candidate.filter_snr) if candidate.filter_snr is not None else np.nan for candidate in stack_candidates],
        dtype=np.float64,
    )
    # 去重：一次性查询所有候选到“已解释位置”的最近距离，而不是逐个查询。
    explained_mask = np.zeros(len(stack_candidates), dtype=bool)
    if tree is not None:
        distances, _indices = tree.query(np.column_stack((ref_xs, ref_ys)), k=1)
        explained_mask = np.isfinite(distances) & (distances <= float(link_radius_px))
    if audit_sink is not None:
        audit_sink["reject_stack_explained"] = int(explained_mask.sum())
    keep = ~explained_mask
    kept_xs = ref_xs[keep]
    kept_ys = ref_ys[keep]
    kept_filter_snrs = filter_snrs[keep]
    kept_candidates = [candidate for candidate, is_kept in zip(stack_candidates, keep, strict=True) if is_kept]
    if kept_xs.size == 0:
        return ()

    # 逐帧向量化强制测光：每一帧只做一次按坐标广播的数组计算，再按帧并行。
    # 对约 4.9 万个候选 × 15 帧，标量版本需要约 73 万次 Python 调用；这里
    # 降到每帧一次（内部再分块），峰值内存可控。
    frame_count = len(frame_analyses)
    chunk_size = max(1, int(chunk_size))

    def measure_frame(
        item: tuple[int, np.ndarray, np.ndarray, float, float, tuple[float, float]],
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_index, image, mask, background, noise, shift = item
        # 该帧会被数万候选分成多个块测量；边界填充与掩膜填充与候选坐标无关，
        # 因此只做一次。测量函数仍保留无缓存参数路径，保证普通调用口径不变。
        pad = int(aperture_radius) + 4
        padded_image = np.pad(
            np.asarray(image, dtype=np.float64),
            pad,
            mode="constant",
            constant_values=np.nan,
        )
        padded_mask = np.pad(
            np.asarray(mask, dtype=bool),
            pad,
            mode="constant",
            constant_values=True,
        )
        raw_x = kept_xs + float(shift[0])
        raw_y = kept_ys + float(shift[1])
        flux_snr_parts: list[np.ndarray] = []
        support_parts: list[np.ndarray] = []
        center_valid_parts: list[np.ndarray] = []
        fwhm_parts: list[np.ndarray] = []
        ellipticity_parts: list[np.ndarray] = []
        for start in range(0, kept_xs.size, chunk_size):
            stop = min(kept_xs.size, start + chunk_size)
            chunk_flux, chunk_support, chunk_valid, chunk_fwhm, chunk_ell = _stack_forced_frame_measure_batch(
                image,
                mask,
                raw_x[start:stop],
                raw_y[start:stop],
                aperture_radius=aperture_radius,
                min_psf_support_pixels=min_psf_support_pixels,
                global_background=background,
                global_noise=noise,
                padded_image=padded_image,
                padded_mask=padded_mask,
            )
            flux_snr_parts.append(chunk_flux)
            support_parts.append(chunk_support)
            center_valid_parts.append(chunk_valid)
            fwhm_parts.append(chunk_fwhm)
            ellipticity_parts.append(chunk_ell)
        return (
            frame_index,
            np.concatenate(flux_snr_parts) if flux_snr_parts else np.empty(0, dtype=np.float64),
            np.concatenate(support_parts) if support_parts else np.empty(0, dtype=np.int16),
            np.concatenate(center_valid_parts) if center_valid_parts else np.empty(0, dtype=bool),
            np.concatenate(fwhm_parts) if fwhm_parts else np.empty(0, dtype=np.float64),
            np.concatenate(ellipticity_parts) if ellipticity_parts else np.empty(0, dtype=np.float64),
        )

    work_items = tuple(
        (
            frame_index,
            frame_images[frame_index],
            frame_masks[frame_index],
            frame_backgrounds[frame_index],
            frame_noises[frame_index],
            cumulative_shifts[frame_index],
        )
        for frame_index in range(frame_count)
    )
    worker_count = min(max(1, int(workers)), frame_count)
    if worker_count == 1:
        measurements = []
        for completed, item in enumerate(map(measure_frame, work_items), start=1):
            measurements.append(item)
            if progress is not None:
                progress(completed, frame_count, "暗星逐帧强制测光")
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="rst19-stack") as executor:
            measurements = []
            for completed, item in enumerate(executor.map(measure_frame, work_items), start=1):
                measurements.append(item)
                if progress is not None:
                    progress(completed, frame_count, "暗星逐帧强制测光")
    measurements.sort(key=lambda item: item[0])
    frame_flux = [item[1] for item in measurements]
    frame_support = [item[2] for item in measurements]
    frame_center_valid = [item[3] for item in measurements]
    frame_fwhm = [item[4] for item in measurements]
    frame_ellipticity = [item[5] for item in measurements]

    rendered: list[SourceTrack] = []
    for candidate_index, candidate in enumerate(kept_candidates):
        ref_x = float(kept_xs[candidate_index])
        ref_y = float(kept_ys[candidate_index])
        points: list[TrackPoint] = []
        frame_flux_snrs: list[float] = []
        for frame_index in range(frame_count):
            flux_snr = frame_flux[frame_index][candidate_index]
            support = int(frame_support[frame_index][candidate_index])
            center_valid = bool(frame_center_valid[frame_index][candidate_index])
            fwhm = frame_fwhm[frame_index][candidate_index]
            ellipticity = frame_ellipticity[frame_index][candidate_index]
            if not center_valid:
                continue
            if support < min_psf_support_pixels:
                continue
            if not np.isfinite(flux_snr) or flux_snr < float(stack_frame_min_flux_snr):
                continue
            if not np.isfinite(fwhm) or not 0.8 <= fwhm <= 12.0:
                continue
            if not np.isfinite(ellipticity) or ellipticity > 0.65:
                continue
            shift_x, shift_y = cumulative_shifts[frame_index]
            points.append(
                TrackPoint(
                    frame_index=int(frame_index),
                    detection_id=-1,
                    x=float(ref_x + shift_x),
                    y=float(ref_y + shift_y),
                    aligned_x=ref_x,
                    aligned_y=ref_y,
                    flux_snr=float(flux_snr),
                    quality_passed=False,
                    candidate_snr=float(kept_filter_snrs[candidate_index])
                    if np.isfinite(kept_filter_snrs[candidate_index])
                    else float(flux_snr),
                )
            )
            frame_flux_snrs.append(float(flux_snr))
        if len(points) < int(min_presence):
            if audit_sink is not None:
                audit_sink["reject_stack_presence"] = audit_sink.get("reject_stack_presence", 0) + 1
            continue
        # 少数帧被亮星翼部抬噪导致的低 flux_snr 可以吸收；要求多数出现帧
        # 有正的稳定通量，避免只凭 3×3 支持就把固定纹理算成星。
        median_frame_flux_snr = float(np.median(frame_flux_snrs)) if frame_flux_snrs else 0.0
        if median_frame_flux_snr < float(stack_frame_min_flux_snr):
            if audit_sink is not None:
                audit_sink["reject_stack_flux_snr"] = audit_sink.get("reject_stack_flux_snr", 0) + 1
            continue
        displacement, speed, fit_rms = _fit_track(
            points,
            frame_times=resolved_frame_times,
        )
        rendered.append(
            SourceTrack(
                track_id=-1,
                classification="persistent",
                points=tuple(points),
                displacement_px=float(displacement),
                speed_px_per_frame=float(speed),
                fit_rms_px=float(fit_rms),
                evidence_level="stack_faint",
            )
        )
        if audit_sink is not None:
            audit_sink["accepted_stack_faint"] = audit_sink.get("accepted_stack_faint", 0) + 1
    return tuple(rendered)


def _fit_sequence_relative_photometry(
    analyses: Sequence[FrameAnalysis],
    tracks: Sequence[SourceTrack],
    *,
    min_presence: int,
    max_sources: int,
    spatial_order: int,
    validation_fraction: float,
) -> RelativePhotometryResult:
    """Fit a compact relative scale from stable quality tracks.

    The dense design used by :func:`fit_relative_photometry` is intentionally
    capped here.  A 4096×4096 frame can contain thousands of detections; using
    every source would make the source/frame design matrix unnecessarily large.
    We therefore choose the highest-SNR static tracks that span the sequence,
    while keeping the selection deterministic.  This is a relative scale only:
    it does not manufacture a Gaia zero point or an absolute magnitude.
    """

    if max_sources < 1:
        raise ValueError("relative_photometry_max_sources must be positive")
    track_by_observation: dict[tuple[int, int], SourceTrack] = {}
    eligible: list[tuple[float, int, SourceTrack]] = []
    for track in tracks:
        if track.classification != "static" or track.presence < min_presence:
            continue
        snr_values = [
            float(point.flux_snr)
            for point in track.points
            if point.flux_snr is not None and np.isfinite(float(point.flux_snr)) and float(point.flux_snr) > 0
        ]
        if not snr_values:
            continue
        eligible.append((float(np.median(np.asarray(snr_values))), int(track.track_id), track))
        for point in track.points:
            track_by_observation[(int(point.frame_index), int(point.detection_id))] = track
    eligible.sort(key=lambda item: (-item[0], item[1]))
    selected = {track_id for _snr, track_id, _track in eligible[:max_sources]}

    observations: list[Observation] = []
    for frame_index, analysis in enumerate(analyses):
        exposure_s = exposure_seconds(analysis.frame.header)
        for source in analysis.detection.quality_sources:
            track = track_by_observation.get((frame_index, int(source.detection_id)))
            if track is None or int(track.track_id) not in selected:
                continue
            magnitude = instrumental_magnitude(float(source.flux), exposure_s=exposure_s)
            signal_snr = source.flux_snr if source.flux_snr is not None else source.snr
            if magnitude is None or signal_snr is None or not np.isfinite(float(signal_snr)) or float(signal_snr) <= 0:
                continue
            observations.append(
                Observation(
                    frame_id=frame_index,
                    source_id=f"track:{track.track_id}",
                    instrumental_magnitude=float(magnitude),
                    magnitude_error=float(1.0857362047581296 / float(signal_snr)),
                    x=float(source.x),
                    y=float(source.y),
                    quality_passed=True,
                    is_moving=False,
                )
            )

    return fit_relative_photometry(
        observations,
        spatial_order=spatial_order,
        validation_fraction=validation_fraction,
        random_state=0,
        min_sources=2,
        min_frames=2,
    )


def analyze_sequence(
    paths: Iterable[str | Path],
    *,
    link_radius_px: float = 4.0,
    min_presence: int | None = None,
    motion_min_displacement_px: float = 2.0,
    motion_line_link_radius_px: float | None = None,
    registration_radius_px: float = 8.0,
    max_motion_fit_rms_px: float = 0.75,
    persistent_min_presence: int | None = None,
    fast_point_motion: bool = True,
    fast_point_min_snr: float = DEFAULT_FAST_POINT_MIN_SNR,
    fast_point_max_step_px: float = DEFAULT_FAST_POINT_MAX_STEP_PX,
    fast_point_gate_px: float = DEFAULT_FAST_POINT_GATE_PX,
    fast_point_max_fit_rms_px: float = DEFAULT_FAST_POINT_FIT_RMS_PX,
    fast_point_candidate_limit: int = DEFAULT_FAST_POINT_CANDIDATE_WORKING_LIMIT,
    motion_multiscale_line_proposals: bool = True,
    sequence_max_sources: int | None = DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT,
    sequence_workers: int = DEFAULT_SEQUENCE_WORKERS,
    candidate_consensus_min_snr: float = 15.0,
    temporal_candidate_min_snr: float | None = None,
    temporal_reference_min_snr: float | None = None,
    temporal_multiscale: bool = False,
    temporal_min_psf_correlation: float = 0.8,
    temporal_proposal_mode: str = "median",
    stack_faint_recovery: bool = True,
    stack_reference_mode: str = "median",
    stack_threshold_sigma: float = 4.0,
    stack_min_flux_snr: float = 5.0,
    stack_frame_min_flux_snr: float = 3.0,
    stack_min_presence: int | None = None,
    relative_photometry: bool = False,
    relative_photometry_max_sources: int = 400,
    relative_photometry_spatial_order: int = 0,
    relative_photometry_validation_fraction: float = 0.2,
    progress: Callable[[str, int, int], None] | None = None,
    detail_progress: Callable[[int, int, float, str], None] | None = None,
    **detector_kwargs: object,
) -> SequenceResult:
    """读取并检测一组 FITS，再关联质量源。

    序列默认只对每帧按匹配滤波 SNR 排名前 ``sequence_max_sources`` 个源
    做源级测量和点轨迹关联，但仍保留匹配滤波得到的完整 ``candidate_count``。
    传入 ``None`` 可恢复全量序列工作集；单图 ``analyze_frame`` 不受此默认值影响。
    快速路径还保留每帧完整的匹配滤波峰坐标。序列关联完成后，会对没有
    进入单帧质量层、但在配准坐标中持续出现的峰做一次廉价的一对一共识
    检验，作为 ``persistent`` 低置信稳定候选；它不改变单帧质量计数。
    ``temporal_reference_min_snr`` 只作用于时间中值/稳健叠加提出的补充坐标，
    不能降低普通逐帧候选的 SNR 门槛；所有补提案仍需逐帧原图支持、形状、
    掩膜和持续性审计。
    ``temporal_multiscale=True`` 只扩大时间参考图的宽筛 PSF 尺度组，默认
    关闭；它适合研究窄 PSF 漏检，不能仅凭候选数增加就当作正式星表口径。
    ``motion_multiscale_line_proposals=True`` 会在每一帧增加当前图像的
    多尺度线段空间提议，再交给原有的跨帧关联、形态和拟合门；它不会把
    单帧提议直接标成运动，也不使用时间参考图替代跨帧证据。
    序列还会输出固定像素值审计及其孔径影响关联；这些字段只用于定位
    坏像素/填充值对候选测量的影响，不会自动屏蔽固定值。
    运动判定会在平移配准后尝试拟合小范围全局仿射残差，并以稳定多数
    轨迹的 robust MAD 估计一个有效位移下限；这能避免把视场旋转造成的
    2--13 px 亚像素漂移批量标成 moving。该下限只影响 moving 分类，
    不会改写用户输入的 ``motion_min_displacement_px``。
    ``relative_photometry=True`` 时，序列结束前会从跨帧静态质量轨迹中
    选择有限个高 SNR 参考源，拟合帧间相对零点和源相对星等；该结果不
    需要联网，也不声称已经完成 Gaia/标准系统或绝对星等标定。
    整数 FITS 默认启用序列快速路径：每个 256 px 背景块最多抽样
    ``DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT`` 个像素，并在稀疏掩膜时
    使用单遍匹配滤波；快速口径会写入返回结果，单图默认不启用。
    """

    frame_paths = tuple(Path(path) for path in paths)
    if not frame_paths:
        raise ValueError("paths cannot be empty")
    total = len(frame_paths)
    if sequence_max_sources is not None and sequence_max_sources < 1:
        raise ValueError("sequence_max_sources must be positive or None")
    if sequence_workers < 1:
        raise ValueError("sequence_workers must be positive")
    if fast_point_min_snr <= 0 or fast_point_max_step_px <= 0 or fast_point_gate_px <= 0 or fast_point_max_fit_rms_px <= 0:
        raise ValueError("fast point motion thresholds must be positive")
    if fast_point_candidate_limit < 1:
        raise ValueError("fast_point_candidate_limit must be positive")
    if candidate_consensus_min_snr <= 0:
        raise ValueError("candidate_consensus_min_snr must be positive")
    temporal_proposal_mode = str(temporal_proposal_mode).strip().lower()
    if temporal_proposal_mode not in {"median", "coadd", "both"}:
        raise ValueError("temporal_proposal_mode must be 'median', 'coadd', or 'both'")
    resolved_temporal_reference_min_snr = (
        float(candidate_consensus_min_snr)
        if temporal_reference_min_snr is None
        else float(temporal_reference_min_snr)
    )
    resolved_temporal_candidate_min_snr = (
        max(5.0, 0.5 * float(candidate_consensus_min_snr))
        if temporal_candidate_min_snr is None
        else float(temporal_candidate_min_snr)
    )
    if (
        resolved_temporal_candidate_min_snr <= 0
        or resolved_temporal_candidate_min_snr > float(candidate_consensus_min_snr)
    ):
        raise ValueError("temporal_candidate_min_snr must be positive and no greater than candidate_consensus_min_snr")
    if (
        resolved_temporal_reference_min_snr <= 0
        or resolved_temporal_reference_min_snr > float(candidate_consensus_min_snr)
    ):
        raise ValueError("temporal_reference_min_snr must be positive and no greater than candidate_consensus_min_snr")
    if not 0.0 < float(temporal_min_psf_correlation) <= 1.0:
        raise ValueError("temporal_min_psf_correlation must be between 0 and 1")
    stack_reference_mode = str(stack_reference_mode).strip().lower()
    if stack_reference_mode not in {"median", "coadd"}:
        raise ValueError("stack_reference_mode must be 'median' or 'coadd'")
    if stack_threshold_sigma <= 0 or stack_min_flux_snr <= 0 or stack_frame_min_flux_snr <= 0:
        raise ValueError("stack_faint thresholds must be positive")
    if stack_min_presence is not None and stack_min_presence < 1:
        raise ValueError("stack_min_presence must be positive when provided")
    if relative_photometry_max_sources < 1:
        raise ValueError("relative_photometry_max_sources must be positive")
    if relative_photometry_spatial_order not in {0, 1, 2}:
        raise ValueError("relative_photometry_spatial_order must be 0, 1, or 2")
    if not 0.0 <= float(relative_photometry_validation_fraction) < 1.0:
        raise ValueError("relative_photometry_validation_fraction must be in [0, 1)")
    resolved_psf_fwhm = float(detector_kwargs.get("psf_fwhm", 3.0))
    resolved_motion_line_link_radius_px = (
        max(48.0, 32.0 * resolved_psf_fwhm)
        if motion_line_link_radius_px is None
        else float(motion_line_link_radius_px)
    )
    if (
        not np.isfinite(resolved_motion_line_link_radius_px)
        or resolved_motion_line_link_radius_px <= 0.0
    ):
        raise ValueError("motion_line_link_radius_px must be positive")
    if detector_kwargs.get("max_sources") is None and sequence_max_sources is not None:
        detector_kwargs["max_sources"] = int(sequence_max_sources)
    # 序列只需要稳定的空间噪声场；更大的网格减少背景统计开销，仍会在
    # 每个像素位置插值得到 background_map/noise_map，不改变输入图像。
    detector_kwargs.setdefault("background_box_size", 256)
    # 15 帧主要需要稳定的配准参考，不需要为数万候选逐个重复做环形
    # sigma-clipping；允许调用方显式传 True 进行高精度单帧式测量。
    detector_kwargs.setdefault("refine_local_background", False)
    # 当前数据为 16 位 FITS；float32 对每个输入 ADU 是精确的，只减少
    # 4096² 像素级中间阵列的内存带宽。单图入口仍保持 float64 默认口径。
    detector_kwargs.setdefault("use_float32", True)
    detector_kwargs.setdefault(
        "background_sample_limit",
        DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT if detector_kwargs.get("use_float32", True) else 100_000,
    )
    detector_kwargs.setdefault("fast_sequence", bool(detector_kwargs.get("use_float32", True)))
    if progress is not None:
        progress("prepare", 0, total)

    # 连续观测帧共享同一片天空背景。快速序列路径先用首帧生成二维
    # background/RMS 模型，后续帧仍各自计算全局统计和有效像素掩膜，
    # 但不再重复 256 个局部网格的 sigma-clipping。测试中的虚拟路径
    # 没有真实文件，因此自然回退到逐帧模型。
    shared_background_model: tuple[np.ndarray, np.ndarray] | None = None
    if (
        total >= 2
        and bool(detector_kwargs.get("fast_sequence", False))
        and bool(detector_kwargs.get("use_float32", True))
        and frame_paths[0].is_file()
    ):
        pilot = read_fits(frame_paths[0])
        shared_background_model = build_background_model(
            pilot.data,
            mask=auxiliary_mask(pilot.data.shape),
            background_box_size=int(detector_kwargs["background_box_size"]),
            background_sample_limit=int(detector_kwargs["background_sample_limit"]),
            mask_zero_pixels=(
                bool(detector_kwargs["mask_zero_pixels"])
                if detector_kwargs.get("mask_zero_pixels") is not None
                else None
            ),
            refine_local_background=bool(detector_kwargs.get("refine_local_background", False)),
            use_float32=True,
        )
        del pilot
        if progress is not None:
            progress("background", total, total)

    def analyze_one(item: tuple[int, Path]) -> tuple[int, FrameAnalysis]:
        frame_index, path = item

        def frame_progress(value: float, label: str) -> None:
            if detail_progress is not None:
                detail_progress(frame_index, total, value, label)

        frame_kwargs = dict(detector_kwargs)
        if shared_background_model is not None:
            frame_kwargs["background_model"] = shared_background_model
        analysis = analyze_frame(path, progress=frame_progress, **frame_kwargs)
        return frame_index, analysis

    analyses: list[FrameAnalysis | None] = [None] * total
    work_items = tuple(enumerate(frame_paths, start=1))
    worker_count = min(int(sequence_workers), total)
    if worker_count == 1:
        completed_items = map(analyze_one, work_items)
        for completed, (frame_index, analysis) in enumerate(completed_items, start=1):
            analyses[frame_index - 1] = analysis
            if progress is not None:
                progress("frame", completed, total)
    else:
        # NumPy/SciPy 的卷积、分位数和极大值滤波会释放 GIL；按帧并行能
        # NumPy/SciPy 的卷积、分位数和极大值滤波会释放 GIL；默认 4 个
        # worker 是当前 4096² 数据的实测吞吐点。结果按 frame_index 写回，
        # 输出顺序稳定；内存紧张时调用方可显式降到 1/2/3。
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="rst19-frame") as executor:
            completed_items = executor.map(analyze_one, work_items)
            for completed, (frame_index, analysis) in enumerate(completed_items, start=1):
                analyses[frame_index - 1] = analysis
                if progress is not None:
                    progress("frame", completed, total)
    resolved_analyses = [analysis for analysis in analyses if analysis is not None]
    if len(resolved_analyses) != total:
        raise RuntimeError("sequence frame analysis did not produce all results")
    # 关联仍使用整数 frame_index，物理运动拟合则使用 FITS 的观测时间。
    # 开运一号这组数据的帧间隔约 1.5 s 且存在毫秒级抖动，不能把帧号
    # 直接当成等间隔秒数。
    frame_times = _observation_times_seconds(resolved_analyses)
    # 对疑似填充值/坏像素做序列级固定值审计。它只保留统计与坐标证据，
    # 不改变单帧检测的原图、掩膜或质量计数，避免把“固定异常码”误当成
    # 已经确认的物理坏像素。
    fixed_sentinel_audit = audit_fixed_sentinel(
        tuple(analysis.frame.data for analysis in resolved_analyses),
        sentinel_value=-1,
    )
    fixed_sentinel_impact_audit = audit_fixed_sentinel_impacts(
        resolved_analyses,
        fixed_sentinel_audit,
    )
    if progress is not None:
        progress("sentinel-audit", total, total)
    calculation_dtype = (
        "float32"
        if all(int(analysis.detection.parameters.get("use_float32", 0)) for analysis in resolved_analyses)
        else "float64"
    )
    if progress is not None:
        progress("registration", 0, total)
    result = track_detections(
        [analysis.detection.quality_sources for analysis in resolved_analyses],
        frame_names=[str(path) for path in frame_paths],
        frame_times=frame_times,
        link_radius_px=link_radius_px,
        min_presence=min_presence,
        motion_min_displacement_px=motion_min_displacement_px,
        registration_radius_px=registration_radius_px,
        max_motion_fit_rms_px=max_motion_fit_rms_px,
        persistent_min_presence=persistent_min_presence,
    )
    if progress is not None:
        progress("registration", total, total)
    effective_motion_gate = float(
        getattr(result, "effective_motion_min_displacement_px", None)
        or motion_min_displacement_px
    )
    relative_photometry_result: RelativePhotometryResult | None = None
    if relative_photometry:
        if progress is not None:
            progress("relative-photometry", 0, total)
        relative_photometry_result = _fit_sequence_relative_photometry(
            resolved_analyses,
            result.tracks,
            min_presence=result.min_presence,
            max_sources=int(relative_photometry_max_sources),
            spatial_order=int(relative_photometry_spatial_order),
            validation_fraction=float(relative_photometry_validation_fraction),
        )
        if progress is not None:
            progress("relative-photometry", total, total)
    fast_point_quality_tracks: tuple[SourceTrack, ...] = ()
    if fast_point_motion and total >= 2:
        if progress is not None:
            progress("fast-point-motion", 0, total)
        fast_point_quality_tracks = track_fast_point_movers(
            [analysis.detection.quality_sources for analysis in resolved_analyses],
            result.cumulative_shifts,
            frame_times=frame_times,
            link_radius_px=link_radius_px,
            min_presence=result.min_presence,
            motion_min_displacement_px=effective_motion_gate,
            min_flux_snr=fast_point_min_snr,
            max_step_px=fast_point_max_step_px,
            gate_px=fast_point_gate_px,
            max_fit_rms_px=fast_point_max_fit_rms_px,
            existing_tracks=result.tracks,
        )
    # 线状层本来就需要注册时间中值参考；提前构建一次并复用到候选补检，
    # 避免为了补回单帧 Gaussian 漏掉的稳定源而再复制/计算一遍 4096² 参考图。
    temporal_reference = _registered_median_reference(
        resolved_analyses,
        result.cumulative_shifts,
    )
    temporal_candidate_frames, temporal_candidate_count = _temporal_reference_candidate_frames(
        resolved_analyses,
        result.cumulative_shifts,
        temporal_reference,
        psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
        min_candidate_snr=resolved_temporal_reference_min_snr,
        temporal_multiscale=bool(temporal_multiscale),
    )
    temporal_coadd_candidate_count = 0
    if temporal_proposal_mode in {"coadd", "both"}:
        if progress is not None:
            progress("temporal-coadd", 0, total)
        temporal_coadd_reference = _registered_coadd_reference(
            resolved_analyses,
            result.cumulative_shifts,
        )
        coadd_candidate_frames, temporal_coadd_candidate_count = _temporal_reference_candidate_frames(
            resolved_analyses,
            result.cumulative_shifts,
            temporal_coadd_reference,
            psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
            min_candidate_snr=resolved_temporal_reference_min_snr,
            temporal_multiscale=bool(temporal_multiscale),
        )
        del temporal_coadd_reference
        if temporal_proposal_mode == "coadd":
            temporal_candidate_frames = coadd_candidate_frames
        else:
            temporal_candidate_frames = tuple(
                np.vstack((median_rows, coadd_rows)).astype(np.float32, copy=False)
                if median_rows.size and coadd_rows.size
                else coadd_rows
                if coadd_rows.size
                else median_rows
                for median_rows, coadd_rows in zip(
                    temporal_candidate_frames,
                    coadd_candidate_frames,
                    strict=True,
                )
            )
        if progress is not None:
            progress("temporal-coadd", total, total)
    candidate_peak_frames: list[np.ndarray] = []
    fast_candidate_frames: list[np.ndarray] = []
    for analysis, temporal_rows in zip(resolved_analyses, temporal_candidate_frames, strict=True):
        raw = np.asarray(analysis.detection.candidate_peaks, dtype=np.float32)
        if raw.size == 0:
            base_rows = np.empty((0, 3), dtype=np.float32)
        elif raw.ndim == 2 and raw.shape[1] >= 3:
            base_rows = raw[:, :3]
        else:
            # 旧/非快速结果没有临时候选峰证据；不要把异常数组猜成坐标。
            base_rows = np.empty((0, 3), dtype=np.float32)
        # 高速点源宽筛只使用当前帧的匹配滤波峰。时间中值/叠加补提案
        # 用于稳定暗星恢复，不能混入高速点源的速度种子，否则一条静态
        # 补提案可能与瞬态噪点串成伪高速轨迹。
        fast_candidate_frames.append(base_rows)
        if base_rows.size:
            base_rows = np.column_stack(
                (
                    base_rows[:, :3],
                    np.zeros(base_rows.shape[0], dtype=np.float32),
                    np.full(base_rows.shape[0], np.nan, dtype=np.float32),
                )
            ).astype(np.float32, copy=False)
        if temporal_rows.size == 0:
            candidate_peak_frames.append(base_rows)
        elif base_rows.size == 0:
            candidate_peak_frames.append(temporal_rows)
        else:
            candidate_peak_frames.append(np.vstack((base_rows, temporal_rows)).astype(np.float32, copy=False))
    if fast_point_motion and total >= 3:
        fast_point_candidate_tracks = _track_fast_candidate_movers(
            fast_candidate_frames,
            resolved_analyses,
            result.cumulative_shifts,
            frame_times=frame_times,
            link_radius_px=link_radius_px,
            min_presence=result.min_presence,
            motion_min_displacement_px=effective_motion_gate,
            min_filter_snr=fast_point_min_snr,
            max_step_px=fast_point_max_step_px,
            gate_px=fast_point_gate_px,
            max_fit_rms_px=fast_point_max_fit_rms_px,
            candidate_limit=int(fast_point_candidate_limit),
            existing_tracks=result.tracks + fast_point_quality_tracks,
            aperture_radius=int(detector_kwargs.get("aperture_radius", 4)),
            min_psf_support_pixels=int(detector_kwargs.get("min_psf_support_pixels", 3)),
            min_fwhm=float(detector_kwargs.get("min_fwhm", 0.8)),
            max_fwhm=float(detector_kwargs.get("max_fwhm", 12.0)),
            max_ellipticity=float(detector_kwargs.get("max_ellipticity", 0.65)),
            line_artifact_frames=[
                analysis.detection.line_artifact_coordinates
                for analysis in resolved_analyses
            ],
            progress=(
                (lambda index, count, label: progress("fast-point-motion", index, count))
                if progress is not None
                else None
            ),
        )
    else:
        fast_point_candidate_tracks = ()
    fast_point_tracks = fast_point_quality_tracks + fast_point_candidate_tracks
    if progress is not None and fast_point_motion and total >= 2:
        progress("fast-point-motion", total, total)
    candidate_audit: dict[str, int] = {}
    candidate_consensus = _candidate_consensus_tracks(
        candidate_peak_frames,
        result.cumulative_shifts,
        result.tracks,
        frame_analyses=resolved_analyses,
        frame_times=frame_times,
        link_radius_px=link_radius_px,
        min_presence=result.min_presence,
        persistent_min_presence=int(
            getattr(
                result,
                "persistent_min_presence",
                persistent_min_presence
                if persistent_min_presence is not None
                else max(3, int(np.ceil(total * 0.5))),
            )
        ),
        motion_min_displacement_px=effective_motion_gate,
        max_motion_fit_rms_px=max_motion_fit_rms_px,
        min_candidate_snr=float(candidate_consensus_min_snr),
        temporal_candidate_min_snr=resolved_temporal_candidate_min_snr,
        temporal_reference_min_snr=resolved_temporal_reference_min_snr,
        temporal_min_psf_correlation=float(temporal_min_psf_correlation),
        line_artifact_frames=[
            analysis.detection.line_artifact_coordinates
            for analysis in resolved_analyses
        ],
        audit_sink=candidate_audit,
    )
    quality_tracks = tuple(result.tracks)
    fast_point_tracks_with_ids = tuple(
        SourceTrack(
            track_id=len(quality_tracks) + index,
            classification=track.classification,
            points=track.points,
            displacement_px=track.displacement_px,
            speed_px_per_frame=track.speed_px_per_frame,
            fit_rms_px=track.fit_rms_px,
            evidence_level=track.evidence_level,
        )
        for index, track in enumerate(fast_point_tracks)
    )
    all_tracks = quality_tracks + fast_point_tracks_with_ids
    candidate_tracks = tuple(
        SourceTrack(
            track_id=len(all_tracks) + index,
            classification=track.classification,
            points=track.points,
            displacement_px=track.displacement_px,
            speed_px_per_frame=track.speed_px_per_frame,
            fit_rms_px=track.fit_rms_px,
            evidence_level=track.evidence_level,
        )
        for index, track in enumerate(candidate_consensus)
    )
    all_tracks = all_tracks + candidate_tracks
    resolved_stack_min_presence = (
        int(getattr(result, "persistent_min_presence", max(3, int(np.ceil(total * 0.5)))))
        if stack_min_presence is None
        else int(stack_min_presence)
    )
    stack_faint_candidate_count = 0
    stack_faint_tracks: tuple[SourceTrack, ...] = ()
    if stack_faint_recovery and total >= 2:
        if progress is not None:
            progress("stack-faint", 0, total)
        stack_reference = temporal_reference
        if stack_reference_mode == "coadd":
            stack_reference = _registered_coadd_reference(
                resolved_analyses,
                result.cumulative_shifts,
            )
        stack_audit: dict[str, int] = {}
        stack_faint_tracks = _stack_faint_tracks(
            resolved_analyses,
            result.cumulative_shifts,
            stack_reference,
            threshold_sigma=float(detector_kwargs.get("threshold_sigma", 4.0)),
            min_distance=int(detector_kwargs.get("min_distance", 3)),
            aperture_radius=int(detector_kwargs.get("aperture_radius", 4)),
            psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
            min_flux_snr=float(stack_min_flux_snr),
            min_psf_support_pixels=int(detector_kwargs.get("min_psf_support_pixels", 3)),
            proposal_mode=str(detector_kwargs.get("proposal_mode", "gaussian")),
            stack_frame_min_flux_snr=float(stack_frame_min_flux_snr),
            min_presence=resolved_stack_min_presence,
            link_radius_px=link_radius_px,
            existing_tracks=all_tracks,
            audit_sink=stack_audit,
            workers=sequence_workers,
            max_reference_sources=DEFAULT_SEQUENCE_STACK_WORKING_LIMIT,
            progress=(
                (lambda index, count, label: progress("stack-faint", index, count))
                if progress is not None
                else None
            ),
        )
        stack_faint_candidate_count = int(stack_audit.get("stack_reference_quality_sources", 0))
        if stack_reference is not temporal_reference:
            del stack_reference
        if stack_faint_tracks:
            stack_faint_with_ids = tuple(
                SourceTrack(
                    track_id=len(all_tracks) + index,
                    classification=track.classification,
                    points=track.points,
                    displacement_px=track.displacement_px,
                    speed_px_per_frame=track.speed_px_per_frame,
                    fit_rms_px=track.fit_rms_px,
                    evidence_level=track.evidence_level,
                )
                for index, track in enumerate(stack_faint_tracks)
            )
            all_tracks = all_tracks + stack_faint_with_ids
        if progress is not None:
            progress("stack-faint", total, total)
    if progress is not None:
        progress("consensus", total, total)
    summaries = tuple(
        FrameSequenceSummary(
            frame_index=index,
            path=str(path),
            candidate_count=analysis.detection.candidate_count,
            returned_count=analysis.detection.returned_count,
            quality_count=analysis.detection.star_count,
            timestamp=str(analysis.frame.header.get("DATE-OBS")) if analysis.frame.header.get("DATE-OBS") is not None else None,
            exposure_ms=exposure_milliseconds(analysis.frame.header),
            auxiliary=tuple(analysis.frame.auxiliary.as_dict().items()) if analysis.frame.auxiliary is not None else (),
            width_px=analysis.frame.width,
            height_px=analysis.frame.height,
        )
        for index, (path, analysis) in enumerate(zip(frame_paths, resolved_analyses, strict=True))
    )
    # 点轨迹关联已经消费了完整的 quality_sources。线状检测只需要原图、
    # 背景/噪声和屏蔽参数，不需要再次持有每帧数万条 Detection 对象；构造
    # 轻量视图后释放完整单帧分析，避免 GUI 后台在时间中值阶段继续占用峰值内存。
    motion_analyses = tuple(
        FrameAnalysis(
            frame=analysis.frame,
            detection=DetectionResult(
                image_shape=analysis.detection.image_shape,
                background=analysis.detection.background,
                noise=analysis.detection.noise,
                threshold=analysis.detection.threshold,
                candidate_count=analysis.detection.candidate_count,
                sources=(),
                parameters=dict(analysis.detection.parameters),
                quality_count=0,
                line_artifact_coordinates=analysis.detection.line_artifact_coordinates,
            ),
            matching=None,
            faintest=None,
        )
        for analysis in resolved_analyses
    )
    model_modes = {
        str(analysis.detection.parameters.get("background_model_mode", "per_frame_local"))
        for analysis in resolved_analyses
    }
    del analyses, resolved_analyses
    motion_audits: list[MotionFrameAudit] = []
    def motion_detail_progress(value: float, label: str) -> None:
        if progress is not None:
            progress("motion-detail", int(round(float(value))), 100)

    motion_features, motion_threshold = detect_motion_features(
        motion_analyses,
        result.cumulative_shifts,
        psf_fwhm=float(detector_kwargs.get("psf_fwhm", 3.0)),
        temporal_reference=temporal_reference,
        progress=motion_detail_progress,
        audit_sink=motion_audits,
        # 线形补充层不依赖单帧的单独工作集；使用保守的并行度，降低
        # 15 帧重做原图连通域时的等待时间，同时避免和前面的源级检测
        # 争抢全部内存带宽。单帧入口不会经过 analyze_sequence。
        raw_fallback_workers=min(2, max(1, int(sequence_workers))),
        multiscale_line_proposals=bool(motion_multiscale_line_proposals),
        line_link_radius_px=resolved_motion_line_link_radius_px,
    )
    # 点状高速轨迹和曝光线是两个互补检测通道。它们可能同时描述同一
    # 个物体；保留两份原始证据，但在汇总层显式建立关联，避免界面把
    # 同一目标误报成两个运动物体。
    motion_evidence_links = _link_motion_evidence_tracks(
        all_tracks,
        motion_features,
    )
    del motion_analyses
    del temporal_reference
    if progress is not None:
        progress("motion", total, total)
        # ``complete`` 属于 GUI 的提交态，不属于算法线程的中间进度。
        # 此时 SequenceResult 还没有返回给主线程；如果这里提前发完成，
        # GUI 会先显示 100%，随后才收到结果事件，造成“100%但图/右卡为空”。
    if model_modes == {"shared_sequence_pilot"}:
        background_model_mode = "shared_sequence_pilot"
    elif "shared_sequence_pilot" in model_modes:
        background_model_mode = "shared_sequence_pilot_with_fallback"
    else:
        background_model_mode = "per_frame_local"
    detector_parameter_names = (
        "threshold_sigma",
        "min_distance",
        "aperture_radius",
        "max_sources",
        "psf_fwhm",
        "min_flux_snr",
        "min_psf_support_pixels",
        "min_fwhm",
        "max_fwhm",
        "max_ellipticity",
        "min_sharpness",
        "max_sharpness",
        "min_footprint_pixels",
        "proposal_mode",
        "enable_local_deblend",
        "reject_linear_artifacts",
        "mask_zero_pixels",
        "background_box_size",
        "refine_local_background",
        "use_float32",
        "background_sample_limit",
        "fast_sequence",
    )
    detector_parameters: list[tuple[str, object]] = []
    for name in detector_parameter_names:
        value = detector_kwargs.get(name)
        if isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            detector_parameters.append((name, value))
    return SequenceResult(
        frames=summaries,
        cumulative_shifts=result.cumulative_shifts,
        tracks=all_tracks,
        link_radius_px=result.link_radius_px,
        min_presence=result.min_presence,
        motion_min_displacement_px=result.motion_min_displacement_px,
        max_motion_fit_rms_px=result.max_motion_fit_rms_px,
        motion_line_link_radius_px=resolved_motion_line_link_radius_px,
        persistent_min_presence=int(
            getattr(
                result,
                "persistent_min_presence",
                persistent_min_presence
                if persistent_min_presence is not None
                else max(3, int(np.ceil(total * 0.5))),
            )
        ),
        motion_features=motion_features,
        motion_evidence_links=motion_evidence_links,
        motion_residual_threshold_adu=motion_threshold,
        motion_reference_mode=(
            "per_frame_background"
            if len(frame_paths) < 2
            else "temporal_median_small_shift"
            if max(float(np.max(np.abs(result.cumulative_shifts))), 0.0) <= 0.75
            else "registered_median"
        ),
        motion_frame_audits=tuple(motion_audits),
        source_working_limit=(
            int(detector_kwargs["max_sources"])
            if detector_kwargs.get("max_sources") is not None
            else None
        ),
        calculation_dtype=calculation_dtype,
        background_sample_limit=(
            int(detector_kwargs["background_sample_limit"])
            if detector_kwargs.get("background_sample_limit") is not None
            else None
        ),
        fast_sequence=bool(detector_kwargs.get("fast_sequence", False)),
        background_model_mode=background_model_mode,
        temporal_reference_candidate_count=int(temporal_candidate_count),
        temporal_coadd_candidate_count=int(temporal_coadd_candidate_count),
        temporal_proposal_mode=temporal_proposal_mode,
        temporal_candidate_min_snr=float(resolved_temporal_candidate_min_snr),
        temporal_reference_min_snr=float(resolved_temporal_reference_min_snr),
        temporal_multiscale=bool(temporal_multiscale),
        temporal_min_psf_correlation=float(temporal_min_psf_correlation),
        candidate_consensus_audit=tuple(sorted(candidate_audit.items())),
        stack_faint_candidate_count=stack_faint_candidate_count,
        stack_reference_mode=stack_reference_mode,
        stack_threshold_sigma=float(stack_threshold_sigma),
        stack_min_flux_snr=float(stack_min_flux_snr),
        stack_frame_min_flux_snr=float(stack_frame_min_flux_snr),
        stack_min_presence=resolved_stack_min_presence,
        fixed_sentinel_audit=fixed_sentinel_audit,
        fixed_sentinel_impact_audit=fixed_sentinel_impact_audit,
        relative_photometry=relative_photometry_result,
        track_association_method=(
            "local_component_hungarian+exposure_velocity_gate+line_geometry_soft_cost+motion_axis_gate+contiguous_run_gate"
        ),
        detector_parameters=tuple(detector_parameters),
        registration_model=str(getattr(result, "registration_model", "translation")),
        registration_affine_frame_count=int(getattr(result, "registration_affine_frame_count", 0)),
        registration_residual_median_px=(
            float(getattr(result, "registration_residual_median_px"))
            if getattr(result, "registration_residual_median_px", None) is not None
            else None
        ),
        effective_motion_min_displacement_px=(
            float(getattr(result, "effective_motion_min_displacement_px"))
            if getattr(result, "effective_motion_min_displacement_px", None) is not None
            else None
        ),
    )
