"""人工阈值反馈的本地记录。

反馈不是在线训练，也不会修改检测器的物理量。它把用户在真实帧上试过的
阈值、结果规模和拒绝原因保存为 JSONL，供后续实验或人工调参读取。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FEEDBACK_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManualThresholdFeedback:
    """一次已经实际运行过的人工阈值试验。"""

    timestamp_utc: str
    frame_path: str
    threshold_sigma: float
    min_flux_snr: float
    min_distance: int
    psf_fwhm: float
    candidate_count: int
    quality_count: int
    returned_count: int
    flag_counts: dict[str, int] = field(default_factory=dict)
    judgement: str = "当前平衡"
    note: str = ""
    source: str = "tkinter_manual"

    def as_dict(self) -> dict[str, Any]:
        """返回稳定、可序列化的记录格式。"""

        payload = asdict(self)
        payload["schema_version"] = FEEDBACK_SCHEMA_VERSION
        return payload


def append_manual_feedback(path: Path, feedback: ManualThresholdFeedback) -> Path:
    """以追加方式写入一条 JSONL 反馈，并返回实际路径。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(feedback.as_dict(), handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return target


def load_manual_feedback(path: Path) -> tuple[ManualThresholdFeedback, ...]:
    """读取已有反馈；空文件返回空元组。

    记录一旦损坏就显式报错，避免后续调参悄悄漏掉一部分样本。
    """

    target = Path(path)
    if not target.exists():
        return ()
    records: list[ManualThresholdFeedback] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(
                    ManualThresholdFeedback(
                        timestamp_utc=str(payload["timestamp_utc"]),
                        frame_path=str(payload["frame_path"]),
                        threshold_sigma=float(payload["threshold_sigma"]),
                        min_flux_snr=float(payload["min_flux_snr"]),
                        min_distance=int(payload["min_distance"]),
                        psf_fwhm=float(payload["psf_fwhm"]),
                        candidate_count=int(payload["candidate_count"]),
                        quality_count=int(payload["quality_count"]),
                        returned_count=int(payload["returned_count"]),
                        flag_counts={str(key): int(value) for key, value in dict(payload.get("flag_counts", {})).items()},
                        judgement=str(payload.get("judgement", "当前平衡")),
                        note=str(payload.get("note", "")),
                        source=str(payload.get("source", "tkinter_manual")),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"人工反馈第 {line_number} 行格式无效：{exc}") from exc
    return tuple(records)

