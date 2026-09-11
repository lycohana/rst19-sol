"""Directory-independent relative photometric calibration for frame sequences.

This module deliberately implements a *relative* magnitude scale only.  It does
not know a filter, a standard catalogue, an exposure calibration, or a distance,
so none of its outputs are absolute magnitudes.

For an accepted observation ``(i, j)`` the fitted model is

``m_ij = mu_j + z_i + spatial_terms(x_ij, y_ij)``

where ``mu_j`` is a source-relative magnitude and ``z_i`` is a frame zero-point
offset.  The additive gauge is fixed after fitting by shifting all ``z_i`` so
that ``median(z) == 0`` and applying the opposite shift to ``mu_j``.

The public entry point accepts dataclass rows, mappings, attribute-based rows,
or nine-element sequences.  No filesystem, GUI, catalogue, or directory name is
used by the fit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Hashable

import numpy as np


@dataclass(frozen=True, slots=True)
class RelativePhotometryObservation:
    """One row in a frame/source relative-photometry table.

    ``is_variable_optional`` may be ``None`` when no variability classifier is
    available.  A value of ``True`` excludes the row from the reference fit.
    """

    frame_id: Hashable
    source_id: Hashable
    instrumental_magnitude: float
    magnitude_error: float
    x: float
    y: float
    quality_passed: bool
    is_moving: bool
    is_variable_optional: bool | None = None


# Short aliases make the row type convenient without forcing callers to use a
# project-specific name.
Observation = RelativePhotometryObservation
PhotometryObservation = RelativePhotometryObservation


@dataclass(frozen=True, slots=True)
class RelativePhotometryResult:
    """Fit result and evidence for a directory-independent relative scale.

    ``frame_zero_points`` and ``relative_magnitudes`` are plain mappings so the
    result can be consumed directly by tests, notebooks, or a later UI layer.
    They are not calibrated apparent or absolute magnitudes.

    Residual RMS values are unweighted magnitude residuals.  MAD values are the
    unscaled median absolute deviation around the residual median.  Uncertainty
    values are covariance estimates from the weighted least-squares fit; they
    are ``None`` when there are no residual degrees of freedom.
    """

    status: str
    flags: tuple[str, ...] = ()
    frame_zero_points: dict[Hashable, float] = field(default_factory=dict)
    relative_magnitudes: dict[Hashable, float] = field(default_factory=dict)
    frame_zero_point_uncertainties: dict[Hashable, float | None] = field(default_factory=dict)
    relative_magnitude_uncertainties: dict[Hashable, float | None] = field(default_factory=dict)
    reference_sample_count: int = 0
    reference_sample_count_by_frame: dict[Hashable, int] = field(default_factory=dict)
    reference_sample_count_by_source: dict[Hashable, int] = field(default_factory=dict)
    training_residual_rms: float | None = None
    training_residual_mad: float | None = None
    validation_residual_rms: float | None = None
    validation_residual_mad: float | None = None
    validation_sample_count: int = 0
    validation_source_ids: tuple[Hashable, ...] = ()
    rejected_observation_indices: tuple[int, ...] = ()
    excluded_observation_count: int = 0
    spatial_coefficients: tuple[float, ...] = ()
    spatial_order: int = 0

    @property
    def zero_points(self) -> dict[Hashable, float]:
        """Alias for callers that use the shorter frame-offset name."""

        return self.frame_zero_points

    @property
    def source_relative_magnitudes(self) -> dict[Hashable, float]:
        """Alias emphasizing that source values are relative, not absolute."""

        return self.relative_magnitudes

    @property
    def train_rms(self) -> float | None:
        return self.training_residual_rms

    @property
    def train_mad(self) -> float | None:
        return self.training_residual_mad

    @property
    def validation_rms(self) -> float | None:
        return self.validation_residual_rms

    @property
    def validation_mad(self) -> float | None:
        return self.validation_residual_mad

    @staticmethod
    def _json_key(value: Hashable) -> str:
        """Return a deterministic string key for JSON-facing consumers."""

        return str(value)

    @classmethod
    def _json_mapping(cls, values: Mapping[Hashable, object]) -> dict[str, object]:
        """Convert possibly non-string IDs to a stable JSON mapping."""

        return {
            cls._json_key(key): values[key]
            for key in sorted(values, key=lambda item: str(item))
        }

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the relative fit.

        Source/frame identifiers are intentionally stringified at this
        reporting boundary.  The fit itself keeps the original hashable IDs,
        so callers that need typed keys can continue using the dataclass
        attributes directly.  The payload explicitly says ``relative`` in
        its field names and never presents these values as apparent or
        absolute magnitudes.
        """

        return {
            "status": self.status,
            "flags": list(self.flags),
            "frame_zero_points": self._json_mapping(self.frame_zero_points),
            "relative_magnitudes": self._json_mapping(self.relative_magnitudes),
            "frame_zero_point_uncertainties": self._json_mapping(self.frame_zero_point_uncertainties),
            "relative_magnitude_uncertainties": self._json_mapping(self.relative_magnitude_uncertainties),
            "reference_sample_count": self.reference_sample_count,
            "reference_sample_count_by_frame": self._json_mapping(self.reference_sample_count_by_frame),
            "reference_sample_count_by_source": self._json_mapping(self.reference_sample_count_by_source),
            "training_residual_rms": self.training_residual_rms,
            "training_residual_mad": self.training_residual_mad,
            "validation_residual_rms": self.validation_residual_rms,
            "validation_residual_mad": self.validation_residual_mad,
            "validation_sample_count": self.validation_sample_count,
            "validation_source_ids": [self._json_key(value) for value in self.validation_source_ids],
            "rejected_observation_indices": list(self.rejected_observation_indices),
            "excluded_observation_count": self.excluded_observation_count,
            "spatial_coefficients": list(self.spatial_coefficients),
            "spatial_order": self.spatial_order,
        }


@dataclass(frozen=True, slots=True)
class _PreparedObservation:
    input_index: int
    row: RelativePhotometryObservation


@dataclass(frozen=True, slots=True)
class _WlsSolution:
    beta: np.ndarray
    residuals: np.ndarray
    covariance: np.ndarray | None
    rank: int
    parameter_count: int
    condition_number: float
    degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class _FittedModel:
    frame_zero_points: dict[Hashable, float]
    relative_magnitudes: dict[Hashable, float]
    frame_uncertainties: dict[Hashable, float | None]
    source_uncertainties: dict[Hashable, float | None]
    spatial_coefficients: tuple[float, ...]
    frame_ids: tuple[Hashable, ...]
    source_ids: tuple[Hashable, ...]
    reference_frame_id: Hashable
    spatial_center: tuple[float, float]
    spatial_scale: tuple[float, float]
    inlier_local_indices: np.ndarray
    residuals: np.ndarray
    rank: int
    parameter_count: int
    condition_number: float
    degrees_of_freedom: int
    rejected_input_indices: tuple[int, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FitAttempt:
    model: _FittedModel | None
    status: str
    flags: tuple[str, ...]


_MISSING = object()


def _dedupe_flags(flags: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(flag) for flag in flags))


def _field_value(row: Any, name: str, position: int, default: Any = _MISSING) -> Any:
    """Read a field from a mapping, object, or positional row."""

    if isinstance(row, Mapping):
        if name in row:
            return row[name]
        if default is not _MISSING:
            return default
        raise KeyError(name)
    if hasattr(row, name):
        return getattr(row, name)
    if not isinstance(row, (str, bytes, bytearray)) and hasattr(row, "__getitem__"):
        try:
            return row[position]
        except (IndexError, KeyError, TypeError):
            pass
    if default is not _MISSING:
        return default
    raise KeyError(name)


def _coerce_observation(
    row: Any,
    input_index: int,
    *,
    require_coordinates: bool,
) -> tuple[_PreparedObservation | None, str | None]:
    """Convert an input row and return a reason when it is excluded."""

    try:
        frame_id = _field_value(row, "frame_id", 0)
        source_id = _field_value(row, "source_id", 1)
        quality_passed = _field_value(row, "quality_passed", 6)
        is_moving = _field_value(row, "is_moving", 7)
        is_variable = _field_value(row, "is_variable_optional", 8, None)
        if isinstance(row, Mapping) and "is_variable_optional" not in row:
            # A few upstream tables use the shorter spelling.  It is accepted as
            # an input convenience but never changes the output vocabulary.
            is_variable = row.get("is_variable", row.get("variable", None))
        elif is_variable is None and hasattr(row, "is_variable"):
            is_variable = getattr(row, "is_variable")
        magnitude = _field_value(row, "instrumental_magnitude", 2)
        magnitude_error = _field_value(row, "magnitude_error", 3)
        x = _field_value(row, "x", 4)
        y = _field_value(row, "y", 5)
    except (KeyError, IndexError, TypeError, AttributeError):
        return None, "INVALID_OBSERVATION"

    if not bool(quality_passed):
        return None, "QUALITY_REJECTED"
    if bool(is_moving):
        return None, "MOVING_REJECTED"
    if is_variable is not None and bool(is_variable):
        return None, "VARIABLE_REJECTED"

    try:
        frame_hash = hash(frame_id)
        source_hash = hash(source_id)
        del frame_hash, source_hash
        magnitude_value = float(magnitude)
        error_value = float(magnitude_error)
        x_value = float(x)
        y_value = float(y)
    except (TypeError, ValueError, OverflowError):
        return None, "INVALID_OBSERVATION"

    if not math.isfinite(magnitude_value) or not math.isfinite(error_value) or error_value <= 0:
        return None, "INVALID_MAGNITUDE_ERROR"
    if require_coordinates and (not math.isfinite(x_value) or not math.isfinite(y_value)):
        return None, "INVALID_COORDINATES"
    if not math.isfinite(x_value):
        x_value = 0.0
    if not math.isfinite(y_value):
        y_value = 0.0

    prepared = RelativePhotometryObservation(
        frame_id=frame_id,
        source_id=source_id,
        instrumental_magnitude=magnitude_value,
        magnitude_error=error_value,
        x=x_value,
        y=y_value,
        quality_passed=True,
        is_moving=False,
        is_variable_optional=None,
    )
    return _PreparedObservation(input_index=input_index, row=prepared), None


def _unique_in_order(values: Sequence[Hashable]) -> tuple[Hashable, ...]:
    result: list[Hashable] = []
    seen: set[Hashable] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _stable_id_order(values: Sequence[Hashable]) -> tuple[Hashable, ...]:
    return tuple(sorted(values, key=lambda value: (type(value).__name__, repr(value))))


def _spatial_feature_count(spatial_order: int) -> int:
    return 0 if spatial_order <= 0 else 2 if spatial_order == 1 else 5


def _spatial_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    spatial_order: int,
    center: tuple[float, float],
    scale: tuple[float, float],
) -> np.ndarray:
    if spatial_order <= 0:
        return np.empty((x.size, 0), dtype=float)
    x_normalized = (x - center[0]) / scale[0]
    y_normalized = (y - center[1]) / scale[1]
    linear = [x_normalized, y_normalized]
    if spatial_order == 1:
        return np.column_stack(linear)
    return np.column_stack(
        [
            x_normalized,
            y_normalized,
            x_normalized * x_normalized,
            x_normalized * y_normalized,
            y_normalized * y_normalized,
        ]
    )


def _spatial_normalization(records: Sequence[_PreparedObservation]) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.asarray([record.row.x for record in records], dtype=float)
    y = np.asarray([record.row.y for record in records], dtype=float)
    center = (float(np.mean(x)), float(np.mean(y)))
    raw_scale_x = float(np.std(x))
    raw_scale_y = float(np.std(y))
    scale = (
        raw_scale_x if raw_scale_x > np.finfo(float).eps else 1.0,
        raw_scale_y if raw_scale_y > np.finfo(float).eps else 1.0,
    )
    return center, scale


def _build_design(
    records: Sequence[_PreparedObservation],
    *,
    source_ids: tuple[Hashable, ...],
    frame_ids: tuple[Hashable, ...],
    spatial_order: int,
    spatial_center: tuple[float, float],
    spatial_scale: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[Hashable, int], dict[Hashable, int], int]:
    """Build a full-rank gauge-fixed design when the data support it.

    The first frame is the temporary reference frame and has no explicit column.
    This removes the one additive null direction.  The solution is subsequently
    shifted to the requested median-zero convention.
    """

    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    reference_frame_id = frame_ids[0]
    frame_index: dict[Hashable, int] = {}
    next_frame_column = len(source_ids)
    for frame_id in frame_ids:
        if frame_id != reference_frame_id:
            frame_index[frame_id] = next_frame_column
            next_frame_column += 1
    spatial_start = next_frame_column
    parameter_count = spatial_start + _spatial_feature_count(spatial_order)

    x_values = np.asarray([record.row.x for record in records], dtype=float)
    y_values = np.asarray([record.row.y for record in records], dtype=float)
    spatial = _spatial_features(
        x_values,
        y_values,
        spatial_order=spatial_order,
        center=spatial_center,
        scale=spatial_scale,
    )
    design = np.zeros((len(records), parameter_count), dtype=float)
    magnitudes = np.empty(len(records), dtype=float)
    errors = np.empty(len(records), dtype=float)
    for row_index, record in enumerate(records):
        design[row_index, source_index[record.row.source_id]] = 1.0
        frame_column = frame_index.get(record.row.frame_id)
        if frame_column is not None:
            design[row_index, frame_column] = 1.0
        if spatial.shape[1]:
            design[row_index, spatial_start:] = spatial[row_index]
        magnitudes[row_index] = record.row.instrumental_magnitude
        errors[row_index] = record.row.magnitude_error
    return design, magnitudes, errors, source_index, frame_index, spatial_start


def _weighted_least_squares(
    design: np.ndarray,
    magnitudes: np.ndarray,
    errors: np.ndarray,
) -> tuple[_WlsSolution | None, str | None]:
    rows, parameters = design.shape
    if rows < parameters:
        return None, "INSUFFICIENT_SAMPLES"
    if rows == 0 or parameters == 0:
        return None, "INSUFFICIENT_SAMPLES"

    weights = 1.0 / np.square(errors)
    maximum_weight = float(np.max(weights))
    if not math.isfinite(maximum_weight) or maximum_weight <= 0:
        return None, "INVALID_MAGNITUDE_ERROR"
    weights /= maximum_weight
    square_root_weights = np.sqrt(weights)
    weighted_design = design * square_root_weights[:, None]
    weighted_magnitudes = magnitudes * square_root_weights

    try:
        singular_values = np.linalg.svd(weighted_design, compute_uv=False)
        if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
            return None, "DEGENERATE_DESIGN"
        tolerance = max(weighted_design.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank < parameters:
            return None, "DEGENERATE_DESIGN"
        beta, _, _, _ = np.linalg.lstsq(weighted_design, weighted_magnitudes, rcond=None)
    except np.linalg.LinAlgError:
        return None, "DEGENERATE_DESIGN"

    residuals = magnitudes - design @ beta
    degrees_of_freedom = rows - rank
    covariance: np.ndarray | None = None
    if degrees_of_freedom > 0:
        standardized_residual_sum = float(np.sum(np.square(residuals / errors)))
        variance_scale = standardized_residual_sum / degrees_of_freedom
        try:
            # The weights were divided by ``maximum_weight`` above only to
            # keep the SVD numerically scaled.  Undo that normalization in the
            # covariance so the reported errors retain magnitude units.
            covariance = (
                np.linalg.pinv(weighted_design.T @ weighted_design)
                * variance_scale
                / maximum_weight
            )
        except np.linalg.LinAlgError:
            covariance = None
    smallest = float(singular_values[-1])
    condition_number = float(singular_values[0] / smallest) if smallest > 0 else math.inf
    return (
        _WlsSolution(
            beta=beta,
            residuals=residuals,
            covariance=covariance,
            rank=rank,
            parameter_count=parameters,
            condition_number=condition_number,
            degrees_of_freedom=degrees_of_freedom,
        ),
        None,
    )


def _residual_mad(residuals: np.ndarray) -> float | None:
    if residuals.size == 0:
        return None
    median = float(np.median(residuals))
    value = float(np.median(np.abs(residuals - median)))
    return value if math.isfinite(value) else None


def _uncertainty_from_gradient(gradient: np.ndarray, covariance: np.ndarray | None) -> float | None:
    if covariance is None:
        return None
    try:
        variance = float(gradient @ covariance @ gradient)
    except (FloatingPointError, ValueError):
        return None
    if not math.isfinite(variance):
        return None
    return math.sqrt(max(variance, 0.0))


def _fit_model(
    records: Sequence[_PreparedObservation],
    *,
    spatial_order: int,
    mad_threshold: float,
    max_iterations: int,
) -> _FitAttempt:
    source_ids = _unique_in_order([record.row.source_id for record in records])
    frame_ids = _unique_in_order([record.row.frame_id for record in records])
    if not records or not source_ids or not frame_ids:
        return _FitAttempt(None, "NO_VALID_REFERENCES", ("NO_VALID_REFERENCES",))

    spatial_center, spatial_scale = _spatial_normalization(records)
    active_indices = np.arange(len(records), dtype=int)
    rejected_input_indices: list[int] = []
    fit_flags: list[str] = []
    last_solution: _WlsSolution | None = None
    last_design: np.ndarray | None = None
    last_source_index: dict[Hashable, int] | None = None
    last_frame_index: dict[Hashable, int] | None = None
    last_spatial_start: int | None = None

    for iteration in range(max_iterations):
        active_records = [records[index] for index in active_indices]
        (
            design,
            magnitudes,
            errors,
            source_index,
            frame_index,
            spatial_start,
        ) = _build_design(
            active_records,
            source_ids=source_ids,
            frame_ids=frame_ids,
            spatial_order=spatial_order,
            spatial_center=spatial_center,
            spatial_scale=spatial_scale,
        )
        solution, failure_status = _weighted_least_squares(design, magnitudes, errors)
        if solution is None:
            if last_solution is None:
                flags = _dedupe_flags([failure_status or "DEGENERATE_DESIGN", *fit_flags])
                return _FitAttempt(None, failure_status or "DEGENERATE_DESIGN", flags)
            fit_flags.append("MAD_REJECTION_STOPPED_BY_RANK")
            break

        last_solution = solution
        last_design = design
        last_source_index = source_index
        last_frame_index = frame_index
        last_spatial_start = spatial_start
        residuals = solution.residuals
        residual_median = float(np.median(residuals))
        raw_mad = float(np.median(np.abs(residuals - residual_median)))
        error_floor = float(np.median(errors))
        robust_scale = max(1.4826 * raw_mad, error_floor, 1e-9)
        outlier_mask = np.abs(residuals - residual_median) > mad_threshold * robust_scale
        if not np.any(outlier_mask):
            fit_flags.append("MAD_REJECTION_CONVERGED")
            break

        removed_mask = outlier_mask.copy()
        candidate_indices = active_indices[~removed_mask]
        candidate_solution: _WlsSolution | None = None
        if candidate_indices.size >= solution.parameter_count:
            candidate_records = [records[index] for index in candidate_indices]
            (
                candidate_design,
                candidate_magnitudes,
                candidate_errors,
                _,
                _,
                _,
            ) = _build_design(
                candidate_records,
                source_ids=source_ids,
                frame_ids=frame_ids,
                spatial_order=spatial_order,
                spatial_center=spatial_center,
                spatial_scale=spatial_scale,
            )
            candidate_solution, _ = _weighted_least_squares(
                candidate_design,
                candidate_magnitudes,
                candidate_errors,
            )

        if candidate_solution is None:
            # A single bad cell in an additive source/frame table can create
            # several large residuals along its source and frame.  Removing all
            # of them at once may disconnect the bipartite design.  Fall back
            # to the largest residual that can be removed while preserving a
            # full-rank candidate, then let the next MAD iteration reassess the
            # remaining residuals.
            ranked_positions = np.flatnonzero(outlier_mask)
            ranked_positions = ranked_positions[
                np.argsort(-np.abs(residuals[ranked_positions] - residual_median), kind="mergesort")
            ]
            candidate_solution = None
            for position in ranked_positions:
                trial_mask = np.zeros_like(outlier_mask, dtype=bool)
                trial_mask[int(position)] = True
                trial_indices = active_indices[~trial_mask]
                if trial_indices.size < solution.parameter_count:
                    continue
                trial_records = [records[index] for index in trial_indices]
                (
                    trial_design,
                    trial_magnitudes,
                    trial_errors,
                    _,
                    _,
                    _,
                ) = _build_design(
                    trial_records,
                    source_ids=source_ids,
                    frame_ids=frame_ids,
                    spatial_order=spatial_order,
                    spatial_center=spatial_center,
                    spatial_scale=spatial_scale,
                )
                candidate_solution, _ = _weighted_least_squares(
                    trial_design,
                    trial_magnitudes,
                    trial_errors,
                )
                if candidate_solution is not None:
                    removed_mask = trial_mask
                    candidate_indices = trial_indices
                    fit_flags.append("MAD_REJECTION_SINGLE_STEP")
                    break
        if candidate_solution is None:
            fit_flags.append("MAD_REJECTION_STOPPED_BY_RANK")
            break
        rejected_input_indices.extend(
            records[index].input_index for index in active_indices[removed_mask]
        )
        active_indices = candidate_indices
        fit_flags.append("MAD_REJECTED_OBSERVATIONS")
        if iteration == max_iterations - 1:
            fit_flags.append("MAD_REJECTION_MAX_ITERATIONS")

    if last_solution is None or last_design is None or last_source_index is None or last_frame_index is None:
        return _FitAttempt(None, "DEGENERATE_DESIGN", ("DEGENERATE_DESIGN",))

    # The last solution is always associated with the last active set.  If a
    # candidate was accepted on the final loop iteration, solve once more so
    # the returned parameters and residuals describe that same active set.
    active_records = [records[index] for index in active_indices]
    (
        final_design,
        final_magnitudes,
        final_errors,
        source_index,
        frame_index,
        spatial_start,
    ) = _build_design(
        active_records,
        source_ids=source_ids,
        frame_ids=frame_ids,
        spatial_order=spatial_order,
        spatial_center=spatial_center,
        spatial_scale=spatial_scale,
    )
    final_solution, final_failure = _weighted_least_squares(final_design, final_magnitudes, final_errors)
    if final_solution is None:
        # This should only be reachable after an unusual numerical change
        # between the candidate check and the final solve.  Return a clear
        # status instead of exposing a linear-algebra exception.
        return _FitAttempt(None, final_failure or "DEGENERATE_DESIGN", (final_failure or "DEGENERATE_DESIGN",))

    reference_frame_id = frame_ids[0]
    raw_frame_values: dict[Hashable, float] = {reference_frame_id: 0.0}
    frame_gradients: dict[Hashable, np.ndarray] = {
        reference_frame_id: np.zeros(final_solution.parameter_count, dtype=float)
    }
    for frame_id, column in frame_index.items():
        raw_frame_values[frame_id] = float(final_solution.beta[column])
        gradient = np.zeros(final_solution.parameter_count, dtype=float)
        gradient[column] = 1.0
        frame_gradients[frame_id] = gradient

    ordered_frame_values = np.asarray([raw_frame_values[frame_id] for frame_id in frame_ids], dtype=float)
    median_order = np.argsort(ordered_frame_values, kind="mergesort")
    if len(frame_ids) % 2:
        median_indices = (int(median_order[len(frame_ids) // 2]),)
    else:
        median_indices = (
            int(median_order[len(frame_ids) // 2 - 1]),
            int(median_order[len(frame_ids) // 2]),
        )
    median_shift = float(np.mean([ordered_frame_values[index] for index in median_indices]))
    shift_gradient = np.mean(
        [frame_gradients[frame_ids[index]] for index in median_indices],
        axis=0,
    )

    frame_zero_points = {
        frame_id: float(raw_frame_values[frame_id] - median_shift) for frame_id in frame_ids
    }
    relative_magnitudes = {
        source_id: float(final_solution.beta[column] + median_shift)
        for source_id, column in source_index.items()
    }

    frame_uncertainties: dict[Hashable, float | None] = {}
    for frame_id in frame_ids:
        gradient = frame_gradients[frame_id] - shift_gradient
        frame_uncertainties[frame_id] = _uncertainty_from_gradient(gradient, final_solution.covariance)
    source_uncertainties: dict[Hashable, float | None] = {}
    for source_id, column in source_index.items():
        gradient = np.zeros(final_solution.parameter_count, dtype=float)
        gradient[column] = 1.0
        gradient += shift_gradient
        source_uncertainties[source_id] = _uncertainty_from_gradient(gradient, final_solution.covariance)

    if final_solution.covariance is not None:
        fit_flags.append("UNCERTAINTY_ESTIMATED")
    else:
        fit_flags.append("UNCERTAINTY_UNAVAILABLE_NO_DOF")
    if final_solution.condition_number > 1e12:
        fit_flags.append("ILL_CONDITIONED_DESIGN")
    fit_flags.extend(("MEDIAN_ZERO_POINT_FIXED",))
    if spatial_order > 0:
        fit_flags.append("SPATIAL_TERMS_ENABLED")

    model = _FittedModel(
        frame_zero_points=frame_zero_points,
        relative_magnitudes=relative_magnitudes,
        frame_uncertainties=frame_uncertainties,
        source_uncertainties=source_uncertainties,
        spatial_coefficients=tuple(float(value) for value in final_solution.beta[spatial_start:]),
        frame_ids=frame_ids,
        source_ids=source_ids,
        reference_frame_id=reference_frame_id,
        spatial_center=spatial_center,
        spatial_scale=spatial_scale,
        inlier_local_indices=active_indices.copy(),
        residuals=final_solution.residuals.copy(),
        rank=final_solution.rank,
        parameter_count=final_solution.parameter_count,
        condition_number=final_solution.condition_number,
        degrees_of_freedom=final_solution.degrees_of_freedom,
        rejected_input_indices=tuple(dict.fromkeys(rejected_input_indices)),
        flags=_dedupe_flags(fit_flags),
    )
    return _FitAttempt(model, "VALID", model.flags)


def _split_source_block(
    source_ids: Sequence[Hashable],
    *,
    validation_fraction: float,
    random_state: int | np.random.Generator | None,
    minimum_training_sources: int,
) -> tuple[Hashable, ...]:
    if validation_fraction <= 0 or len(source_ids) <= minimum_training_sources:
        return ()
    requested = max(1, int(round(len(source_ids) * validation_fraction)))
    maximum = len(source_ids) - minimum_training_sources
    validation_count = min(requested, maximum)
    if validation_count <= 0:
        return ()
    ordered = list(_stable_id_order(source_ids))
    if isinstance(random_state, np.random.Generator):
        generator = random_state
    else:
        generator = np.random.default_rng(random_state)
    selected_indices = generator.permutation(len(ordered))[:validation_count]
    return tuple(_stable_id_order([ordered[int(index)] for index in selected_indices]))


def _source_block_validation(
    records: Sequence[_PreparedObservation],
    model: _FittedModel,
    *,
    spatial_order: int,
) -> tuple[float | None, float | None, int, tuple[str, ...]]:
    """Evaluate unseen source blocks with frame terms fixed by training data."""

    if not records:
        return None, None, 0, ("NO_VALIDATION_SAMPLES",)
    frame_values = model.frame_zero_points
    x_values = np.asarray([record.row.x for record in records], dtype=float)
    y_values = np.asarray([record.row.y for record in records], dtype=float)
    spatial = _spatial_features(
        x_values,
        y_values,
        spatial_order=spatial_order,
        center=model.spatial_center,
        scale=model.spatial_scale,
    )
    spatial_coefficients = np.asarray(model.spatial_coefficients, dtype=float)
    spatial_prediction = spatial @ spatial_coefficients if spatial.shape[1] else np.zeros(len(records), dtype=float)

    values_by_source: dict[Hashable, list[tuple[float, float]]] = {}
    flags: list[str] = []
    for index, record in enumerate(records):
        if record.row.frame_id not in frame_values:
            flags.append("VALIDATION_FRAME_NOT_IN_TRAINING")
            continue
        adjusted = (
            record.row.instrumental_magnitude
            - frame_values[record.row.frame_id]
            - float(spatial_prediction[index])
        )
        values_by_source.setdefault(record.row.source_id, []).append(
            (adjusted, record.row.magnitude_error)
        )

    validation_residuals: list[float] = []
    for values in values_by_source.values():
        adjusted_values = np.asarray([value for value, _ in values], dtype=float)
        errors = np.asarray([error for _, error in values], dtype=float)
        weights = 1.0 / np.square(errors)
        source_mean = float(np.sum(weights * adjusted_values) / np.sum(weights))
        validation_residuals.extend(float(value - source_mean) for value in adjusted_values)

    if not validation_residuals:
        return None, None, 0, _dedupe_flags([*flags, "NO_VALIDATION_SAMPLES"])
    residual_array = np.asarray(validation_residuals, dtype=float)
    rms = float(np.sqrt(np.mean(np.square(residual_array))))
    mad = _residual_mad(residual_array)
    return rms, mad, int(residual_array.size), _dedupe_flags(flags)


def _empty_result(
    status: str,
    flags: Sequence[str],
    *,
    excluded_observation_count: int = 0,
    spatial_order: int = 0,
    validation_source_ids: Sequence[Hashable] = (),
) -> RelativePhotometryResult:
    return RelativePhotometryResult(
        status=status,
        flags=_dedupe_flags(flags),
        excluded_observation_count=excluded_observation_count,
        spatial_order=spatial_order,
        validation_source_ids=tuple(validation_source_ids),
    )


def fit_relative_photometry(
    observations: Sequence[Any],
    *,
    spatial_order: int = 0,
    spatial_terms: bool | int | None = None,
    include_spatial_terms: bool | None = None,
    validation_fraction: float = 0.2,
    random_state: int | np.random.Generator | None = 0,
    mad_threshold: float = 5.0,
    max_iterations: int = 8,
    min_sources: int = 2,
    min_frames: int = 2,
) -> RelativePhotometryResult:
    """Fit a robust, directory-independent relative photometric scale.

    Parameters
    ----------
    observations:
        Rows containing ``frame_id``, ``source_id``, ``instrumental_magnitude``,
        ``magnitude_error``, ``x``, ``y``, ``quality_passed``, ``is_moving``, and
        optional ``is_variable_optional``.  Mappings, objects, the public row
        dataclass, and positional rows in that order are accepted.
    spatial_order:
        ``0`` disables spatial terms; ``1`` adds normalized ``x`` and ``y``;
        ``2`` additionally adds ``x²``, ``xy``, and ``y²``.  Coefficients are in
        normalized pixel coordinates and are only relative nuisance terms.
    spatial_terms / include_spatial_terms:
        Convenience aliases.  ``True`` means ``spatial_order=1`` and an integer
        selects that order.  Supplying both aliases with conflicting values is
        rejected as a configuration error.
    validation_fraction:
        Fraction of source IDs held out as a complete source block.  The
        held-out source magnitudes are estimated only after the training frame
        terms have been fitted, so the validation block cannot set the frame
        zero points.
    mad_threshold / max_iterations:
        Iterative residual-MAD rejection settings.  The weighted fit uses
        ``1 / magnitude_error²`` weights.

    Returns
    -------
    RelativePhotometryResult
        A non-throwing data result for insufficient rows or a degenerate design
        matrix.  Configuration errors still raise ``ValueError`` because they
        are caller errors rather than data failures.
    """

    if spatial_terms is not None:
        requested_order = 1 if spatial_terms is True else int(spatial_terms)
        if spatial_order not in (0, requested_order):
            raise ValueError("spatial_order and spatial_terms disagree")
        spatial_order = requested_order
    if include_spatial_terms is not None:
        requested_order = 1 if include_spatial_terms else 0
        if spatial_order not in (0, requested_order):
            raise ValueError("spatial_order and include_spatial_terms disagree")
        spatial_order = requested_order
    if spatial_order not in (0, 1, 2):
        raise ValueError("spatial_order must be 0, 1, or 2")
    if not math.isfinite(float(validation_fraction)) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not math.isfinite(float(mad_threshold)) or mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least 1")
    if int(min_sources) < 1 or int(min_frames) < 1:
        raise ValueError("min_sources and min_frames must be positive")

    rows = list(observations)
    require_coordinates = spatial_order > 0
    prepared: list[_PreparedObservation] = []
    exclusion_reasons: Counter[str] = Counter()
    for input_index, row in enumerate(rows):
        converted, reason = _coerce_observation(
            row,
            input_index,
            require_coordinates=require_coordinates,
        )
        if converted is None:
            exclusion_reasons[reason or "INVALID_OBSERVATION"] += 1
        else:
            prepared.append(converted)

    global_flags: list[str] = []
    reason_flags = {
        "QUALITY_REJECTED": "QUALITY_FAILED_EXCLUDED",
        "MOVING_REJECTED": "MOVING_EXCLUDED",
        "VARIABLE_REJECTED": "VARIABLE_EXCLUDED",
        "INVALID_OBSERVATION": "INVALID_OBSERVATION_EXCLUDED",
        "INVALID_MAGNITUDE_ERROR": "INVALID_MAGNITUDE_ERROR_EXCLUDED",
        "INVALID_COORDINATES": "INVALID_COORDINATES_EXCLUDED",
    }
    for reason, flag in reason_flags.items():
        if exclusion_reasons[reason]:
            global_flags.append(flag)

    excluded_count = len(rows) - len(prepared)
    if not prepared:
        global_flags.append("NO_VALID_REFERENCES")
        return _empty_result(
            "NO_VALID_REFERENCES",
            global_flags,
            excluded_observation_count=excluded_count,
            spatial_order=spatial_order,
        )

    source_ids = _unique_in_order([record.row.source_id for record in prepared])
    frame_ids = _unique_in_order([record.row.frame_id for record in prepared])
    if len(source_ids) < min_sources or len(frame_ids) < min_frames:
        global_flags.append("INSUFFICIENT_SOURCES_OR_FRAMES")
        return _empty_result(
            "INSUFFICIENT_SAMPLES",
            global_flags,
            excluded_observation_count=excluded_count,
            spatial_order=spatial_order,
        )

    validation_source_ids = _split_source_block(
        source_ids,
        validation_fraction=float(validation_fraction),
        random_state=random_state,
        minimum_training_sources=max(2, int(min_sources)),
    )
    validation_set = set(validation_source_ids)
    if validation_source_ids:
        global_flags.append("HOLDOUT_SOURCE_BLOCK")
        training_records = [record for record in prepared if record.row.source_id not in validation_set]
        validation_records = [record for record in prepared if record.row.source_id in validation_set]
    else:
        global_flags.append("NO_HOLDOUT_VALIDATION")
        training_records = prepared
        validation_records = []

    final_attempt = _fit_model(
        prepared,
        spatial_order=spatial_order,
        mad_threshold=float(mad_threshold),
        max_iterations=int(max_iterations),
    )
    if final_attempt.model is None:
        return _empty_result(
            final_attempt.status,
            [*global_flags, *final_attempt.flags],
            excluded_observation_count=excluded_count,
            spatial_order=spatial_order,
            validation_source_ids=validation_source_ids,
        )

    model = final_attempt.model
    global_flags.extend(model.flags)
    if model.rejected_input_indices:
        global_flags.append("MAD_REJECTION_APPLIED")

    training_rms = float(np.sqrt(np.mean(np.square(model.residuals)))) if model.residuals.size else None
    training_mad = _residual_mad(model.residuals)
    reference_count_by_frame: Counter[Hashable] = Counter()
    reference_count_by_source: Counter[Hashable] = Counter()
    for local_index in model.inlier_local_indices:
        row = prepared[int(local_index)].row
        reference_count_by_frame[row.frame_id] += 1
        reference_count_by_source[row.source_id] += 1

    validation_rms: float | None = None
    validation_mad: float | None = None
    validation_count = 0
    if validation_records:
        training_attempt = _fit_model(
            training_records,
            spatial_order=spatial_order,
            mad_threshold=float(mad_threshold),
            max_iterations=int(max_iterations),
        )
        if training_attempt.model is None:
            global_flags.append("VALIDATION_MODEL_UNAVAILABLE")
            global_flags.extend(training_attempt.flags)
        else:
            validation_rms, validation_mad, validation_count, validation_flags = _source_block_validation(
                validation_records,
                training_attempt.model,
                spatial_order=spatial_order,
            )
            global_flags.extend(validation_flags)
    else:
        validation_count = 0

    return RelativePhotometryResult(
        status="VALID",
        flags=_dedupe_flags(global_flags),
        frame_zero_points=model.frame_zero_points,
        relative_magnitudes=model.relative_magnitudes,
        frame_zero_point_uncertainties=model.frame_uncertainties,
        relative_magnitude_uncertainties=model.source_uncertainties,
        reference_sample_count=int(model.inlier_local_indices.size),
        reference_sample_count_by_frame=dict(reference_count_by_frame),
        reference_sample_count_by_source=dict(reference_count_by_source),
        training_residual_rms=training_rms,
        training_residual_mad=training_mad,
        validation_residual_rms=validation_rms,
        validation_residual_mad=validation_mad,
        validation_sample_count=validation_count,
        validation_source_ids=tuple(validation_source_ids),
        rejected_observation_indices=model.rejected_input_indices,
        excluded_observation_count=excluded_count,
        spatial_coefficients=model.spatial_coefficients,
        spatial_order=spatial_order,
    )


# Explicit aliases for callers that describe the same operation as fitting a
# relative scale rather than fitting photometry.
fit_relative_scale = fit_relative_photometry
fit_relative_photometric_scale = fit_relative_photometry


__all__ = [
    "Observation",
    "PhotometryObservation",
    "RelativePhotometryObservation",
    "RelativePhotometryResult",
    "fit_relative_photometry",
    "fit_relative_photometric_scale",
    "fit_relative_scale",
]
