from __future__ import annotations

import json

from rst19.feature_evidence_matrix import (
    build_feature_evidence_matrix,
    write_feature_evidence_matrix_artifacts,
)


def _write(path, text: str):
    path.write_text(text, encoding="utf-8-sig")
    return path


def test_feature_evidence_matrix_keeps_layers_separate_and_writes_json(tmp_path) -> None:
    morphology = _write(
        tmp_path / "morphology.csv",
        "feature_class,feature_class_label,candidate_count,quality_count,quality_fraction,"
        "high_flux_snr_rejection_fraction,spike_or_support_fraction,weak_or_background_fraction\n"
        "range_anomaly,范围异常,20,0,0.0,1.0,0.0,0.0\n"
        "crowded_blend,拥挤,100,0,0.0,1.0,0.5,0.2\n"
        "compact_quality,紧凑,100,90,0.9,0.0,0.0,0.0\n",
    )
    feature_cross = _write(
        tmp_path / "cross.csv",
        "feature_class,feature_class_label,anchor_candidate_count,candidate_presence_ge_required_count,"
        "candidate_presence_fraction,quality_presence_ge_required_count,quality_presence_fraction,"
        "candidate_any_response_fraction,candidate_same_class_response_fraction,"
        "quality_any_response_fraction,quality_same_class_response_fraction\n"
        "range_anomaly,范围异常,20,2,0.1,0,,0.2,0.1,,\n"
        "crowded_blend,拥挤,100,80,0.8,0,,0.8,0.5,,\n"
        "compact_quality,紧凑,100,80,0.8,70,0.7,0.9,0.8,0.95,0.85\n",
    )
    psf = _write(
        tmp_path / "psf.csv",
        "feature_class,feature_class_label,sample_count,leaveout_valid_count,"
        "leaveout_correlation_median,leaveout_correlation_ge_threshold_fraction,"
        "leaveout_residual_median\n"
        "range_anomaly,范围异常,2,2,0.75,0.2,0.7\n"
        "crowded_blend,拥挤,2,2,0.2,0.0,0.98\n"
        "compact_quality,紧凑,2,2,0.9,1.0,0.4\n",
    )
    spatial = _write(
        tmp_path / "spatial.csv",
        "feature_class,feature_class_label,local_template_available_cell_count,"
        "local_template_source_count_median,local_valid_count,local_fallback_count,"
        "paired_correlation_count,local_correlation_median,local_residual_median,"
        "local_correlation_improved_fraction,local_residual_reduced_fraction,"
        "median_correlation_delta_local_minus_global,median_residual_delta_local_minus_global\n"
        "crowded_blend,拥挤,3,6.0,11,5,11,0.17,0.985,0.64,0.64,0.002,-0.0002\n"
        "compact_quality,紧凑,3,6.0,11,5,11,0.84,0.55,0.45,0.45,-0.002,0.0002\n",
    )

    result = build_feature_evidence_matrix(morphology, feature_cross, psf, psf_spatial_path=spatial)
    by_class = {row.feature_class: row for row in result.rows}
    assert by_class["compact_quality"].evidence_pattern == "质量/时序/PSF 三层一致"
    assert by_class["compact_quality"].recommended_audit_stage == "catalog_wcs"
    assert by_class["compact_quality"].counting_policy == "identity_review_queue"
    assert by_class["crowded_blend"].evidence_pattern == "候选持续但质量/PSF 不支持"
    assert by_class["crowded_blend"].recommended_audit_stage == "joint_psf_deblend"
    assert by_class["crowded_blend"].counting_policy == "candidate_only"
    assert by_class["range_anomaly"].evidence_pattern == "值域优先审计"
    assert by_class["range_anomaly"].recommended_audit_stage == "value_domain"
    assert by_class["compact_quality"].quality_fraction == 0.9
    assert by_class["crowded_blend"].dominant_secondary_feature == "spike_or_support"
    assert by_class["crowded_blend"].dominant_secondary_fraction == 0.5
    assert by_class["compact_quality"].candidate_persistence_fraction == 0.8
    assert by_class["compact_quality"].quality_response_fraction == 0.7
    assert by_class["compact_quality"].candidate_same_class_response_fraction == 0.8
    assert by_class["compact_quality"].quality_same_class_response_fraction == 0.85
    assert 0.70 < by_class["compact_quality"].candidate_persistence_wilson95_low < 0.75
    assert 0.80 < by_class["compact_quality"].quality_fraction_wilson95_low < 0.85
    assert by_class["range_anomaly"].quality_response_wilson95_low == 0.0
    assert by_class["compact_quality"].leaveout_correlation_median == 0.9
    assert by_class["compact_quality"].spatial_local_valid_count == 11
    assert by_class["compact_quality"].spatial_local_correlation_median == 0.84
    assert by_class["crowded_blend"].spatial_local_fallback_count == 5
    assert by_class["crowded_blend"].spatial_median_residual_delta == -0.0002
    assert result.rows[-1].feature_class == "other_rejected"
    assert result.psf_spatial_path == str(spatial)

    output = write_feature_evidence_matrix_artifacts(result, tmp_path / "matrix")
    assert (output / "feature_evidence_matrix.csv").is_file()
    payload = json.loads((output / "feature_evidence_matrix.json").read_text(encoding="utf-8"))
    assert payload["rows"][0]["feature_class"] == "range_anomaly"
    assert payload["rows"][7]["evidence_pattern"] == "质量/时序/PSF 三层一致"
    assert payload["rows"][7]["recommended_audit_stage"] == "catalog_wcs"
    assert payload["parameters"]["pattern_is_not_truth"] is True
    assert payload["parameters"]["routing_definition"]
