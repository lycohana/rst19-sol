"""Generate the complete module/function and entry-point migration inventory."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def symbols(tree:ast.AST,prefix:str="")->list[str]:
    result=[]
    for node in getattr(tree,"body",[]):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            name=f"{prefix}{node.name}"
            result.append(name)
            if isinstance(node,ast.ClassDef):result.extend(symbols(node,f"{name}."))
    return result


if __name__=="__main__":
    scripts=tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    manifest_path=ROOT/"backend/migration.json"
    previous_payload={}
    previous={}
    if manifest_path.is_file():
        previous_payload=json.loads(manifest_path.read_text(encoding="utf-8"))
        previous={item["python_module"]:item["migration_status"]
                  for item in previous_payload.get("modules",[])}
    native_complete_modules={"format_audit","format_audit_cli",
                             "sequence_feature_comparison","feature_sequence_compare_cli",
                             "feature_raw_evidence_audit","feature_raw_evidence_audit_cli",
                             "signed_null_forward_overlap","signed_null_forward_overlap_cli",
                             "signed_null","signed_null_cli"}
    tools={name:target for name,target in scripts.items() if target not in ("rst19.gui:main","rst19.desktop_launcher:main")}
    tools["rst19"]="native Rust executable; Python CLI parser retained only for desktop form metadata"
    tools["rst19-format-audit"]="native Rust executable and desktop worker route; Python entrypoint/module import removed; static desktop schema remains"
    tools["rst19-feature-sequence-compare"]="native Rust JSONL operation, standalone executable, and desktop worker route; Python console entrypoint removed; comparison module retained as a parity baseline and shared class-order source"
    tools["rst19-feature-raw-evidence-audit"]="native Rust JSONL operation, standalone executable, and desktop worker route; Python console entrypoint removed; CSV join and statistical summaries match the Python baseline"
    tools["rst19-signed-null-overlap"]="native Rust FITS detection, local spatial audit, standalone executable, and desktop worker route; Python CLI entrypoint removed"
    tools["rst19-signed-null-audit"]="native Rust single-frame signed-null detector audit, standalone executable, and desktop worker route; Python console entrypoint removed; Python runtime packaging remains until all backend modules migrate"
    administrative={"gui.py","desktop_worker.py","desktop_launcher.py","__init__.py","__main__.py"}
    modules=[]
    for path in sorted((ROOT/"src/rst19").glob("*.py")):
        if path.name in administrative:continue
        text=path.read_text(encoding="utf-8")
        status="complete" if path.stem in native_complete_modules else previous.get(f"rst19.{path.stem}",
                            "in_progress" if path.stem in {"fits","models","detection","wcs","attitude","matching","catalog","photometry","pipeline","sequence"} else "pending")
        modules.append({"python_module":f"rst19.{path.stem}","sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "line_count":len(text.splitlines()),"symbols":symbols(ast.parse(text)),
                        "migration_status":status})
    payload={"objective":"all backend calculations implemented in Rust; production runtime has no Python dependency",
             "complete":False,"python_runtime_removed":False,"modules":modules,"tools":tools,
             "native_implemented":["fits raw typed primary-HDU reader","auxiliary decode and plausibility",
                                   "exposure units","f64 sigma-clipped statistics","f64 local background maps",
                                   "TAN projection","quaternion boresight","affine projection and velocity transform",
                                   "prior-WCS affine fitting with MAD clipping, geometry diagnostics, and leave-one-out residuals",
                                   "auxiliary attitude boresight residual audit, header cross-check, range summaries, and CSV/JSON CLI artifacts",
                                   "time-aware track linear fit and robust registration-derived motion floor",
                                   "mutual-nearest high-SNR translation registration with displacement consensus",
                                   "sparse connected-component track-to-source assignment with gate and soft-cost hooks",
                                   "point-sequence translation registration, affine drift correction, track construction, and classification",
                                   "supplemental high-SNR quality-point fast-motion association with existing-track reservation and one-frame-gap prediction",
                                   "candidate-consensus anchoring and mutual-nearest matching, temporal SNR channels, quality-track overlap, motion-fit rejection, and optional raw-image validity/aperture support/shape/PSF-correlation/line-artifact gates",
                                   "registered temporal median and MAD-clipped coadd reference construction with masking and bilinear registration",
                                   "temporal-reference candidate proposals with multiscale Gaussian filtering, reference-shape checks, and per-frame forced PSF response with masks/saturation",
                                   "bilinearly sampled motion-feature line-profile continuity and relative fill diagnostics",
                                   "single-frame trail rejection reasons, local line-model support, and sequence line-shape admission",
                                   "cross-frame trail shape/length/direction gates, motion-axis majority, artifact compatibility, and edge truncation",
                                   "longest contiguous observation support and bounded two-run internal-gap bridge gate",
                                   "PSF-deconvolved exposure-line versus frame-speed consistency with FITS timing metadata",
                                   "sequence-length confirmation presence and stable motion-feature rejection reason codes",
                                   "metadata-aware line proposal and exposure-aware short-trail area thresholds",
                                   "validated relative frame-time axis and median reference interval",
                                   "UTC-normalized FITS DATE-OBS parsing with monotonic frame fallback",
                                   "soft pairwise line-feature association cost with endpoint and motion-axis evidence",
                                   "PCA collinear track span/RMS/axis and exposure-aware motion speed",
                                   "best monotonic one-per-frame collinear motion group selection",
                                   "review-gated single-frame trail association by constant-velocity prediction, temporal gap, line-axis, PSF-relative length, and artifact-overlap evidence",
                                   "one-to-one point-motion and exposure-line evidence linking by shared frames, direction, displacement scale, and perpendicular separation",
                                   "iterative disjoint grouping of long single-frame trails by monotonic collinear geometry",
                                   "per-frame sequence residual connected components with gap grouping, PCA line geometry, profile continuity, edge and line-artifact audits",
                                   "coherent raw per-frame line-proposal chains with elapsed-time and exposure-scaled displacement gates, direction continuity, and strong-shape admission",
                                   "single-frame raw-pixel finite-line proposal search with masked Gaussian normalization, spatial seed budgets, local refinement, compact-source controls, and geometry suppression"],
             "native_detection_helpers":["f64 masked Gaussian and starlet filters","sliding maximum filter",
                                          "peak-window extraction and NMS","Gaussian-anchor multichannel merging",
                                          "local-pixel finite-line proposal search with point and double-source controls",
                                          "connected-component PCA line audit","exact 2-D KD-tree queries",
                                          "source aperture photometry shape and noise","fixed single/pair PSF BIC and covariance",
                                          "Gaussian close-pair guard","local residual deblend proposals","legacy compact KD-tree tie handling"],
             "native_detection_execution":"float64 and float32 Gaussian/hybrid/ensemble; optional local deblend; shared-background shape validation; exactly overlapping PSF templates return no identifiable pair evidence",
             "native_matching_helpers":["rectangular shortest augmenting path assignment with baseline ties","sparse connected-component global candidate assignment","historical greedy candidate assignment"],
             "native_catalog_execution":"complete source fields and validation; offline CSV aliases/provenance/conflicts; Julian-year proper-motion propagation; JSONL catalog.load/catalog.propagate",
             "native_matching_execution":"TAN or full affine WCS projection; finite projected catalog gating; radius KD-tree; global/greedy assignment; all match fields and statistics; catalog arrays or direct CSV path; JSONL matching.catalog",
             "native_astrometry_execution":"wcs.affine.fit, wcs.validate.matches, and wcs.validate.sequence; prior-WCS local affine refinement, robust clipping, per-frame leave-one-out checks; not blind plate solving",
             "native_attitude_execution":"attitude.audit.frames and attitude.audit.paths; quaternion/boresight residuals, header pointing consistency, missing-auxiliary handling, RA/Dec and attitude range summaries; rst19-aux-audit writes Python-byte-identical CSV and field-identical JSON",
             "native_format_audit_execution":"audit.format_paths and rst19-format-audit inspect FITS dimensions, storage types, scaling, special values, timestamps and auxiliary position/velocity consistency; JSON fields and CSV bytes match the Python baseline",
             "native_sequence_feature_comparison_execution":"sequence.feature_parameter_compare reads existing sequence JSON only; computes first-frame counts, per-class candidate/quality persistence and class transitions; Rust JSONL engine, standalone CLI, and desktop worker match Python result fields and three CSV artifacts byte-for-byte",
             "native_feature_raw_evidence_audit_execution":"audit.feature_raw_evidence joins source_catalog and stratified raw-sample CSV on unique detection_id values; validates fields/classes, preserves requested metric/class order, excludes other_rejected by default, and uses NumPy-compatible linear quantiles; Rust JSONL engine, standalone CLI, and desktop worker match Python fields and CSV bytes",
             "native_sequence_execution":"sequence.translation.estimate, sequence.frame_assign, sequence.track_motion.fit, sequence.motion_displacement_floor, sequence.track_detections, sequence.fast_point_movers, sequence.candidate_consensus with optional per-frame raw-image gates, sequence.reference.median, sequence.reference.coadd, sequence.temporal_reference_candidates for multiscale Gaussian proposals plus forced per-frame PSF responses, sequence.line_profile_metrics for bilinear trail continuity/fill diagnostics, sequence.trail_shape_evidence for single-frame rejection reasons and sequence shape admission, sequence.motion_feature_track_gates for cross-frame shape/motion-axis/artifact/edge checks, sequence.motion_feature_continuity_gates for continuous observation runs and bounded interior-gap bridging, sequence.motion_feature_exposure_evidence for PSF-deconvolved exposure speed versus inter-frame motion, sequence.motion_feature_confirmation_evidence for required coverage and stable rejection codes, sequence.motion_proposal_thresholds for metadata-aware raw trail proposal floors, sequence.motion_time_axis for validated elapsed times and median reference interval, sequence.observation_time_axis for UTC-normalized FITS DATE-OBS parsing with monotonic frame fallback, sequence.motion_feature_pair_geometry for the soft trail association cost, sequence.fit_collinear_motion_geometry for PCA trail span and time-aware legacy speed, sequence.best_collinear_motion_feature_group for deterministic monotonic cross-frame candidate grouping, sequence.stitch_unassociated_single_frame_points for review-gated constant-velocity singleton recovery, sequence.link_motion_evidence_tracks for one-to-one point-motion/exposure-line evidence de-duplication, sequence.rescue_collinear_motion_feature_groups for iterative disjoint long-trail grouping, sequence.coherent_raw_motion_point_indices for adjacent-frame raw line-proposal chains with shape and timing gates; local-pixel trail proposals are implemented as a reusable leaf; multiscale/raw-candidate feature assembly and high-level multi-frame orchestration remain pending; affine RANSAC uses deterministic SplitMix64 rather than NumPy PCG64",
             "native_sequence_execution":"sequence.translation.estimate, sequence.frame_assign, sequence.track_motion.fit, sequence.motion_displacement_floor, sequence.track_detections, sequence.fast_point_movers, sequence.candidate_consensus with optional per-frame raw-image gates, sequence.reference.median, sequence.reference.coadd, sequence.temporal_reference_candidates for multiscale Gaussian proposals plus forced per-frame PSF responses, sequence.line_profile_metrics for bilinear trail continuity/fill diagnostics, sequence.motion_feature_frame_components for thresholded per-frame connected components, gap grouping, PCA line geometry and morphology audit, sequence.trail_shape_evidence for single-frame rejection reasons and sequence shape admission, sequence.motion_feature_track_gates for cross-frame shape/motion-axis/artifact/edge checks, sequence.motion_feature_continuity_gates for continuous observation runs and bounded interior-gap bridging, sequence.motion_feature_exposure_evidence for PSF-deconvolved exposure speed versus inter-frame motion, sequence.motion_feature_confirmation_evidence for required coverage and stable rejection codes, sequence.motion_proposal_thresholds for metadata-aware raw trail proposal floors, sequence.motion_time_axis for validated elapsed times and median reference interval, sequence.observation_time_axis for UTC-normalized FITS DATE-OBS parsing with monotonic frame fallback, sequence.motion_feature_pair_geometry for the soft trail association cost, sequence.fit_collinear_motion_geometry for PCA trail span and time-aware legacy speed, sequence.best_collinear_motion_feature_group for deterministic monotonic cross-frame candidate grouping, sequence.stitch_unassociated_single_frame_points for review-gated constant-velocity singleton recovery, sequence.link_motion_evidence_tracks for one-to-one point-motion/exposure-line evidence de-duplication, sequence.rescue_collinear_motion_feature_groups for iterative disjoint long-trail grouping, sequence.coherent_raw_motion_point_indices for adjacent-frame raw line-proposal chains with shape and timing gates; local-pixel trail proposals are implemented as a reusable leaf; multiscale/raw-candidate feature assembly and high-level multi-frame orchestration remain pending; affine RANSAC uses deterministic SplitMix64 rather than NumPy PCG64",
             "native_trail_proposals_execution":"trails.local_pixel_proposals; masked constant-boundary Gaussian whitening, spatially tiled local maxima, per-length/angle line responses, local refinement, point/double-source negative controls, and geometry NMS; candidate proposals only",
             "native_photometry_execution":"instrumental/apparent/legacy absolute arithmetic; structured parallax/model-distance/catalog absolute estimates; derived error/interval/strict gates; weighted zero point/color fit; clipping/holdout/catalog quality gates/sample roles; calibration application and error; native JSONL operations",
             "native_source_photometry":"complete source table status/flags/reference/model magnitude separation; faintest ranking, ties, calibration consistency and partial selection scope; JSONL photometry.table/photometry.faintest",
             "native_frame_pipeline":"direct FITS single-frame analysis through detection/matching/calibration/source rows/faintest; native library recalibration without repeating detection; JSONL pipeline.analyze",
             "native_cli_contracts":{"rst19":"native rst19 executable is the public command; rst19-native alias retained; stdout/json-out/error contracts match across 89 parity invocations; standalone Windows executable opens help with PATH disabled; package build/verification and macOS/Linux package validation pending","rst19-aux-audit":"single FITS or sorted directory input; CSV/JSON output matches Python; cross-platform packaging and broader parser edges pending","rst19-format-audit":"single FITS or directory input; Rust JSONL engine, standalone CLI, and desktop worker match Python result fields and byte-identical CSV artifacts on synthetic and real FITS; package build and cross-platform acceptance pending","rst19-feature-sequence-compare":"reads existing sequence JSON without FITS access or detection; JSONL engine, standalone CLI, and desktop worker match Python result fields and three CSV artifacts byte-for-byte; package build and cross-platform acceptance pending"},
             "verification":["tests/parity.rs","tests/measurement_parity.rs","tests/detection_parity.rs","tests/spatial_parity.rs","tests/float32_parity.rs","tests/assignment_parity.rs","tests/matching_parity.rs","tests/photometry_parity.rs","tests/calibration_parity.rs","tests/source_photometry_parity.rs","tests/pipeline_parity.rs","tests/wcs_sequence_parity.rs","tests/sequence_fast_parity.rs","tests/sequence_consensus_parity.rs","tests/sequence_temporal_reference_parity.rs","tests/sequence_reference_parity.rs","tests/sequence_trail_metrics_parity.rs","tests/sequence_trail_shape_parity.rs","tests/sequence_motion_feature_gates_parity.rs","tests/sequence_motion_feature_continuity_parity.rs","tests/sequence_motion_feature_exposure_parity.rs","tests/sequence_motion_feature_confirmation_parity.rs","tests/sequence_motion_proposal_thresholds_parity.rs","tests/sequence_motion_time_axis_parity.rs","tests/sequence_observation_time_axis_parity.rs","tests/sequence_motion_feature_pair_geometry_parity.rs","tests/sequence_motion_components_parity.rs","tests/collinear_motion_geometry_parity.rs","tests/collinear_motion_group_parity.rs","tests/single_frame_stitch_parity.rs","tests/motion_evidence_link_parity.rs","tests/collinear_motion_rescue_parity.rs","tests/coherent_raw_motion_points_parity.rs","tests/local_pixel_trail_proposals_parity.rs","tests/attitude_parity.rs","verify_fits.py","verify_detection.py","verify_float32.py","verify_assignment.py","verify_matching.py","verify_photometry.py","verify_pipeline.py","verify_wcs_sequence.py","verify_sequence_fast.py","verify_sequence_consensus.py","verify_sequence_temporal_reference.py","verify_sequence_reference.py","verify_sequence_trail_metrics.py","verify_sequence_trail_shape.py","verify_sequence_motion_feature_gates.py","verify_sequence_motion_feature_continuity.py","verify_sequence_motion_feature_exposure.py","verify_sequence_motion_feature_confirmation.py","verify_sequence_motion_proposal_thresholds.py","verify_sequence_motion_time_axis.py","verify_sequence_observation_time_axis.py","verify_sequence_motion_feature_pair_geometry.py","verify_sequence_motion_components.py","verify_collinear_motion_geometry.py","verify_collinear_motion_group.py","verify_single_frame_stitch.py","verify_motion_evidence_link.py","verify_collinear_motion_rescue.py","verify_coherent_raw_motion_points.py","verify_local_pixel_trail_proposals.py","verify_sequence_chain.py","verify_attitude.py","verify_format_audit.py","verify_spatial_ties.py","verify_full_snapshot.py"],
             "remaining":["full float32 detection acceptance coverage","blind plate solving and remaining astrometric workflows",
                          "remaining multiscale/raw-candidate feature assembly, high-level multi-frame orchestration, raw fast-candidate point workflows; exact NumPy PCG64 RANSAC sampling parity",
                          "all scientific audits and experiments","remaining CLI input/output contracts and exhaustive parser edge coverage","native desktop routing",
                          "remove Python from runtime packaging","full scientific and cross-platform acceptance"]}
    generated=payload
    payload={**generated,**previous_payload}
    payload.update({"objective":generated["objective"],"modules":modules,"tools":tools})
    payload["complete"]=previous_payload.get("complete",generated["complete"])
    payload["python_runtime_removed"]=previous_payload.get("python_runtime_removed",generated["python_runtime_removed"])
    payload["native_implemented"]=list(dict.fromkeys([
        *previous_payload.get("native_implemented",[]),*generated.get("native_implemented",[])]))
    contracts={**generated.get("native_cli_contracts",{}),**previous_payload.get("native_cli_contracts",{})}
    contracts["rst19-format-audit"]=generated["native_cli_contracts"]["rst19-format-audit"]
    contracts["rst19-feature-sequence-compare"]=generated["native_cli_contracts"]["rst19-feature-sequence-compare"]
    contracts["rst19-feature-raw-evidence-audit"]="joins existing source_catalog and stratified raw-summary CSV; JSONL engine, standalone CLI, and desktop worker match Python fields and CSV bytes for default/custom classes and metrics; package build and cross-platform acceptance pending"
    contracts["rst19-signed-null-overlap"]="reads FITS frames and reverse-quality CSV, runs the native detector, and writes local candidate/quality overlap evidence; parser, Rust unit, Python baseline, package and cross-platform acceptance pending"
    contracts["rst19-signed-null-audit"]="reads one FITS frame, performs positive and sign-flipped detections, source feature summaries, SNR-threshold leakage counts, and raw aperture evidence in Rust; synthetic parity passed at 1e-10 numeric tolerance; package and cross-platform acceptance pending"
    payload["native_cli_contracts"]=contracts
    payload["native_format_audit_execution"]=generated["native_format_audit_execution"]
    payload["native_sequence_feature_comparison_execution"]=generated["native_sequence_feature_comparison_execution"]
    payload["native_feature_raw_evidence_audit_execution"]=generated["native_feature_raw_evidence_audit_execution"]
    payload["native_implemented"]=list(dict.fromkeys([
        *payload.get("native_implemented",[]),
        "raw local-evidence audit joined to source catalogs by detection_id with finite-value filtering and linear quantiles"]))
    payload["native_implemented"]=list(dict.fromkeys([
        *payload.get("native_implemented",[]),
        "signed-null reverse-quality source overlap against native forward FITS detections with KD-tree radius and nearest queries and feature classification"]))
    payload["native_implemented"]=list(dict.fromkeys([
        *payload.get("native_implemented",[]),
        "single-frame signed-null source audit with sign-flipped detection, quality leakage thresholds, feature classification, range-transfer exclusion, and raw aperture evidence"]))
    payload["verification"]=list(dict.fromkeys([
        *previous_payload.get("verification",[]),*generated.get("verification",[])]))
    payload["verification"]=list(dict.fromkeys([
        *payload["verification"],"verify_sequence_feature_comparison.py"]))
    payload["verification"]=list(dict.fromkeys([
        *payload["verification"],"verify_feature_raw_evidence_audit.py"]))
    payload["verification"]=list(dict.fromkeys([
        *payload["verification"],"cargo test --lib signed_null_forward_overlap","tests/test_signed_null_forward_overlap.py"]))
    payload["verification"]=list(dict.fromkeys([
        *payload["verification"],"cargo test --lib signed_null","tests/test_signed_null.py","tests/test_desktop_worker.py"]))
    target=ROOT/"backend/migration.json"
    target.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{len(modules)} Python backend modules, {len(tools)} tools, {sum(len(m['symbols']) for m in modules)} symbols inventoried")
