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
    tools={name:target for name,target in scripts.items() if target not in ("rst19.gui:main","rst19.desktop_launcher:main")}
    administrative={"gui.py","desktop_worker.py","desktop_launcher.py","__init__.py","__main__.py"}
    modules=[]
    for path in sorted((ROOT/"src/rst19").glob("*.py")):
        if path.name in administrative:continue
        text=path.read_text(encoding="utf-8")
        modules.append({"python_module":f"rst19.{path.stem}","sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "line_count":len(text.splitlines()),"symbols":symbols(ast.parse(text)),
                        "migration_status":"in_progress" if path.stem in {"fits","models","detection","wcs","attitude","matching","catalog","photometry","pipeline","sequence","cli"} else "pending"})
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
                                   "one-to-one point-motion and exposure-line evidence linking by shared frames, direction, displacement scale, and perpendicular separation"],
             "native_detection_helpers":["f64 masked Gaussian and starlet filters","sliding maximum filter",
                                          "peak-window extraction and NMS","Gaussian-anchor multichannel merging",
                                          "connected-component PCA line audit","exact 2-D KD-tree queries",
                                          "source aperture photometry shape and noise","fixed single/pair PSF BIC and covariance",
                                          "Gaussian close-pair guard","local residual deblend proposals","legacy compact KD-tree tie handling"],
             "native_detection_execution":"float64 and float32 Gaussian/hybrid/ensemble; optional local deblend; shared-background shape validation; exactly overlapping PSF templates return no identifiable pair evidence",
             "native_matching_helpers":["rectangular shortest augmenting path assignment with baseline ties","sparse connected-component global candidate assignment","historical greedy candidate assignment"],
             "native_catalog_execution":"complete source fields and validation; offline CSV aliases/provenance/conflicts; Julian-year proper-motion propagation; JSONL catalog.load/catalog.propagate",
             "native_matching_execution":"TAN or full affine WCS projection; finite projected catalog gating; radius KD-tree; global/greedy assignment; all match fields and statistics; catalog arrays or direct CSV path; JSONL matching.catalog",
             "native_astrometry_execution":"wcs.affine.fit, wcs.validate.matches, and wcs.validate.sequence; prior-WCS local affine refinement, robust clipping, per-frame leave-one-out checks; not blind plate solving",
             "native_attitude_execution":"attitude.audit.frames and attitude.audit.paths; quaternion/boresight residuals, header pointing consistency, missing-auxiliary handling, RA/Dec and attitude range summaries; rst19-aux-audit writes Python-byte-identical CSV and field-identical JSON",
             "native_sequence_execution":"sequence.translation.estimate, sequence.frame_assign, sequence.track_motion.fit, sequence.motion_displacement_floor, sequence.track_detections, sequence.fast_point_movers, sequence.candidate_consensus with optional per-frame raw-image gates, sequence.reference.median, sequence.reference.coadd, sequence.temporal_reference_candidates for multiscale Gaussian proposals plus forced per-frame PSF responses, sequence.line_profile_metrics for bilinear trail continuity/fill diagnostics, sequence.trail_shape_evidence for single-frame rejection reasons and sequence shape admission, sequence.motion_feature_track_gates for cross-frame shape/motion-axis/artifact/edge checks, sequence.motion_feature_continuity_gates for continuous observation runs and bounded interior-gap bridging, sequence.motion_feature_exposure_evidence for PSF-deconvolved exposure speed versus inter-frame motion, sequence.motion_feature_confirmation_evidence for required coverage and stable rejection codes, sequence.motion_proposal_thresholds for metadata-aware raw trail proposal floors, and sequence.motion_time_axis for validated elapsed times and median reference interval, sequence.observation_time_axis for UTC-normalized FITS DATE-OBS parsing with monotonic frame fallback, sequence.motion_feature_pair_geometry for the soft trail association cost, sequence.fit_collinear_motion_geometry for PCA trail span and time-aware legacy speed, sequence.best_collinear_motion_feature_group for deterministic monotonic cross-frame candidate grouping, sequence.stitch_unassociated_single_frame_points for review-gated constant-velocity singleton recovery, sequence.link_motion_evidence_tracks for one-to-one point-motion/exposure-line evidence de-duplication;    high-level multi-frame orchestration and full motion-feature/trail extraction remain pending; affine RANSAC uses deterministic SplitMix64 rather than NumPy PCG64",
             "native_photometry_execution":"instrumental/apparent/legacy absolute arithmetic; structured parallax/model-distance/catalog absolute estimates; derived error/interval/strict gates; weighted zero point/color fit; clipping/holdout/catalog quality gates/sample roles; calibration application and error; native JSONL operations",
             "native_source_photometry":"complete source table status/flags/reference/model magnitude separation; faintest ranking, ties, calibration consistency and partial selection scope; JSONL photometry.table/photometry.faintest",
             "native_frame_pipeline":"direct FITS single-frame analysis through detection/matching/calibration/source rows/faintest; native library recalibration without repeating detection; JSONL pipeline.analyze",
             "native_cli_contracts":{"rst19":"rst19-native executable; synthetic/default modes, stdout/json-out/errors and bounded original FITS verified; desktop routing not switched","rst19-aux-audit":"single FITS or sorted directory input; CSV/JSON output matches Python; cross-platform packaging and broader parser edges pending"},
             "verification":["tests/parity.rs","tests/measurement_parity.rs","tests/detection_parity.rs","tests/spatial_parity.rs","tests/float32_parity.rs","tests/assignment_parity.rs","tests/matching_parity.rs","tests/photometry_parity.rs","tests/calibration_parity.rs","tests/source_photometry_parity.rs","tests/pipeline_parity.rs","tests/wcs_sequence_parity.rs","tests/sequence_fast_parity.rs","tests/sequence_consensus_parity.rs","tests/sequence_temporal_reference_parity.rs","tests/sequence_reference_parity.rs","tests/sequence_trail_metrics_parity.rs","tests/sequence_trail_shape_parity.rs","tests/sequence_motion_feature_gates_parity.rs","tests/sequence_motion_feature_continuity_parity.rs","tests/sequence_motion_feature_exposure_parity.rs","tests/sequence_motion_feature_confirmation_parity.rs","tests/sequence_motion_proposal_thresholds_parity.rs","tests/sequence_motion_time_axis_parity.rs","tests/sequence_observation_time_axis_parity.rs","tests/sequence_motion_feature_pair_geometry_parity.rs","tests/collinear_motion_geometry_parity.rs","tests/collinear_motion_group_parity.rs","tests/single_frame_stitch_parity.rs","tests/motion_evidence_link_parity.rs","tests/attitude_parity.rs","verify_fits.py","verify_detection.py","verify_float32.py","verify_assignment.py","verify_matching.py","verify_photometry.py","verify_pipeline.py","verify_wcs_sequence.py","verify_sequence_fast.py","verify_sequence_consensus.py","verify_sequence_temporal_reference.py","verify_sequence_reference.py","verify_sequence_trail_metrics.py","verify_sequence_trail_shape.py","verify_sequence_motion_feature_gates.py","verify_sequence_motion_feature_continuity.py","verify_sequence_motion_feature_exposure.py","verify_sequence_motion_feature_confirmation.py","verify_sequence_motion_proposal_thresholds.py","verify_sequence_motion_time_axis.py","verify_sequence_observation_time_axis.py","verify_sequence_motion_feature_pair_geometry.py","verify_collinear_motion_geometry.py","verify_collinear_motion_group.py","verify_single_frame_stitch.py","verify_motion_evidence_link.py","verify_attitude.py","verify_spatial_ties.py","verify_full_snapshot.py"],
             "remaining":["full float32 detection acceptance coverage","blind plate solving and remaining astrometric workflows",
                          "sequence motion-feature/trail extraction, high-level multi-frame orchestration, raw fast-candidate point workflows; exact NumPy PCG64 RANSAC sampling parity",
                          "all scientific audits and experiments","remaining CLI input/output contracts and exhaustive parser edge coverage","native desktop routing",
                          "remove Python from runtime packaging","full scientific and cross-platform acceptance"]}
    target=ROOT/"backend/migration.json"
    target.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{len(modules)} Python backend modules, {len(tools)} tools, {sum(len(m['symbols']) for m in modules)} symbols inventoried")
