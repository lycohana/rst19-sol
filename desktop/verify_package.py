"""Exercise a packaged worker with no external Python on its search path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys


def make_fits(path: Path) -> None:
    def card(key: str, value: str) -> bytes:
        return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")
    header = b"".join([card("SIMPLE", "T"), card("BITPIX", "16"), card("NAXIS", "2"),
                       card("NAXIS1", "128"), card("NAXIS2", "128"), b"END".ljust(80)]).ljust(2880, b" ")
    rng = random.Random(19)
    pixels = []
    for y in range(128):
        for x in range(128):
            value = rng.gauss(1000, 8)
            for px, py in [(30, 40), (70, 80), (98, 50)]:
                value += 1000 * math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * 1.4 ** 2))
            pixels.append(round(value))
    path.write_bytes(header + struct.pack(f">{len(pixels)}h", *pixels))


def make_feature_catalog(path: Path) -> None:
    path.write_bytes(
        b"\xef\xbb\xbf"
        + b"detection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,filter_snr,fwhm,ellipticity,sharpness,psf_support_pixels,footprint_pixels,centroid_shift_px,peak\n"
        + b"1,compact,Compact,true,wing|blend|wing,12,10,2.5,0.1,0.3,5,20,0.1,100\n"
        + b"2,compact,Compact,false,blend,11,9,2.5,0.1,0.3,4,18,0.2,90\n"
        + b"3,diffuse,Diffuse,yes,,3,2,0.5,0.8,0.9,2,1,0.9,20\n"
    )


def make_pair_catalog(path: Path) -> None:
    path.write_bytes(
        b"\xef\xbb\xbf"
        + b"detection_id,x,y,peak_x,peak_y,quality_passed,feature_class,flags\n"
        + b'2,2.738,0,4,0,false,crowded_blend,"UNRESOLVED_BLEND,MANUAL"\n'
        + b"4,25,0,25,0,true,compact_quality,\n"
        + b'1,0,0,0,0,true,range_anomaly,"CODE_PATTERN,MANUAL"\n'
        + b"3,-0.5,0.5,3.5,0,true,crowded_blend,\n"
    )


def make_topology_fixture(frame_path: Path, catalog_path: Path) -> None:
    def card(key: str, value: str) -> bytes:
        return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")

    values = [10] * (40 * 40)
    for x, y, value in ((20, 20, 130), (21, 20, 99), (19, 21, 80),
                        (18, 21, 70), (17, 21, 50), (16, 21, 40)):
        values[y * 40 + x] = value
    header = b"".join((
        card("SIMPLE", "T"), card("BITPIX", "16"), card("NAXIS", "2"),
        card("NAXIS1", "40"), card("NAXIS2", "40"), b"END".ljust(80),
    )).ljust(2880, b" ")
    image = struct.pack(f">{len(values)}h", *values)
    image = image.ljust(((len(image) + 2879) // 2880) * 2880, b"\0")
    frame_path.write_bytes(header + image)
    catalog_path.write_text(
        "detection_id,x,y,peak,flux,background,noise,snr,quality_passed,peak_x,peak_y,repeated_code_values\n"
        "1,21,20,99,100,10,2,20,false,21,20,99\n"
        "2,16,21,40,100,10,2,20,false,16,21,99\n",
        encoding="utf-8",
    )


def make_aperture_peak_ratio_catalog(path: Path) -> None:
    path.write_bytes(
        b"\xef\xbb\xbf"
        + b"detection_id,feature_class,peak,flux,proposal_methods\r\n"
        + b'82931,range_anomaly,4,24,"gaussian|dog_narrow"\r\n'
        + b"82932,range_anomaly,10,20,gaussian\r\n"
        + b"82934,crowded_blend,2,50,gaussian\r\n"
        + b"82935,crowded_blend,5,5,dog_narrow\r\n"
        + b'82936,complete_set,4,48,"dog_broad|gaussian|dog_narrow|gaussian"\r\n'
        + b'82937,dog_only,3,45,"dog_narrow|dog_broad"\r\n'
    )


def make_feature_correlation_catalog(path: Path) -> None:
    payload = (
        "\ufefffeature_class,feature_class_label,a,b,c\r\n"
        'weak,"弱,背景",4,4,1\r\n'
        "weak,ignored,5,5,2\r\n"
        "weak,ignored,6,6,3\r\n"
        'weak,"ignored,again",,8,4\r\n'
        'compact_quality,"紧凑,清晰",1,3,10\r\n'
        "compact_quality,ignored,2,2,20\r\n"
        "compact_quality,ignored,3,1,30\r\n"
        ",,7,7,NaN\r\n"
    )
    path.write_bytes(payload.encode("utf-8"))


def make_feature_transition_flow_csv(path: Path) -> None:
    payload = (
        "\ufefflayer,anchor_feature_class,anchor_feature_class_label,response_feature_class,"
        "response_feature_class_label,anchor_count,frame_count,possible_match_count,matched_count\r\n"
        'candidate,compact_quality,"紧凑,清晰",compact_quality,紧凑,4,3,12,3\r\n'
        "candidate,compact_quality,ignored,weak_or_background,背景,4,3,12,2\r\n"
        "candidate,compact_quality,ignored,linear_artifact,线伪迹,4,3,12,2\r\n"
        "quality,masked_or_edge,边缘,masked_or_edge,边缘,5,3,15,0\r\n"
        "quality,other_rejected,拒绝,other_rejected,拒绝,0,0,0,0\r\n"
        ",,,,,,,,\r\n"
    )
    path.write_bytes(payload.encode("utf-8"))


def make_feature_gate_route_catalog(path: Path) -> None:
    payload = (
        "\ufeffdetection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,"
        "psf_support_pixels,fwhm,ellipticity,sharpness,footprint_pixels\r\n"
        '1,compact_quality,"紧凑,清晰",True,,12,5,2.5,0.1,0.3,20\r\n'
        "2,weak_or_background,弱,False,LOW_FLUX_SNR,3,5,2.5,0.1,0.3,20\r\n"
        "3,linear_artifact,线状,False,LINE_ARTIFACT,20,5,2.5,0.1,0.3,20\r\n"
        "4,crowded_blend,拥挤,False,LOW_FLUX_SNR|UNRESOLVED_BLEND,3,2,2.5,0.1,0.3,20\r\n"
    )
    path.write_bytes(payload.encode("utf-8"))


def make_feature_hard_negative_catalog(path: Path) -> None:
    payload = (
        "\ufeffdetection_id,feature_class,feature_class_label,quality_passed,flags,flux_snr,filter_snr,"
        "peak,fwhm,ellipticity,sharpness,psf_support_pixels,footprint_pixels\r\n"
        "1,compact_quality,紧凑,True,,2,4,20,2,0.1,0.3,5,20\r\n"
        "2,compact_quality,紧凑,False,LOW_FLUX_SNR,12,24,120,2,0.1,0.3,5,20\r\n"
        "3,compact_quality,紧凑,False,UNRESOLVED_BLEND,20,40,200,2,0.1,0.3,5,20\r\n"
        "4,other_rejected,其他,False,OTHER,100,200,1000,2,0.1,0.3,5,20\r\n"
    )
    path.write_bytes(payload.encode("utf-8"))


def make_feature_evidence_axes_matrix(path: Path) -> None:
    payload = (
        "\ufefffeature_class,feature_class_label,candidate_count,candidate_presence_ge_required_count,"
        "quality_response_ge_required_count,spatial_local_correlation_median,spatial_local_residual_median\r\n"
        'compact_quality,"紧凑,清晰",100,95,85,0.84,0.54\r\n'
        "crowded_blend,拥挤,100,60,1,0.17,0.98\r\n"
        "masked_or_edge,边缘,100,80,20,0.67,0.74\r\n"
        "linear_artifact,线伪迹,10,2,1,0.40,0.91\r\n"
        "other_rejected,其它,0,0,0,,\r\n"
    )
    path.write_bytes(payload.encode("utf-8"))


def make_direct_neighbor_catalog(path: Path) -> None:
    payload = (
        "\ufeffdetection_id,x,y,peak_x,peak_y,peak,flux_snr,filter_snr,fwhm,"
        "deblend_delta_bic,deblend_component_snr,quality_passed,feature_class,flags\r\n"
        "1,0,0,0,0,3992,204.6,1195,2.59,-4.09,0.87,False,range_anomaly,CODE_PATTERN\r\n"
        "2,2.738,0,4,0,569,115.5,300,2.44,-4.09,0.87,False,crowded_blend,UNRESOLVED_BLEND\r\n"
        "3,10,0,10,0,250,12,20,2,,,True,compact_quality,\r\n"
        "4,20,0,20,0,900,20,100,2,12,6,True,compact_quality,\r\n"
        "5,23,0,23,0,400,10,60,2,,,True,compact_quality,\r\n"
    )
    path.write_bytes(payload.encode("utf-8"))


def make_feature_cross_axis_inputs(out: Path) -> tuple[Path, Path, Path, Path]:
    files = {
        "catalog": (
            "detection_id,feature_class,feature_class_label,quality_passed\r\n"
            "1,compact_quality,紧凑,True\r\n2,crowded_blend,拥挤,False\r\n"
        ),
        "raw": (
            "detection_id,core_fraction_median,noise_median,frames_support_ge3,frames_flux_snr_ge5,"
            "frames_negative_extreme_count_gt0,frames_repeated_code_count_gt0\r\n"
            "1,0.7,8,15,15,0,0\r\n2,0.2,14,0,2,1,1\r\n"
        ),
        "peak": (
            "detection_id,feature_class,feature_class_label,quality_passed,raw_valid,"
            "raw_local_maximum_r1,raw_local_max_gap_adu_r1\r\n"
            "1,compact_quality,紧凑,True,True,True,0\r\n"
            "2,crowded_blend,拥挤,False,True,False,100\r\n"
        ),
        "temporal": (
            "detection_id,feature_class,expected_frame_count,frame_count,coverage_fraction,"
            "flux_snr_relative_mad,flux_snr_positive_fraction,flux_snr_ge5_fraction,"
            "flux_snr_sign_flip_count\r\n"
            "1,compact_quality,15,15,1,0.1,1,1,0\r\n"
            "2,crowded_blend,15,10,0.6667,0.3,0.8,0.1,2\r\n"
        ),
    }
    paths = []
    for name, contents in files.items():
        path = out / f"cross-axis {name}.csv"
        path.write_bytes(("\ufeff" + contents).encode("utf-8"))
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


def make_gaia_merge_inputs(out: Path) -> tuple[Path, Path]:
    first = out / "gaia calibration layer.csv"
    second = out / "gaia deep layer.csv"
    first.write_bytes(
        "\ufeffsource_id,ra_deg,dec_deg,magnitude,color\n"
        "1,129.5,-1.8,6.2,0.4\n"
        "2,129.6,-1.7,7.1,0.8\n".encode("utf-8")
    )
    second.write_bytes(
        "source_id,ra_deg,dec_deg,magnitude,color,parallax_mas\n"
        "2,129.6,-1.7,7.1,0.8,2.3\n"
        "3,129.7,-1.6,15.4,1.1,1.7\n".encode("utf-8")
    )
    for path in (first, second):
        rows = 2
        audit = {
            "catalog": "Gaia DR3",
            "complete": True,
            "center_ra_deg": 129.5,
            "center_dec_deg": -1.8,
            "search_radius_deg": 6.92,
            "min_g_mag": 5.0,
            "max_g_mag": 18.0,
            "csv_row_count": rows,
            "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
    return first, second


def verify(package: Path, out: Path) -> dict:
    package = package.resolve()
    out.mkdir(parents=True, exist_ok=False)
    worker = package / "runtime" / ("rst19-worker.exe" if os.name == "nt" else "rst19-worker")
    native_worker = package / "runtime" / ("rst19-native-worker.exe" if os.name == "nt" else "rst19-native-worker")
    assert native_worker.is_file(), "Rust single-frame worker is missing from the package"
    environment = {**os.environ, "PYTHONPATH": "", "PYTHONHOME": "", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    environment["PATH"] = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32") if os.name == "nt" else "/usr/bin:/bin"
    catalog = subprocess.run([str(worker), "--catalog"], capture_output=True, text=True, encoding="utf-8",
                             env=environment, cwd=out, timeout=60, check=True)
    entry = json.loads(catalog.stdout)
    assert entry["ok"] and entry["payload"]["environment"]["frozen"]
    assert len(entry["payload"]["tools"]) == 78
    frame = out / "原始数据 with spaces.fits"
    make_fits(frame)
    original = hashlib.sha256(frame.read_bytes()).hexdigest()
    request = {"operation": "tool", "tool": "rst19", "args": [str(frame.resolve())], "out_dir": str((out / "job").resolve())}
    response = subprocess.run([str(native_worker)], input=json.dumps(request, ensure_ascii=False), capture_output=True,
                              text=True, encoding="utf-8", env=environment, cwd=out, timeout=60, check=True)
    events = [json.loads(line) for line in response.stdout.splitlines()]
    terminal = [event for event in events if event["event"] == "done"]
    assert len(terminal) == 1 and terminal[0]["ok"]
    payload = terminal[0]["payload"]
    assert payload["result"]["detection"]["quality_count"] >= 3
    assert Path(payload["preview"]).is_file()
    attitude_out = out / "attitude-job"
    attitude_request = {
        "operation": "tool",
        "tool": "rst19-aux-audit",
        "args": [str(frame.resolve()), "--out-dir", str(attitude_out.resolve())],
        "out_dir": str(attitude_out.resolve()),
    }
    attitude_response = subprocess.run(
        [str(native_worker)], input=json.dumps(attitude_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    attitude_events = [json.loads(line) for line in attitude_response.stdout.splitlines()]
    assert len(attitude_events) == 1 and attitude_events[0]["ok"]
    attitude = attitude_events[0]["payload"]
    assert attitude["tool"] == "rst19-aux-audit"
    assert attitude["result"]["summary"]["frame_count"] == 1
    assert attitude["result"]["summary"]["missing_auxiliary_count"] == 1
    assert Path(attitude["result_path"]).is_file()
    assert (attitude_out / "auxiliary_boresight_audit.csv").is_file()
    catalog = out / "source catalog.csv"
    make_feature_catalog(catalog)
    profile_out = out / "feature-flag-profile"
    profile_request = {
        "operation": "tool",
        "tool": "rst19-feature-flag-profile",
        "args": [
            str(catalog.resolve()), "--target-ids", "3", "1", "3", "missing",
            "--out-dir", str(profile_out.resolve()),
        ],
        "out_dir": str((out / "feature-flag-job").resolve()),
    }
    profile_response = subprocess.run(
        [str(native_worker)], input=json.dumps(profile_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    profile_events = [json.loads(line) for line in profile_response.stdout.splitlines()]
    assert len(profile_events) == 1 and profile_events[0]["ok"]
    profile = profile_events[0]["payload"]
    assert profile["tool"] == "rst19-feature-flag-profile"
    assert profile["result"]["targets"] == ["3", "1", "missing"]
    assert profile["result"]["rows"][0]["flag"] == "blend"
    assert profile["result"]["rows"][0]["flag_count"] == 2
    assert Path(profile["result_path"]).is_file()
    assert (profile_out / "feature_flag_profile.csv").is_file()
    assert (profile_out / "feature_flag_profile_targets.csv").is_file()
    interaction_out = out / "feature-flag-interaction"
    interaction_request = {
        "operation": "tool",
        "tool": "rst19-feature-flag-interaction",
        "args": [
            str(catalog.resolve()), "--high-snr-threshold", "10", "--top-n", "2",
            "--out-dir", str(interaction_out.resolve()),
        ],
        "out_dir": str((out / "feature-flag-interaction-job").resolve()),
    }
    interaction_response = subprocess.run(
        [str(native_worker)], input=json.dumps(interaction_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    interaction_events = [json.loads(line) for line in interaction_response.stdout.splitlines()]
    assert len(interaction_events) == 1 and interaction_events[0]["ok"]
    interaction = interaction_events[0]["payload"]
    assert interaction["tool"] == "rst19-feature-flag-interaction"
    assert interaction["result"]["record_count"] == 3
    assert interaction["result"]["rows"][0]["interaction"] == "blend"
    assert interaction["result"]["rows"][0]["candidate_count"] == 2
    assert Path(interaction["result_path"]).is_file()
    assert (interaction_out / "feature_flag_interactions.csv").is_file()
    margin_out = out / "feature-gate-margin"
    margin_request = {
        "operation": "tool",
        "tool": "rst19-feature-gate-margin",
        "args": [str(catalog.resolve()), "--out-dir", str(margin_out.resolve())],
        "out_dir": str((out / "feature-gate-margin-job").resolve()),
    }
    margin_response = subprocess.run(
        [str(native_worker)], input=json.dumps(margin_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    margin_events = [json.loads(line) for line in margin_response.stdout.splitlines()]
    assert len(margin_events) == 1 and margin_events[0]["ok"]
    margin = margin_events[0]["payload"]
    assert margin["tool"] == "rst19-feature-gate-margin"
    assert margin["result"]["rows"][0]["feature_class"] == "compact"
    assert margin["result"]["rows"][0]["metric"] == "ellipticity"
    assert margin["result"]["rows"][0]["class_count"] == 2
    assert Path(margin["result_path"]).is_file()
    assert (margin_out / "feature_gate_margins.csv").is_file()
    effect_out = out / "feature-effect-size"
    effect_request = {
        "operation": "tool",
        "tool": "rst19-feature-effect-size",
        "args": [
            str(catalog.resolve()), "--reference-class", "compact",
            "--comparison-classes", "diffuse", "--features", "flux_snr", "fwhm",
            "--top-n", "2", "--out-dir", str(effect_out.resolve()),
        ],
        "out_dir": str((out / "feature-effect-size-job").resolve()),
    }
    effect_response = subprocess.run(
        [str(native_worker)], input=json.dumps(effect_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    effect_events = [json.loads(line) for line in effect_response.stdout.splitlines()]
    assert len(effect_events) == 1 and effect_events[0]["ok"]
    effect = effect_events[0]["payload"]
    assert effect["tool"] == "rst19-feature-effect-size"
    assert effect["result"]["comparison_classes"] == ["diffuse"]
    assert effect["result"]["top_features_by_class"]["diffuse"] == ["flux_snr", "fwhm"]
    assert Path(effect["result_path"]).is_file()
    assert (effect_out / "feature_effect_sizes.csv").is_file()
    context_out = out / "feature-class-context"
    context_request = {
        "operation": "tool",
        "tool": "rst19-feature-class-context",
        "args": [
            str(catalog.resolve()), "--target-id", "2", "--target-id", "1",
            "--target-id", "2", "--out-dir", str(context_out.resolve()),
        ],
        "out_dir": str((out / "feature-class-context-job").resolve()),
    }
    context_response = subprocess.run(
        [str(native_worker)], input=json.dumps(context_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    context_events = [json.loads(line) for line in context_response.stdout.splitlines()]
    assert len(context_events) == 1 and context_events[0]["ok"]
    context = context_events[0]["payload"]
    assert context["tool"] == "rst19-feature-class-context"
    assert [row["detection_id"] for row in context["result"]["targets"]] == [2, 1]
    assert context["result"]["targets"][0]["class_quality_count"] == 1
    assert len(context["result"]["metrics"]) == 18
    assert Path(context["result_path"]).is_file()
    assert (context_out / "feature_class_context_targets.csv").is_file()
    assert (context_out / "feature_class_context_metrics.csv").is_file()
    pair_catalog = out / "pair source catalog.csv"
    make_pair_catalog(pair_catalog)
    pair_context_out = out / "pair-mechanism-context"
    pair_context_request = {
        "operation": "tool",
        "tool": "rst19-pair-mechanism-context",
        "args": [
            str(pair_catalog.resolve()), "--target-id", "2", "--target-id", "1",
            "--target-id", "2", "--out-dir", str(pair_context_out.resolve()),
        ],
        "out_dir": str((out / "pair-mechanism-context-job").resolve()),
    }
    pair_context_response = subprocess.run(
        [str(native_worker)], input=json.dumps(pair_context_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    pair_context_events = [json.loads(line) for line in pair_context_response.stdout.splitlines()]
    assert len(pair_context_events) == 1 and pair_context_events[0]["ok"]
    pair_context = pair_context_events[0]["payload"]
    assert pair_context["tool"] == "rst19-pair-mechanism-context"
    assert pair_context["result"]["source_count"] == 4
    assert len(pair_context["result"]["pairs"]) == 2
    assert pair_context["result"]["target"]["in_geometry_context"] is True
    assert Path(pair_context["result_path"]).is_file()
    assert (pair_context_out / "pair_mechanism_context_pairs.csv").is_file()
    pair_sensitivity_out = out / "pair-mechanism-sensitivity"
    pair_sensitivity_request = {
        "operation": "tool",
        "tool": "rst19-pair-mechanism-sensitivity",
        "args": [
            str(pair_catalog.resolve()), "--target-id", "2", "--target-id", "1",
            "--configuration", "narrow,1,3.5,4.8",
            "--configuration", "target,3.5,3.5,4.8",
            "--out-dir", str(pair_sensitivity_out.resolve()),
        ],
        "out_dir": str((out / "pair-mechanism-sensitivity-job").resolve()),
    }
    pair_sensitivity_response = subprocess.run(
        [str(native_worker)], input=json.dumps(pair_sensitivity_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    pair_sensitivity_events = [json.loads(line) for line in pair_sensitivity_response.stdout.splitlines()]
    assert len(pair_sensitivity_events) == 1 and pair_sensitivity_events[0]["ok"]
    pair_sensitivity = pair_sensitivity_events[0]["payload"]
    assert pair_sensitivity["tool"] == "rst19-pair-mechanism-sensitivity"
    assert [row["configuration"] for row in pair_sensitivity["result"]["rows"]] == ["narrow", "target"]
    assert [row["target_in_geometry_context"] for row in pair_sensitivity["result"]["rows"]] == [False, True]
    assert Path(pair_sensitivity["result_path"]).is_file()
    assert (pair_sensitivity_out / "pair_mechanism_sensitivity.csv").is_file()
    topology_frame = out / "近邻原始像素.fits"
    topology_catalog = out / "topology source catalog.csv"
    make_topology_fixture(topology_frame, topology_catalog)
    topology_out = out / "pair-pixel-topology"
    topology_request = {
        "operation": "tool",
        "tool": "rst19-pair-pixel-topology",
        "args": [
            str(topology_frame.resolve()), "--source-catalog", str(topology_catalog.resolve()),
            "--primary-id", "1", "--secondary-id", "2", "--patch-padding-px", "4",
            "--sigma-levels", "3", "5",
            "--raw-maximum-neighborhoods-px", "3",
            "--out-dir", str(topology_out.resolve()),
        ],
        "out_dir": str((out / "pair-pixel-topology-job").resolve()),
    }
    topology_response = subprocess.run(
        [str(native_worker)], input=json.dumps(topology_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    topology_events = [json.loads(line) for line in topology_response.stdout.splitlines()]
    assert len(topology_events) == 1 and topology_events[0]["ok"]
    topology = topology_events[0]["payload"]
    assert topology["tool"] == "rst19-pair-pixel-topology"
    assert topology["result"]["repeated_code_values"] == [99]
    assert [12, 16, 26, 26] == topology["result"]["patch_bounds"]
    assert len(topology["result"]["component_rows"]) == 4
    for artifact in (
        "pair_pixel_topology_components.csv", "pair_pixel_topology_maxima.csv",
        "pair_pixel_topology.json", "pair_pixel_topology.svg",
    ):
        assert (topology_out / artifact).is_file(), artifact
    ratio_catalog = out / "aperture peak ratio source catalog.csv"
    make_aperture_peak_ratio_catalog(ratio_catalog)
    ratio_out = out / "aperture-peak-ratio"
    ratio_request = {
        "operation": "tool",
        "tool": "rst19-aperture-peak-ratio",
        "args": [
            str(ratio_catalog.resolve()), "--target-ids", "82934", "82931", "82934",
            "--out-dir", str(ratio_out.resolve()),
        ],
        "out_dir": str((out / "aperture-peak-ratio-job").resolve()),
    }
    ratio_response = subprocess.run(
        [str(native_worker)], input=json.dumps(ratio_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    ratio_events = [json.loads(line) for line in ratio_response.stdout.splitlines()]
    assert len(ratio_events) == 1 and ratio_events[0]["ok"]
    ratio = ratio_events[0]["payload"]
    assert ratio["tool"] == "rst19-aperture-peak-ratio"
    assert len(ratio["result"]["summaries"]) == 4
    assert len(ratio["result"]["target_rows"]) == 2
    assert {row["method_group"] for row in ratio["result"]["method_summaries"]} == {
        "all_three", "dog_only", "gaussian_only", "partial",
    }
    assert Path(ratio["result_path"]).is_file()
    for artifact in (
        "aperture_peak_ratio_summary.csv", "aperture_peak_ratio_method_summary.csv",
        "aperture_peak_ratio_targets.csv",
    ):
        assert (ratio_out / artifact).is_file(), artifact
    correlation_catalog = out / "feature correlation source catalog.csv"
    make_feature_correlation_catalog(correlation_catalog)
    correlation_out = out / "feature-correlation"
    correlation_request = {
        "operation": "tool",
        "tool": "rst19-feature-correlation",
        "args": [
            str(correlation_catalog.resolve()), "--features", "a", "b", "c",
            "--strong-threshold", "0.75", "--out-dir", str(correlation_out.resolve()),
        ],
        "out_dir": str((out / "feature-correlation-job").resolve()),
    }
    correlation_response = subprocess.run(
        [str(native_worker)], input=json.dumps(correlation_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    correlation_events = [json.loads(line) for line in correlation_response.stdout.splitlines()]
    assert len(correlation_events) == 1 and correlation_events[0]["ok"]
    correlation = correlation_events[0]["payload"]
    assert correlation["tool"] == "rst19-feature-correlation"
    assert len(correlation["result"]["correlations"]) == 3
    assert [row["sample_count"] for row in correlation["result"]["correlations"]] == [7, 6, 7]
    assert [row["feature_class"] for row in correlation["result"]["class_medians"]] == [
        "weak", "compact_quality",
    ]
    for artifact in (
        "feature_correlations.csv", "feature_class_medians.csv",
        "feature_class_correlations.csv", "feature_correlation_audit.json",
    ):
        assert (correlation_out / artifact).is_file(), artifact
    transition_source = out / "sequence feature transitions.csv"
    make_feature_transition_flow_csv(transition_source)
    transition_out = out / "feature-transition-flow"
    transition_request = {
        "operation": "tool",
        "tool": "rst19-feature-transition-flow",
        "args": [str(transition_source.resolve()), "--out-dir", str(transition_out.resolve())],
        "out_dir": str((out / "feature-transition-flow-job").resolve()),
    }
    transition_response = subprocess.run(
        [str(native_worker)], input=json.dumps(transition_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    transition_events = [json.loads(line) for line in transition_response.stdout.splitlines()]
    assert len(transition_events) == 1 and transition_events[0]["ok"]
    transition = transition_events[0]["payload"]
    assert transition["tool"] == "rst19-feature-transition-flow"
    assert transition["result"]["included_group_count"] == 2
    assert transition["result"]["skipped_zero_anchor_group_count"] == 1
    assert transition["result"]["rows"][0]["top_non_same_feature_class"] == "linear_artifact"
    assert transition["result"]["rows"][1]["q_any"] == 0.0
    assert transition["result"]["rows"][1]["q_same"] is None
    assert Path(transition["result_path"]).is_file()
    for artifact in ("sequence_feature_transition_flow.csv", "sequence_feature_transition_flow.json"):
        assert (transition_out / artifact).is_file(), artifact
    route_catalog = out / "feature gate route source catalog.csv"
    make_feature_gate_route_catalog(route_catalog)
    route_out = out / "feature-gate-route"
    route_request = {
        "operation": "tool",
        "tool": "rst19-feature-gate-route",
        "args": [str(route_catalog.resolve()), "--out-dir", str(route_out.resolve())],
        "out_dir": str((out / "feature-gate-route-job").resolve()),
    }
    route_response = subprocess.run(
        [str(native_worker)], input=json.dumps(route_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    route_events = [json.loads(line) for line in route_response.stdout.splitlines()]
    assert len(route_events) == 1 and route_events[0]["ok"]
    gate_route = route_events[0]["payload"]
    assert gate_route["tool"] == "rst19-feature-gate-route"
    assert len(gate_route["result"]["source_rows"]) == 4
    assert gate_route["result"]["source_rows"][1]["route"] == "rejected_numeric_only"
    assert gate_route["result"]["source_rows"][3]["structural_flags"] == "UNRESOLVED_BLEND"
    assert Path(gate_route["result_path"]).is_file()
    for artifact in (
        "feature_gate_route_sources.csv",
        "feature_gate_route_summary.csv",
    ):
        assert (route_out / artifact).is_file(), artifact
    hard_negative_catalog = out / "feature hard negative source catalog.csv"
    make_feature_hard_negative_catalog(hard_negative_catalog)
    hard_negative_out = out / "feature-hard-negative"
    hard_negative_request = {
        "operation": "tool",
        "tool": "rst19-feature-hard-negative",
        "args": [
            str(hard_negative_catalog.resolve()), "--top-n", "1",
            "--out-dir", str(hard_negative_out.resolve()),
        ],
        "out_dir": str((out / "feature-hard-negative-job").resolve()),
    }
    hard_negative_response = subprocess.run(
        [str(native_worker)], input=json.dumps(hard_negative_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    hard_negative_events = [json.loads(line) for line in hard_negative_response.stdout.splitlines()]
    assert len(hard_negative_events) == 1 and hard_negative_events[0]["ok"]
    hard_negative = hard_negative_events[0]["payload"]
    assert hard_negative["tool"] == "rst19-feature-hard-negative"
    assert [row["detection_id"] for row in hard_negative["result"]["rows"]] == [3]
    assert [row["feature_class"] for row in hard_negative["result"]["summaries"]] == ["compact_quality"]
    assert Path(hard_negative["result_path"]).is_file()
    for artifact in (
        "feature_hard_negative_summary.csv", "feature_hard_negative_rows.csv",
    ):
        assert (hard_negative_out / artifact).is_file(), artifact
    gaia_merge_inputs = make_gaia_merge_inputs(out)
    gaia_output = out / "gaia-merge" / "merged catalog.csv"
    gaia_job = out / "gaia-merge-job"
    gaia_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in gaia_merge_inputs
    }
    gaia_request = {
        "operation": "tool",
        "tool": "rst19-gaia-merge",
        "args": [*(str(path.resolve()) for path in gaia_merge_inputs), "--out", str(gaia_output.resolve())],
        "out_dir": str(gaia_job.resolve()),
    }
    gaia_response = subprocess.run(
        [str(native_worker)], input=json.dumps(gaia_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    gaia_events = [json.loads(line) for line in gaia_response.stdout.splitlines()]
    assert len(gaia_events) == 1 and gaia_events[0]["ok"]
    gaia_merge = gaia_events[0]["payload"]
    assert gaia_merge["tool"] == "rst19-gaia-merge"
    assert gaia_merge["result"]["complete"] is True
    assert gaia_merge["result"]["row_count"] == 3
    assert gaia_merge["result"]["duplicate_count"] == 1
    assert Path(gaia_merge["result_path"]).is_file()
    gaia_sidecar = gaia_output.with_suffix(gaia_output.suffix + ".meta.json")
    assert gaia_sidecar.is_file()
    assert json.loads(gaia_sidecar.read_text(encoding="utf-8"))["csv_row_count"] == 3
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == gaia_hashes[str(path)]
        for path in gaia_merge_inputs
    )
    gaia_cli = package / "runtime" / (
        "rst19-gaia-merge.exe" if os.name == "nt" else "rst19-gaia-merge"
    )
    assert gaia_cli.is_file(), "Native Gaia merge CLI is missing from the package"
    help_response = subprocess.run(
        [str(gaia_cli), "--help"], capture_output=True, text=True, encoding="utf-8",
        env=environment, cwd=out, timeout=60,
    )
    assert help_response.returncode == 0 and "用法:" in help_response.stdout
    invalid_response = subprocess.run(
        [str(gaia_cli), str(gaia_merge_inputs[0]), "--out", str((out / "unused.csv").resolve())],
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60,
    )
    assert invalid_response.returncode == 2, invalid_response.stderr
    cli_output = out / "gaia-cli" / "merged.csv"
    cli_response = subprocess.run(
        [str(gaia_cli), *(str(path.resolve()) for path in gaia_merge_inputs),
         "--out", str(cli_output.resolve())],
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60,
    )
    assert cli_response.returncode == 0, cli_response.stderr
    assert json.loads(
        cli_output.with_suffix(cli_output.suffix + ".meta.json").read_text(encoding="utf-8")
    )["complete"] is True
    incomplete_dir = out / "gaia-cli-incomplete"
    incomplete_dir.mkdir()
    incomplete_inputs = make_gaia_merge_inputs(incomplete_dir)
    second_audit = incomplete_inputs[1].with_suffix(
        incomplete_inputs[1].suffix + ".meta.json"
    )
    audit_payload = json.loads(second_audit.read_text(encoding="utf-8"))
    audit_payload["complete"] = False
    second_audit.write_text(json.dumps(audit_payload), encoding="utf-8")
    incomplete_output = incomplete_dir / "merged.csv"
    incomplete_response = subprocess.run(
        [str(gaia_cli), *(str(path.resolve()) for path in incomplete_inputs),
         "--out", str(incomplete_output.resolve())],
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60,
    )
    assert incomplete_response.returncode == 3, incomplete_response.stderr
    assert json.loads(
        incomplete_output.with_suffix(incomplete_output.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )["complete"] is False
    axes_matrix = out / "feature evidence axes matrix.csv"
    make_feature_evidence_axes_matrix(axes_matrix)
    axes_out = out / "feature-evidence-axes"
    axes_request = {
        "operation": "tool",
        "tool": "rst19-feature-evidence-axes",
        "args": [str(axes_matrix.resolve()), "--out-dir", str(axes_out.resolve())],
        "out_dir": str((out / "feature-evidence-axes-job").resolve()),
    }
    axes_response = subprocess.run(
        [str(native_worker)], input=json.dumps(axes_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    axes_events = [json.loads(line) for line in axes_response.stdout.splitlines()]
    assert len(axes_events) == 1 and axes_events[0]["ok"]
    axes_payload = axes_events[0]["payload"]
    assert axes_payload["tool"] == "rst19-feature-evidence-axes"
    assert len(axes_payload["result"]["rows"]) == 5
    assert len(axes_payload["result"]["sensitivity_rows"]) == 25
    assert axes_payload["result"]["rows"][0]["evidence_axis_pattern"] == "aligned_quality_psf"
    assert axes_payload["result"]["rows"][1]["evidence_axis_pattern"] == "persistent_without_quality"
    for artifact in (
        "feature_evidence_axes.csv",
        "feature_evidence_axes_sensitivity.csv",
        "feature_evidence_axes.json",
    ):
        assert (axes_out / artifact).is_file(), artifact
    neighbor_catalog = out / "direct neighbor source catalog.csv"
    make_direct_neighbor_catalog(neighbor_catalog)
    neighbor_out = out / "direct-neighbor-audit"
    neighbor_request = {
        "operation": "tool",
        "tool": "rst19-direct-neighbor-audit",
        "args": [
            str(neighbor_catalog.resolve()),
            "--target-id", "1",
            "--target-id", "2",
            "--pair-radius", "4",
            "--out-dir", str(neighbor_out.resolve()),
        ],
        "out_dir": str((out / "direct-neighbor-audit-job").resolve()),
    }
    neighbor_response = subprocess.run(
        [str(native_worker)], input=json.dumps(neighbor_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    neighbor_events = [json.loads(line) for line in neighbor_response.stdout.splitlines()]
    assert len(neighbor_events) == 1 and neighbor_events[0]["ok"]
    neighbor_payload = neighbor_events[0]["payload"]
    assert neighbor_payload["tool"] == "rst19-direct-neighbor-audit"
    assert neighbor_payload["result"]["direct_pair_count"] == 2
    assert neighbor_payload["result"]["unresolved_pair_count"] == 1
    assert neighbor_payload["result"]["independent_pair_candidate_count"] == 1
    assert neighbor_payload["result"]["target"]["classification"] == "unresolved_pair"
    for artifact in (
        "direct_neighbor_pairs.csv",
        "direct_neighbor_summaries.csv",
        "direct_neighbor_targets.csv",
        "direct_neighbor_audit.json",
    ):
        assert (neighbor_out / artifact).is_file(), artifact
    cross_axis_inputs = make_feature_cross_axis_inputs(out)
    cross_axis_out = out / "feature-cross-axis-audit"
    cross_axis_request = {
        "operation": "tool",
        "tool": "rst19-feature-cross-axis-audit",
        "args": [*(str(path.resolve()) for path in cross_axis_inputs), "--out-dir", str(cross_axis_out.resolve())],
        "out_dir": str((out / "feature-cross-axis-audit-job").resolve()),
    }
    cross_axis_response = subprocess.run(
        [str(native_worker)], input=json.dumps(cross_axis_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    cross_axis_events = [json.loads(line) for line in cross_axis_response.stdout.splitlines()]
    assert len(cross_axis_events) == 1 and cross_axis_events[0]["ok"], cross_axis_events
    cross_axis_payload = cross_axis_events[0]["payload"]
    assert cross_axis_payload["tool"] == "rst19-feature-cross-axis-audit"
    assert len(cross_axis_payload["result"]["source_rows"]) == 2
    assert cross_axis_payload["result"]["source_rows"][1]["evidence_pattern"] == (
        "raw_non_local_peak+value_domain_anomaly+no_support_seen"
    )
    for artifact in (
        "feature_cross_axis_sources.csv", "feature_cross_axis_summary.csv",
        "feature_cross_axis_patterns.csv", "feature_cross_axis_audit.json",
    ):
        assert (cross_axis_out / artifact).is_file(), artifact
    source_group_catalog = out / "source group source catalog.csv"
    make_direct_neighbor_catalog(source_group_catalog)
    source_group_out = out / "source-group-audit"
    source_group_request = {
        "operation": "tool",
        "tool": "rst19-source-group-audit",
        "args": [
            str(source_group_catalog.resolve()),
            "--target-id", "1",
            "--target-id", "2",
            "--out-dir", str(source_group_out.resolve()),
        ],
        "out_dir": str((out / "source-group-audit-job").resolve()),
    }
    source_group_response = subprocess.run(
        [str(native_worker)], input=json.dumps(source_group_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    source_group_events = [json.loads(line) for line in source_group_response.stdout.splitlines()]
    assert len(source_group_events) == 1 and source_group_events[0]["ok"], source_group_events
    source_group_payload = source_group_events[0]["payload"]
    assert source_group_payload["tool"] == "rst19-source-group-audit"
    assert source_group_payload["result"]["source_count"] == 5
    assert source_group_payload["result"]["group_count"] == 3
    assert source_group_payload["result"]["multi_member_group_count"] == 2
    assert source_group_payload["result"]["unresolved_group_count"] == 1
    assert source_group_payload["result"]["targets"][0]["classification"] == "unresolved_group"
    for artifact in (
        "source_groups.csv", "source_group_class_summary.csv",
        "source_group_targets.csv", "source_group_summary.json",
    ):
        assert (source_group_out / artifact).is_file(), artifact
    feedback_path = out / "manual threshold feedback.jsonl"
    feedback_record = {
        "timestamp_utc": "2026-10-13T00:00:00+00:00",
        "frame_path": "F01.fits",
        "threshold_sigma": 3.5,
        "min_flux_snr": 3.0,
        "min_distance": 4,
        "psf_fwhm": 3.0,
        "candidate_count": 12,
        "quality_count": 8,
        "returned_count": 12,
        "flag_counts": {"LOW_FLUX_SNR": 4},
        "judgement": "保留弱星",
        "note": "package worker test",
        "source": "native_manual",
        "schema_version": 1,
    }
    feedback_append_request = {
        "operation": "feedback.append",
        "path": str(feedback_path.resolve()),
        "feedback": feedback_record,
    }
    feedback_append_response = subprocess.run(
        [str(native_worker)], input=json.dumps(feedback_append_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    feedback_append_events = [json.loads(line) for line in feedback_append_response.stdout.splitlines()]
    assert len(feedback_append_events) == 1 and feedback_append_events[0]["ok"]
    feedback_load_response = subprocess.run(
        [str(native_worker)],
        input=json.dumps({"operation": "feedback.load", "path": str(feedback_path.resolve())}, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    feedback_load_events = [json.loads(line) for line in feedback_load_response.stdout.splitlines()]
    assert len(feedback_load_events) == 1 and feedback_load_events[0]["ok"]
    assert feedback_load_events[0]["payload"]["records"] == [feedback_record]
    preview_request = {
        "operation": "preview",
        "path": str(frame.resolve()),
        "mode": "noise",
        "out_dir": str((out / "preview-job").resolve()),
    }
    preview_response = subprocess.run(
        [str(native_worker)], input=json.dumps(preview_request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", env=environment,
        cwd=out, timeout=60, check=True,
    )
    preview_events = [json.loads(line) for line in preview_response.stdout.splitlines()]
    assert len(preview_events) == 1 and preview_events[0]["ok"]
    assert Path(preview_events[0]["payload"]["preview"]).is_file()
    assert preview_events[0]["payload"]["width"] == 128
    assert hashlib.sha256(frame.read_bytes()).hexdigest() == original
    checksums = json.loads((package / "SHA256.json").read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        assert hashlib.sha256((package / relative).read_bytes()).hexdigest() == expected, relative
    report = {"package": str(package), "environment": entry["payload"]["environment"], "tool_count": 78,
              "native_backends": [
                  "rst19 single-frame", "rst19-aux-audit", "rst19-feature-flag-profile",
                  "rst19-feature-flag-interaction", "rst19-feature-gate-margin",
                  "rst19-feature-effect-size",
                  "rst19-feature-class-context",
                  "rst19-pair-mechanism-context",
                  "rst19-pair-mechanism-sensitivity",
                  "rst19-pair-pixel-topology",
                  "rst19-aperture-peak-ratio",
                  "rst19-feature-correlation",
                  "rst19-feature-transition-flow",
                  "rst19-feature-gate-route",
                  "rst19-feature-hard-negative",
                  "rst19-gaia-merge",
                  "rst19-gaia-merge native CLI",
                  "rst19-feature-evidence-axes",
                  "rst19-direct-neighbor-audit",
                  "rst19-feature-cross-axis-audit",
                  "rst19-source-group-audit",
                  "manual threshold feedback JSONL",
              ],
              "candidate_count": payload["result"]["detection"]["candidate_count"],
              "quality_count": payload["result"]["detection"]["quality_count"], "raw_fits_unchanged": True,
              "external_python_search_disabled": True, "checksum_files": len(checksums)}
    report["catalog_merge"] = {
        "complete": gaia_merge["result"]["complete"],
        "row_count": gaia_merge["result"]["row_count"],
        "duplicate_count": gaia_merge["result"]["duplicate_count"],
        "sidecar_created": gaia_sidecar.is_file(),
        "inputs_unchanged": True,
        "native_cli_complete_exit_code": cli_response.returncode,
        "native_cli_incomplete_exit_code": incomplete_response.returncode,
        "native_cli_input_error_exit_code": invalid_response.returncode,
        "native_cli_help_exit_code": help_response.returncode,
        "native_cli_runs_without_python": True,
    }
    report["feature_evidence_axes"] = {
        "row_count": len(axes_payload["result"]["rows"]),
        "sensitivity_row_count": len(axes_payload["result"]["sensitivity_rows"]),
        "artifacts_written": True,
    }
    report["direct_neighbor_audit"] = {
        "source_count": neighbor_payload["result"]["source_count"],
        "direct_pair_count": neighbor_payload["result"]["direct_pair_count"],
        "unresolved_pair_count": neighbor_payload["result"]["unresolved_pair_count"],
        "independent_pair_candidate_count": neighbor_payload["result"]["independent_pair_candidate_count"],
        "artifacts_written": True,
    }
    report["feature_cross_axis_audit"] = {
        "source_count": len(cross_axis_payload["result"]["source_rows"]),
        "class_count": len(cross_axis_payload["result"]["class_summaries"]),
        "pattern_count": len(cross_axis_payload["result"]["pattern_summaries"]),
        "artifacts_written": True,
        "external_python_search_disabled": True,
    }
    report["source_group_audit"] = {
        "source_count": source_group_payload["result"]["source_count"],
        "group_count": source_group_payload["result"]["group_count"],
        "multi_member_group_count": source_group_payload["result"]["multi_member_group_count"],
        "artifacts_written": True,
        "external_python_search_disabled": True,
    }
    report["manual_threshold_feedback"] = {
        "append_and_load_verified": True,
        "record_count": len(feedback_load_events[0]["payload"]["records"]),
        "external_python_search_disabled": True,
    }
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.out), ensure_ascii=False, indent=2))
