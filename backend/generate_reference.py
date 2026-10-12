"""Capture Python baseline outputs for Rust unit tests. Never used by the native runtime."""
from __future__ import annotations
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from rst19.fits import _parse_card_value, exposure_seconds, exposure_milliseconds, decode_auxiliary
from rst19.detection import sigma_clipped_stats, _resize_grid, local_background_rms
from rst19.attitude import quaternion_to_active_matrix, vector_to_radec_deg, auxiliary_boresight_radec
from rst19.models import AuxiliaryData
from rst19.wcs import TangentPlaneWCS
from rst19.detection import _masked_gaussian_response, _masked_starlet_smooth, _response_proposals, _merge_proposals, _linear_artifact_mask, _Proposal
from scipy import ndimage


def generate() -> dict:
    rng = np.random.default_rng(19)
    statistics = []
    for count in [1, 2, 31, 64, 1001]:
        for mode in ["constant", "normal", "outliers"]:
            values = np.full(count, 7.0) if mode == "constant" else rng.normal(1200, 15, count)
            if mode == "outliers":
                values[::9] += 1000
            mask = rng.random(count) < 0.1
            mask[0] = False
            for sample_limit in [11, 100_000]:
                center, noise = sigma_clipped_stats(values, mask=mask, sample_limit=sample_limit)
                statistics.append({"values": values.tolist(), "mask": mask.tolist(), "sample_limit": sample_limit,
                                   "center": center, "noise": noise})
    grids = []
    for gh, gw, h, w in [(1, 1, 7, 5), (1, 3, 4, 5), (3, 1, 5, 4), (3, 4, 13, 17), (2, 3, 1, 1), (4, 4, 2, 2)]:
        grid = rng.normal(0, 4, (gh, gw))
        grids.append({"shape": [gh, gw], "target": [h, w], "values": grid.ravel().tolist(),
                      "expected": _resize_grid(grid, (h, w)).ravel().tolist()})
    backgrounds = []
    for h, w in [(16, 16), (31, 35), (33, 49)]:
        values = rng.normal(1000, 15, (h, w))
        values[8:10, 8:10] += 500
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :min(104, w)] = True
        b, n = local_background_rms(values, mask, box_size=16)
        backgrounds.append({"shape": [h, w], "values": values.ravel().tolist(), "mask": mask.ravel().tolist(),
                            "background": b.ravel().tolist(), "noise": n.ravel().tolist()})
    projections = []
    for ra0, dec0, rotation, parity in [(359.8, 72, 0, 1), (0.1, -34, 61, -1), (150, 0, -123, 1)]:
        wcs = TangentPlaneWCS(center_ra_deg=ra0, center_dec_deg=dec0, pixel_scale_arcsec=8.5,
                              crpix_x=2047.5, crpix_y=2047.5, rotation_deg=rotation, parity=parity)
        sky = [(ra0, dec0), (ra0 + 0.4, dec0 + 0.1), (ra0 - 0.2, dec0 - 0.15)]
        pixels = [(2047.5, 2047.5), (0, 0), (4095, 4095), (1540.2, 72.6)]
        projections.append({"wcs": wcs.as_dict(), "world": sky,
                            "world_to_pixel": [wcs.world_to_pixel(*p) for p in sky],
                            "tangent": [wcs.world_to_tangent_arcsec(*p) for p in sky],
                            "pixels": pixels, "pixel_to_world": [wcs.pixel_to_world(*p) for p in pixels]})
    quaternions = []
    for q in [(0, 0, 0, 1), (1, 2, 3, 4), (0.15, -0.33, 0.84, 0.37)]:
        v = [0.0] * 26
        v[10:14] = q
        aux = AuxiliaryData.from_values(v)
        quaternions.append({"q": q, "matrix": quaternion_to_active_matrix(q).tolist(),
                            "boresight": auxiliary_boresight_radec(aux)})
    exposures = []
    for header in [{}, {"EXPTIME": 1.5, "EXPOSURE": 1500}, {"EXPOSURE": 1500},
                   {"EXPOSURE": "1.5 s"}, {"EXPOSURE": 1.5, "TIMEUNIT": "s"},
                   {"EXPTIME": "1.5D3 msec"}, {"EXPTIME": False, "EXPOSURE": "3000"},
                   {"EXPTIME": -1, "EXPOSURE": "2 unknown"}, {"EXPTIME": "bad", "EXPOSURE": 500},
                   {"EXPOSURE": "2s", "EXPOSURE_UNIT": "ms"}, {"EXPTIME": 0}]:
        exposures.append({"header": header, "seconds": exposure_seconds(header), "milliseconds": exposure_milliseconds(header)})
    cards = [{"input": s, "expected": _parse_card_value(s)} for s in ["", "T", "F", "01500", "1.25D3", "0000          01500", "0270.00000 0000.00000", "'star''s field'", "'value / ignored'", "uninterpreted", " 23 / comment"]]
    filters=[]
    for density in [0.003,0.15]:
        image=rng.normal(0,3,(31,35))
        image[10:13,4:29]+=80
        image[22,18]+=100
        valid=rng.random(image.shape)>=density
        image[~valid]=0
        for fast in [False,True]:
            gaussian,gaussian_mode=_masked_gaussian_response(image,valid,sigma=1.3,calculation_dtype=np.float64,fast_sequence=fast)
            starlet,starlet_mode=_masked_starlet_smooth(image,valid,step=2,calculation_dtype=np.float64,fast_sequence=fast)
            filters.append({"shape":list(image.shape),"values":image.ravel().tolist(),"valid":valid.ravel().tolist(),"fast":fast,
                            "gaussian":gaussian.ravel().tolist(),"gaussian_mode":gaussian_mode,
                            "starlet":starlet.ravel().tolist(),"starlet_mode":starlet_mode,
                            "maximum":ndimage.maximum_filter(image,size=7,mode="nearest").ravel().tolist(),
                            "line_mask":_linear_artifact_mask(image,np.ones(image.shape)*3,valid,detection_sigma=4,psf_fwhm=3).ravel().tolist()})
    proposals=[]
    for mode in ["boundary_pair","plateau","edge_peaks"]:
        response=np.zeros((32,32),dtype=float)
        if mode=="boundary_pair":response[16,10],response[16,14]=20,18
        elif mode=="plateau":response[16,10:18]=20
        else:response[0,0],response[31,31],response[20,10]=50,40,20
        valid=np.ones(response.shape,dtype=bool)
        for distance in [0,3,4,8]:
            output=_response_proposals(response,valid,threshold_sigma=4,min_distance=distance,scale_fwhm=3,method="gaussian")
            proposals.append({"shape":[32,32],"values":response.ravel().tolist(),"distance":distance,"expected":[asdict(p) for p in output]})
    primary=[_Proposal(10,10,20.0,"gaussian",3.0),_Proposal(30,22,18.0,"gaussian",3.0)]
    extras=[_Proposal(12,11,30.0,"dog_narrow",2.5),_Proposal(30,23,16.0,"dog_broad",4.0),_Proposal(49,27,15.0,"starlet_s1",1.7),_Proposal(50,27,15.0,"starlet_s2",3.5)]
    merged={"primary":[asdict(p) for p in primary],"extras":[asdict(p) for p in extras],"expected":[asdict(p) for p in _merge_proposals(primary,extras,min_distance=4)]}
    modules = ["fits", "models", "detection", "attitude", "wcs"]
    return {"source_sha256": {name: hashlib.sha256((ROOT / f"src/rst19/{name}.py").read_bytes()).hexdigest() for name in modules},
            "statistics": statistics, "grids": grids, "backgrounds": backgrounds, "projections": projections,
            "quaternions": quaternions, "exposures": exposures, "cards": cards,"filters":filters,"proposals":proposals,"merged":merged}


if __name__ == "__main__":
    target = ROOT / "backend/tests/fixtures/reference.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(generate(), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(target)
