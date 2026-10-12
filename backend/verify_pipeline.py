"""Verify native source rows, direct-FITS pipeline and the original CLI contract."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).parent))
from verify_detection import compare,ROOT
from verify_fits import card
import numpy as np
from rst19.pipeline import analyze_frame
from rst19.catalog import CatalogSource
from rst19.wcs import TangentPlaneWCS
from rst19 import cli as python_cli

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("engine",type=Path);parser.add_argument("cli",type=Path);parser.add_argument("--out",type=Path,required=True);parser.add_argument("--real-frame",type=Path);args=parser.parse_args();count=0;cli_count=0
    def native(request,expected):
        global count
        result=subprocess.run([str(args.engine.resolve())],input=json.dumps(request,ensure_ascii=False),capture_output=True,text=True,encoding="utf-8",timeout=60);event=json.loads(result.stdout);assert result.returncode==0 and event["ok"],event;compare(event["payload"],expected);count+=1
    source_file=Path(__file__).parent/"tests/fixtures/source_photometry.json";source=json.loads(source_file.read_text(encoding="utf-8"));assert source["source_sha256"]==hashlib.sha256((ROOT/"src/rst19/photometry.py").read_bytes()).hexdigest()
    for kind,key in [("table","tables"),("faintest","faintest")]:
        for case in source[key]:native({"operation":f"photometry.{kind}",**{k:v for k,v in case.items() if k not in {"expected","name"}}},case["expected"])
    pipeline_file=Path(__file__).parent/"tests/fixtures/pipeline.json";fixture=json.loads(pipeline_file.read_text(encoding="utf-8"))
    for name,digest in fixture["source_hashes"].items():assert hashlib.sha256((ROOT/"src/rst19"/name).read_bytes()).hexdigest()==digest,"Python baseline changed"
    records=[]
    with tempfile.TemporaryDirectory() as directory:
        directory=Path(directory);paths=[]
        for index,frame in enumerate(fixture["frames"]):
            path=directory/f"{index}.fits";bitpix=frame["header"]["BITPIX"]
            headers=[card("SIMPLE","T"),card("BITPIX",str(bitpix)),card("NAXIS","2"),card("NAXIS1",str(frame["width"])),card("NAXIS2",str(frame["height"])),card("EXPTIME","2.0"),card("DATE-OBS","'2026-03-30T16:32:05'"),b"END".ljust(80)]
            path.write_bytes(b"".join(headers).ljust(2880,b" ")+np.asarray(frame["image"],dtype=">f8" if bitpix==-64 else ">i2").tobytes());paths.append(path)
        for i,case in enumerate(fixture["cases"]):
            path=paths[case["frame"]];before=hashlib.sha256(path.read_bytes()).hexdigest();catalog=None if case["catalog"] is None else [CatalogSource(**s) for s in case["catalog"]];wcs=None if case["wcs"] is None else TangentPlaneWCS(**case["wcs"])
            expected=analyze_frame(path,catalog=catalog,wcs=wcs,**case["options"]).as_dict();native({"operation":"pipeline.analyze","path":str(path),"catalog":case["catalog"],"wcs":case["wcs"],"options":case["options"]},expected)
            # The original CLI does not expose float32/fast_sequence flags.
            if "use_float32" not in case["options"]:
                argv=[str(path)];out=directory/f"out-{i}.json"
                if catalog is not None:
                    csv_path=directory/f"catalog-{i}.csv"
                    with csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
                        columns=list(case["catalog"][0]) if case["catalog"] else ["source_id","ra_deg","dec_deg"]
                        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(case["catalog"])
                    argv+=["--catalog",str(csv_path),"--center-ra",str(wcs.center_ra_deg),"--center-dec",str(wcs.center_dec_deg),"--pixel-scale-arcsec",str(wcs.pixel_scale_arcsec),"--crpix-x",str(wcs.crpix_x),"--crpix-y",str(wcs.crpix_y),"--rotation-deg",str(wcs.rotation_deg),"--parity",str(wcs.parity)]
                for name,value in case["options"].items():
                    flag="--"+name.replace("_","-");argv+=([flag] if value is True else [flag,str(value)])
                assert python_cli.main(argv+["--json-out",str(out)])==0;reference=json.loads(out.read_text(encoding="utf-8"))
                result=subprocess.run([str(args.cli.resolve()),*argv],capture_output=True,text=True,encoding="utf-8",timeout=60);assert result.returncode==0,result.stderr;compare(json.loads(result.stdout),reference);cli_count+=1
                actual_file=directory/"nested"/f"native-{i}.json";result=subprocess.run([str(args.cli.resolve()),*argv,"--json-out",str(actual_file)],capture_output=True,text=True,encoding="utf-8",timeout=60);assert result.returncode==0 and result.stdout=="",result.stderr;compare(json.loads(actual_file.read_text(encoding="utf-8")),reference);cli_count+=1
            assert hashlib.sha256(path.read_bytes()).hexdigest()==before,"FITS changed"
            records.append({"name":case["name"],"fits_sha256_unchanged":before,"candidate_count":expected["detection"]["candidate_count"],"quality_count":expected["detection"]["quality_count"],"all_fields_compared":True})
        for bad in [[],[str(paths[0]),"--proposal-mode","invalid"],[str(paths[0]),"--center-ra","10"],[str(paths[0]),"--unknown-option"],[str(paths[0]),"--photometric-color-order","4"]]:
            result=subprocess.run([str(args.cli.resolve()),*bad],capture_output=True,text=True,encoding="utf-8",timeout=60);assert result.returncode==2,result;cli_count+=1
    real_records=[]
    if args.real_frame:
        before=hashlib.sha256(args.real_frame.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            for mode in ["gaussian","hybrid","ensemble"]:
                output=Path(directory)/f"{mode}.json";argv=[str(args.real_frame.resolve()),"--proposal-mode",mode,"--max-sources","300"]
                assert python_cli.main(argv+["--json-out",str(output)])==0;reference=json.loads(output.read_text(encoding="utf-8"))
                result=subprocess.run([str(args.cli.resolve()),*argv],capture_output=True,text=True,encoding="utf-8",timeout=300);assert result.returncode==0,result.stderr;compare(json.loads(result.stdout),reference);cli_count+=1
                native({"operation":"pipeline.analyze","path":str(args.real_frame.resolve()),"options":{"proposal_mode":mode,"max_sources":300}},reference)
                real_records.append({"mode":mode,"candidate_count":reference["detection"]["candidate_count"],"returned_count":reference["detection"]["returned_count"],"quality_count":reference["detection"]["quality_count"],"all_fields_compared":True,"fits_sha256_unchanged":before,"scope":"full image with explicit max_sources=300; no catalog"})
        assert hashlib.sha256(args.real_frame.read_bytes()).hexdigest()==before,"original FITS changed"
    report={"engine_sha256":hashlib.sha256(args.engine.read_bytes()).hexdigest(),"cli_sha256":hashlib.sha256(args.cli.read_bytes()).hexdigest(),"fixture_hashes":{"source_photometry":hashlib.sha256(source_file.read_bytes()).hexdigest(),"pipeline":hashlib.sha256(pipeline_file.read_bytes()).hexdigest()},"python_source_hashes":fixture["source_hashes"],"source_table_cases":len(source["tables"]),"faintest_cases":len(source["faintest"]),"native_requests_verified":count,"cli_requests_verified":cli_count,"records":records,"real_records":real_records,"scope":fixture["scope"]+"; direct synthetic/optional original FITS and original single-frame CLI, no real-catalog identity or desktop/all-tools claim"}
    args.out.mkdir(parents=True,exist_ok=False);(args.out/"verification.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))
