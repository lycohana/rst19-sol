"""Compare the native FITS reader to Python, including exact i64 samples and real files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from rst19.fits import read_fits, exposure_seconds, exposure_milliseconds


def card(key:str,value:str)->bytes:
    return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")


def write_fits(path:Path,bitpix:int,values:list,extra:list[bytes]|None=None,aux:bool=False)->None:
    width,height=16,16
    header=b"".join([card("SIMPLE","T"),card("BITPIX",str(bitpix)),card("NAXIS","2"),
                     card("NAXIS1",str(width)),card("NAXIS2",str(height)),*(extra or []),b"END".ljust(80)]).ljust(2880,b" ")
    dtype={8:">u1",16:">i2",32:">i4",64:">i8",-32:">f4",-64:">f8"}[bitpix]
    data=bytearray(np.resize(np.array(values,dtype=dtype),width*height).tobytes())
    if aux:
        fields=[0.0]*26
        fields[:4]=[270.0,0.0,178.25,-33.5]
        fields[13]=1.0
        fields[4:10]=[1.0,2.0,3.0,4.0,5.0,6.0]
        raw=bytearray(struct.pack("<26d",*fields))
        for i in range(0,len(raw),2):raw[i],raw[i+1]=raw[i+1],raw[i]
        data[:208]=raw
    path.write_bytes(header+data)


def native(engine:Path,request:dict)->tuple[int,dict]:
    response=subprocess.run([str(engine)],input=json.dumps(request,ensure_ascii=False),capture_output=True,text=True,encoding="utf-8",timeout=30)
    lines=response.stdout.splitlines()
    assert len(lines)==1,response.stdout
    return response.returncode,json.loads(lines[0])


def compare(engine:Path,path:Path,scaling:bool)->None:
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    reference=read_fits(path,apply_scaling=scaling)
    rng=np.random.default_rng(19)
    indices=sorted(set([0,1,103,104,reference.width-1,reference.data.size-1,*rng.integers(0,reference.data.size,size=64).tolist()]))
    code,event=native(engine,{"operation":"fits.inspect","path":str(path.resolve()),"apply_scaling":scaling,"indices":indices})
    assert code==0 and event["ok"],event
    actual=event["payload"]
    assert actual["header"]==dict(reference.header),(path,"header")
    assert actual["dtype"]==str(reference.data.dtype),(path,actual["dtype"],str(reference.data.dtype))
    assert actual["image_shape"]==list(reference.data.shape)
    assert actual["data_offset"]==reference.data_offset
    assert actual["auxiliary"]==(reference.auxiliary.as_dict() if reference.auxiliary else None),(path,"auxiliary")
    expected=reference.data.ravel()[indices].tolist()
    assert actual["samples"]==expected,(path,"pixels")
    assert actual["exposure_seconds"]==exposure_seconds(reference.header)
    assert actual["exposure_ms"]==exposure_milliseconds(reference.header)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def verify(engine:Path,out:Path,data_dir:Path|None)->dict:
    out.mkdir(parents=True,exist_ok=False)
    synthetic=[]
    for bitpix,values in [(8,[0,1,255]),(16,[-32768,-1,0,32767]),(32,[-2147483648,1,2147483647]),
                           (64,[-9223372036854775808,9007199254740993,9223372036854775807]),(-32,[-1.5,0,1.25,1e20]),(-64,[-1.5,1e100,2.125])]:
        path=out/f"像素 with spaces {bitpix}.fits"
        write_fits(path,bitpix,values,[card("BSCALE","2.5"),card("BZERO","11"),card("EXPOSURE","0000          01500")])
        compare(engine,path,False)
        compare(engine,path,True)
        synthetic.append(str(path))
    auxiliary=out/"auxiliary.fits"
    write_fits(auxiliary,16,[1000,1001,999],[card("AZIMUTH","270"),card("ELEVATIO","0")],aux=True)
    compare(engine,auxiliary,False)
    mismatched=out/"not-auxiliary.fits"
    write_fits(mismatched,16,[1000,1001,999],[card("AZIMUTH","180"),card("ELEVATIO","0")],aux=True)
    compare(engine,mismatched,False)
    bad_cases=[]
    for name,raw in [("short-header",b"SIMPLE"),("short-data",auxiliary.read_bytes()[:2900]),
                     ("no-simple",auxiliary.read_bytes().replace(b"                   T",b"                   F",1))]:
        path=out/f"{name}.fits";path.write_bytes(raw)
        try:read_fits(path)
        except ValueError:pass
        else:raise AssertionError(f"baseline accepted {name}")
        code,event=native(engine,{"operation":"fits.inspect","path":str(path.resolve())})
        assert code!=0 and event["ok"] is False,(name,event)
        bad_cases.append(name)
    code,event=native(engine,{"operation":"fits.inspect","path":str((out/"missing.fits").resolve())})
    assert code!=0 and event["ok"] is False
    real=[]
    if data_dir:
        for path in sorted(data_dir.glob("*.fits")):
            compare(engine,path,False)
            real.append({"name":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    report={"engine":str(engine.resolve()),"synthetic_storage_types":6,"scaling_checks":6,"auxiliary_cases":2,
            "invalid_cases":len(bad_cases)+1,"real_files":real,"raw_i64_precision_preserved":True}
    (out/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("engine",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.engine.resolve(),args.out,args.data_dir),ensure_ascii=False,indent=2))
