"""Read-only per-member reproduction of testzip with bounded memory and evidence."""
import argparse
import json
from pathlib import Path
import platform
import sys
import traceback
import zipfile
import zlib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-members',type=int,default=256)
    args=p.parse_args()
    archive=Path(args.archive)
    report=Path(args.out)
    with report.open('x',encoding='utf-8') as log,archive.open('rb') as raw,zipfile.ZipFile(raw) as zf:
        log.write(json.dumps(dict(python=sys.version,platform=platform.platform(),archive=str(archive),
                                  archive_size=archive.stat().st_size))+'\n');log.flush()
        for ordinal,info in enumerate(zf.infolist()[:args.max_members]):
            record=dict(ordinal=ordinal,member=info.filename,header_offset=info.header_offset,
                        compressed_bytes=info.compress_size,expected_bytes=info.file_size)
            size,crc=0,0
            try:
                with zf.open(info.filename,'r') as member:
                    while True:
                        data=member.read(1024**2)
                        if not data: break
                        size+=len(data);crc=zlib.crc32(data,crc)
                if size!=info.file_size or crc&0xffffffff!=info.CRC: raise ValueError('Explicit size/CRC mismatch')
                record.update(status='CRC_OK',bytes=size,crc32=f'{crc&0xffffffff:08x}')
            except Exception as error:
                record.update(status='ERROR',bytes=size,error=repr(error),traceback=traceback.format_exc())
                with archive.open('rb',buffering=0) as direct:
                    probes=[]
                    for offset in (info.header_offset,2**31,2**32,2**33,archive.stat().st_size-32):
                        try:
                            direct.seek(offset);data=direct.read(32)
                            probes.append(dict(offset=offset,status='READ_OK',bytes=len(data),signature=data[:4].hex()))
                        except Exception as probe_error: probes.append(dict(offset=offset,error=repr(probe_error)))
                    record['unbuffered_offset_probes']=probes
                log.write(json.dumps(record)+'\n');log.flush()
                print(json.dumps(record,indent=2),flush=True)
                return
            log.write(json.dumps(record)+'\n');log.flush()
            if (ordinal+1)%500==0: print(f'CRC OK: {ordinal+1}; last={info.filename}',flush=True)
        print(f'No failure reproduced in {min(args.max_members,len(zf.infolist()))} entries',flush=True)


if __name__=='__main__':main()
