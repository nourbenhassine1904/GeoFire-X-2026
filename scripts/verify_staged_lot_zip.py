"""Verify an existing ZIP with bounded sequential reads; never rebuild or upload."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile
import zlib


def verify(canonical,staged,evidence):
    canonical,staged,evidence=Path(canonical),Path(staged),Path(evidence)
    lot=json.loads(canonical.read_text())
    identity=hashlib.sha256(json.dumps({k:v for k,v in lot.items() if k!='identity'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if identity!=lot['identity'] or lot['number'] not in range(2,9):raise ValueError('Pinned lot identity/scope invalid')
    payload=staged/'payload.zip'
    if not payload.is_file() or payload.is_symlink():raise ValueError('Expected existing regular ZIP')
    initial=payload.stat()
    evidence.mkdir(parents=True,exist_ok=False)
    expected={'rasters/'+f['path']:f for f in lot['files']}
    checked=0
    with zipfile.ZipFile(payload) as archive,(evidence/'verified.jsonl').open('x',encoding='utf-8') as log:
        infos=archive.infolist()
        names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or set(names)!=set(expected)|{'lot-inventory.json'}:
            raise ValueError('ZIP central directory inventory differs')
        if any(i.flag_bits&1 or stat.S_ISLNK(i.external_attr>>16) or i.is_dir() for i in infos):
            raise ValueError('Encrypted/symlink/directory member refused')
        inventory=archive.getinfo('lot-inventory.json')
        if inventory.file_size>1_000_000 or json.loads(archive.read(inventory))!=lot:
            raise ValueError('ZIP inventory differs from canonical manifest')
        for number,entry in enumerate(lot['files'],1):
            info=archive.getinfo('rasters/'+entry['path'])
            if info.file_size!=entry['bytes'] or info.CRC!=int(entry['crc32'],16):
                raise ValueError('ZIP member declared size/CRC differs: '+entry['path'])
            sha,crc,n=hashlib.sha256(),0,0
            with archive.open(info) as source:
                for block in iter(lambda:source.read(1024**2),b''):
                    sha.update(block);crc=zlib.crc32(block,crc);n+=len(block)
            actual=dict(path=entry['path'],bytes=n,sha256=sha.hexdigest(),crc32=f'{crc&0xffffffff:08x}')
            if actual!=entry:raise ValueError('ZIP member SHA256/CRC32/size differs: '+entry['path'])
            log.write(json.dumps(actual)+'\n');log.flush();checked+=n
            if number%100==0 or number==len(lot['files']):
                (evidence/'progress.json').write_text(json.dumps(dict(files=number,bytes=checked)))
                print(f'Existing ZIP: {number}/{len(lot["files"])} members verified',flush=True)
    archive_sha=hashlib.sha256()
    with payload.open('rb') as source:
        for block in iter(lambda:source.read(1024**2),b''):archive_sha.update(block)
    final=payload.stat()
    if (initial.st_dev,initial.st_ino,initial.st_size,initial.st_mtime_ns)!=(final.st_dev,final.st_ino,final.st_size,final.st_mtime_ns):
        raise ValueError('Existing ZIP changed during verification')
    receipt=dict(lot_identity=identity,handle=lot['handle'],payload_bytes=final.st_size,
        sha256=archive_sha.hexdigest(),observations=lot['observations'])
    previous=staged/'staging-receipt.json'
    if previous.exists():
        if json.loads(previous.read_text())!=receipt:raise ValueError('Existing staging receipt differs; preserved')
    else:
        with previous.open('x',encoding='utf-8') as output:json.dump(receipt,output,indent=2)
    proof=dict(status='EXISTING_ZIP_FULLY_VERIFIED_REUSED',**receipt,verified_files=len(expected),
        verified_raw_bytes=checked,zip_structure=True,all_crc32=True,all_sha256=True,
        rebuilt=False,network_bytes=0)
    with (evidence/'verification.json').open('x',encoding='utf-8') as output:json.dump(proof,output,indent=2)
    print(json.dumps(proof,indent=2),flush=True)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical',required=True);p.add_argument('--staged',required=True);p.add_argument('--evidence',required=True)
    a=p.parse_args();verify(a.canonical,a.staged,a.evidence)
