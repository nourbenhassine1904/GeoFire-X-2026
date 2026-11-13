"""Freeze stopped acquisition metadata; raw files stay in place, read-only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from geofirex.acquisition_session import session_lock
from geofirex.corpus import json_new,sha256_file,verify_member


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    session=Path('artifacts/scientific-acquisition/stage080-v1')
    output=Path(args.out)
    with session_lock(session):
        output.mkdir(parents=True,exist_ok=False)
        # Prevent an accidental later relaunch; this does not alter the ledger.
        marker=session/'STOP'
        if not marker.exists(): marker.write_text('Stopped by user; no automatic restart.\n',encoding='utf-8')
        backup=output/'session_backup';backup.mkdir()
        for source in session.iterdir():
            if source.is_file() and source.name!='session.lock': shutil.copy2(source,backup/source.name)
        index=json.loads(Path('data/manifests/real/corpus_zip_index_v1.json').read_text())
        entries={e['path']:e for e in index['entries']}
        roots=[Path('data/raw/WildfireSpreadTS')/name for name in ('real_sample','corpus_v1','scientific_corpus_v1')]
        records,parts=[],[]
        for root in roots:
            for path in sorted(root.rglob('*')):
                if not path.is_file(): continue
                relative=path.relative_to(root).as_posix()
                if path.suffix=='.tif':
                    entry=entries[relative]
                    records.append(dict(path=relative,location=str(path.resolve()),root=root.name,
                        bytes=path.stat().st_size,crc_verified=verify_member(path,entry),sha256=sha256_file(path)))
                elif path.suffix=='.part':
                    key=relative[:relative.index('.tif')+4] if '.tif' in relative else None
                    parts.append(dict(location=str(path.resolve()),bytes=path.stat().st_size,
                                      sha256=sha256_file(path),expected_path=key,
                                      full_crc_matches=verify_member(path,entries[key]) if key in entries else None))
        valid={r['path'] for r in records if r['crc_verified']}
        grouped=defaultdict(set)
        for entry in entries.values(): grouped[entry['event_id']].add(entry['path'])
        complete=sorted(event for event,keys in grouped.items() if keys<=valid)
        ledger=[json.loads(line) for line in (session/'transfer.jsonl').read_text().splitlines()]
        last=ledger[-1]
        plan=json.loads(Path('reports/scientific_scaleup_v4/acquisition_80.json').read_text())
        remaining=sum(e['compressed_bytes'] for e in entries.values() if e['path'] not in valid)
        result=dict(status='STOPPED_NO_AUTOMATIC_RESTART',new_tiffs=sum(r['root']=='scientific_corpus_v1' for r in records),
                    total_tiffs=len(records),all_crc_verified=all(r['crc_verified'] for r in records),
                    complete_events=complete,complete_event_count=len(complete),partial_files=parts,
                    stage80_missing_files=sum(e['path'] not in valid for e in plan['entries']),
                    confirmed_http_body_bytes=last['confirmed_body_bytes'],reserved_http_bytes=last['reserved_bytes'],
                    requests=last['requests'],unused_old_cap_bytes=6611516695-last['reserved_bytes'],
                    original_stage_cap_bytes=6611516695,range_full_remaining_payload_bytes=remaining,
                    range_full_cap_proposal_bytes=(remaining*115+99)//100+8*1024**2,
                    source_zip_bytes=index['archive_bytes'],extracted_bytes=sum(e['geotiff_bytes'] for e in entries.values()),
                    C_free_bytes=shutil.disk_usage('.').free,D_free_bytes=shutil.disk_usage('D:/').free,
                    files=records,session_backup_sha256={p.name:sha256_file(p) for p in backup.iterdir() if p.is_file()},
                    stop_note='Console interrupt; process ended and lock released. No terminal attempt report was emitted; this independent snapshot preserves the final durable checkpoint.')
        json_new(output/'inventory.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('files','session_backup_sha256','complete_events')},indent=2))


if __name__=='__main__': main()
