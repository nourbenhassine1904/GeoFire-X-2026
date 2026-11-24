"""Copy the verified preparation ZIP to the requested local and D: roots."""
from pathlib import Path
import json
from geofirex import v2_execution as v2

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    receipt_path=ROOT/v2.REPORT/'preparation_backup_receipt.json'
    receipt=v2.read(receipt_path); archive=Path(receipt['local_archive'])
    local=ROOT/'artifacts/v2_execution/backups'; local.mkdir(parents=True,exist_ok=True)
    copies=v2.backup(archive,archive.with_suffix('.sha256'),local,Path('D:/GeoFire-X'))
    receipt.update(verified_copies=copies,external_copy_status='VERIFIED',
                   external_volume_expected='ADATA HV300 / 4A0E-CF27')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))
